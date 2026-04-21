"""Pipeline orchestration for 3D model georeferencing and metadata export."""

import os
import sys
import subprocess
import json
import logging
import shutil
import numpy as np
from typing import Optional
from PIL import Image
from pipeline.services import satelliteTileDownloader
from pipeline.georeferencing import georef_dim, DinoImageMatcher, OrthoCropper, GeoTransformer, MatrixUtils
from pipeline.utils.elevation import ElevationService
from pipeline.utils.coordinate_transforms import CoordinateTransforms

# Logger configuration
LOG_LEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from external libraries
for noisy_logger in (
    "google",
    "google.genai",
    "google.auth",
    "google.api_core",
    "urllib3",
    "requests",
    "httpx",
    "httpcore",
    "PIL",
):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING if LOG_LEVEL == "DEBUG" else logging.ERROR)


class PipelineProcessor:
    """
    Manages the 3D georeferencing pipeline.
    """

    BLENDER_SCRIPT_PATH = "/app/pipeline/rendering/multiview.py"
    DIM_SCRIPT_DIR = "/workspace/dim"
    DIM_SCRIPT_DEMO = "demo.py"
    DIM_SCRIPT_JOIN = "join_databases.py"

    def __init__(self, args):
        """
        Initialize the PipelineProcessor.

        Args:
            args: Parsed command-line arguments containing:
                - input_file (str): Path to input 3D model.
                - output_folder (str): Path to save results.
                - streetviews (int, optional): Number of streetview renderings.
        """
        self.args = args
        self.base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        self.working_dir = os.path.join("/tmp", self.base_name)

        os.makedirs(self.working_dir, exist_ok=True)

        # Optional API key override from CLI args
        if getattr(self.args, "gemini_api_key", None):
            os.environ["GEMINI_API_KEY"] = self.args.gemini_api_key
        if getattr(self.args, "mapbox_api_key", None):
            os.environ["MAPBOX_API_KEY"] = self.args.mapbox_api_key

        logger.info("Pipeline initialized.")
        logger.debug(f"  Input File   : {args.input_file}")
        logger.debug(f"  Working Dir  : {self.working_dir}")
        logger.debug(f"  Output Folder: {args.output_folder}")

    # -------------------------------------------------------------------------
    # STEP 1 – Synthetic views via Blender
    # -------------------------------------------------------------------------

    def generate_synthetic_views(self, streetviews: Optional[int] = None) -> tuple[bool, Optional[np.ndarray], Optional[list]]:
        """
        Launches Blender in background mode to:
          - Import the input 3D model
          - Render top-view orthographic image
          - Render street-view perspective images
          - Post-process the model (translate + scale)
          - Export *_scaled.glb
          - Save matrix_blender.json with the 4x4 transformation matrix

        Args:
            streetviews (int, optional): Number of street-view cameras around the model.

        Returns:
            tuple:
                - success flag
                - Blender 4x4 matrix (or None)
                - pivot position list (or None)
        """
        logger.info("🔧 [Step 1] Generating synthetic views with Blender...")

        command = [
            "blender", "-b",
            "--python", self.BLENDER_SCRIPT_PATH,
            "--",
            "--input_file", self.args.input_file,
            "--output_folder", self.working_dir,
        ]

        if streetviews is not None:
            command += ["--streetviews", str(streetviews)]

        logger.debug(f"Blender command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            if result.returncode not in (0, 1):
                logger.error(f"❌ Blender exited with unexpected code: {result.returncode}")
                return False, None

            # Parse matrix and pivot position from stdout (tagged lines)
            M_blender = None
            pivot_blender = None
            for line in result.stdout.splitlines():
                if line.startswith("MATRIX_BLENDER:"):
                    matrix_list = json.loads(line[len("MATRIX_BLENDER:"):])
                    M_blender = np.array(matrix_list, dtype=np.float64)
                if line.startswith("PIVOT_BLENDER:"):
                    pivot_blender = json.loads(line[len("PIVOT_BLENDER:"):])

            if M_blender is None:
                logger.error("❌ MATRIX_BLENDER tag not found in Blender stdout.")
                return False, None, None

            if pivot_blender is None:
                logger.warning("⚠️ PIVOT_BLENDER tag not found in Blender stdout.")

            logger.info("✅ Blender process finished.")
            return True, M_blender, pivot_blender

        except FileNotFoundError:
            logger.error("❌ 'blender' executable not found. Is Blender installed and in PATH?")
            return False, None, None
        except Exception as e:
            logger.error(f"❌ Error running Blender: {e}")
            return False, None, None

    # -------------------------------------------------------------------------
    # STEP 1b – Load Blender transformation matrix
    # -------------------------------------------------------------------------

    def _validate_blender_matrix(self, M: np.ndarray) -> Optional[np.ndarray]:
        """
        Validates the 4x4 transformation matrix received from Blender stdout.

        The matrix encodes the combined Translation x Scale applied to the mesh
        during Blender post-processing (postprocess_model in multiview.py):

            M_blender = Scale @ Translation

        where:
          - Translation aligns the bottom-left corner of the ortho camera to the origin
          - Scale brings the mesh width to match the rendered image resolution (px = units)

        Args:
            M (np.ndarray): Matrix to validate.

        Returns:
            np.ndarray: Validated 4x4 float64 matrix, or None if invalid.
        """
        if M.shape != (4, 4):
            logger.error(f"❌ Unexpected matrix shape: {M.shape}. Expected (4, 4).")
            return None

        logger.info("✅ Blender transformation matrix received successfully.")
        logger.debug(f"\n   Matrix (4x4):\n{M}\n")
        return M

    # -------------------------------------------------------------------------
    # STEP 2 – Dimension estimation + metric scaling
    # -------------------------------------------------------------------------

    def estimate_scene_dimension(self) -> Optional[float]:
        """
        Estimates the real-world width (in meters) of the scene captured in
        top_view.png using the GeminiDimensionEstimator.

        Returns:
            float: Estimated width in meters, or None on failure.
        """
        top_view_path = os.path.join(self.working_dir, "top_view.png")

        if not os.path.exists(top_view_path):
            logger.error(f"❌ top_view.png not found at: {top_view_path}")
            return None

        logger.info("📏 [Step 2a] Estimating scene dimension via Gemini...")
        try:
            from pipeline.geolocation.gemini import GeminiDimensionEstimator
            estimator = GeminiDimensionEstimator(model_name=self.args.gemini_model)
            dimension = estimator.estimate_dimension(top_view_path)

            if dimension is None:
                logger.error("❌ Gemini returned no dimension estimate.")
                return None

            logger.info(f"✅ Estimated scene width: {dimension:.2f} m")
            return dimension

        except Exception as e:
            logger.error(f"❌ Error during dimension estimation: {e}")
            return None

    def scale_3d_model(self, scale_factor: float) -> bool:
        """
        Applies a uniform metric scale to the *_scaled.glb model produced by
        Blender so that 1 unit = 1 metre in the output.

        The scale factor is:  dimension_meters / image_width_px
        (because after Blender post-processing 1 px == 1 Blender unit)

        Args:
            scale_factor (float): Scale to apply to the mesh.

        Returns:
            bool: True if successful, False otherwise.
        """
        import trimesh

        # Find *_scaled.glb in working dir
        target_file = None
        for f in os.listdir(self.working_dir):
            if f.endswith("_scaled.glb"):
                target_file = os.path.join(self.working_dir, f)
                break

        if not target_file:
            logger.error("❌ No *_scaled.glb file found in working dir.")
            return False

        logger.info(f"⚖️  [Step 2b] Scaling '{os.path.basename(target_file)}' by factor {scale_factor:.6f}...")
        try:
            scene = trimesh.load(target_file)
            matrix = trimesh.transformations.scale_matrix(scale_factor)

            if isinstance(scene, trimesh.Scene):
                for geom in scene.geometry.values():
                    geom.apply_transform(matrix)
            else:
                scene.apply_transform(matrix)

            scene.export(target_file)
            logger.info(f"✅ Model scaled and saved → {target_file}")
            return True

        except Exception as e:
            logger.error(f"❌ Error scaling 3D model: {e}")
            return False

    def _find_scaled_glb_path(self) -> Optional[str]:
        """Find the current *_scaled.glb path in working directory."""
        for f in sorted(os.listdir(self.working_dir)):
            if f.endswith("_scaled.glb"):
                return os.path.join(self.working_dir, f)
        return None

    def _extract_pivot_center_from_glb(self, glb_path: str) -> Optional[np.ndarray]:
        """
        Load a GLB scene and return the world-space center of the submesh named 'pivot'.

        Matching is case-insensitive and checks both scene node names and geometry names.
        """
        import trimesh

        try:
            scene = trimesh.load(glb_path, force="scene")
        except Exception as e:
            logger.error(f"❌ Failed to load GLB for pivot extraction: {e}")
            return None

        if not isinstance(scene, trimesh.Scene):
            logger.error("❌ Loaded GLB is not a Scene; cannot isolate submesh 'pivot'.")
            return None

        pivot_vertices = []
        for node_name in scene.graph.nodes_geometry:
            geom_name = scene.graph[node_name][1]
            node_l = str(node_name).lower()
            geom_l = str(geom_name).lower()

            if "pivot" not in node_l and "pivot" not in geom_l:
                continue

            geom = scene.geometry.get(geom_name)
            if geom is None:
                continue

            transform, _ = scene.graph.get(node_name)
            mesh_world = geom.copy()
            mesh_world.apply_transform(transform)
            pivot_vertices.append(np.asarray(mesh_world.vertices, dtype=np.float64))

        if not pivot_vertices:
            logger.error("❌ Submesh 'pivot' not found in *_scaled.glb scene graph.")
            return None

        verts = np.vstack(pivot_vertices)
        center = verts.mean(axis=0)
        return center

    @staticmethod
    def _epsg3857_to_latlon(x: float, y: float) -> Optional[tuple]:
        """Convert Web Mercator XY to geographic latitude/longitude (delegated to CoordinateTransforms)."""
        return CoordinateTransforms.epsg3857_to_latlon(x, y)

    def _compute_total_transform_matrix(
        self,
        blender_matrix: np.ndarray,
        metric_scale: float,
        transformer: GeoTransformer,
    ) -> Optional[np.ndarray]:
        """
        Compute the full transform applied to the original input GLB.

        Chain (must mirror current GeoTransformer behavior):
            M_total = R_x(-90) @ M_dim_trimesh @ S_metric @ M_blender
        """
        try:
            dim_matrix = MatrixUtils.load_matrix(os.path.join(self.working_dir, "transformation.txt"))
            m_dim_trimesh = transformer._dim_to_trimesh_matrix(dim_matrix)
            r_x_m90 = transformer._rotation_x_minus_90_matrix()

            s_metric = np.eye(4, dtype=np.float64)
            s_metric[0, 0] = s_metric[1, 1] = s_metric[2, 2] = float(metric_scale)

            return r_x_m90 @ m_dim_trimesh @ s_metric @ blender_matrix
        except Exception as e:
            logger.error(f"❌ Failed to compute total input->output transform: {e}")
            return None



    def _export_metadata_json(
        self,
        metadata: dict,
        filename: str = "heritage_data.json",
    ) -> Optional[str]:
        """Export metadata as a JSON file to output/<basename>/heritage_data.json."""
        import json

        out_dir = os.path.join(self.args.output_folder, self.base_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return out_path
        except Exception as e:
            logger.error(f"❌ Failed to write metadata JSON: {e}")
            return None

    # -------------------------------------------------------------------------
    # STEP 3 – Geolocation estimation
    # -------------------------------------------------------------------------

    def estimate_geolocation(self) -> Optional[tuple]:
        """
        Estimates the geographic coordinates (lat, lon) of the scene based on
        the rendered street-view images using the configured geolocation model.

        Analyzes images in working_dir to determine GPS location with high precision.

        Returns:
            tuple: (latitude, longitude) as floats, or None on failure.
        """
        geoloc_model = getattr(self.args, "geoloc_model", "gemini").lower()
        logger.info(f"📍 [Step 3] Estimating geolocation via {geoloc_model}...")

        try:
            if geoloc_model == "geoclip":
                from pipeline.geolocation.geoclip import GeoClipBatchPredictor

                top_k = int(getattr(self.args, "nr_prediction", 1))
                geolocator = GeoClipBatchPredictor(top_k=top_k)
                most_common, predictions = geolocator.predict_folder(self.working_dir)

            elif geoloc_model == "ollama":
                from pipeline.geolocation.ollama import ImageToCoordinates

                geolocator = ImageToCoordinates(ollama_model="llama3.2-vision")
                most_common, predictions = geolocator.run_pipeline(self.working_dir)

            else:
                from pipeline.geolocation.gemini import GeminiGeolocator

                geolocator = GeminiGeolocator(model_name=self.args.gemini_model)
                most_common, predictions = geolocator.run_pipeline(self.working_dir)

            if most_common is None:
                logger.error(f"❌ {geoloc_model} geolocation returned no predictions.")
                return None

            lat, lon = most_common
            logger.info(f"✅ Estimated coordinates: Lat={lat:.6f}, Lon={lon:.6f}")
            if predictions:
                logger.debug(f"   (from {len(predictions)} predictions)")
            return (float(lat), float(lon))

        except Exception as e:
            logger.error(f"❌ Error during geolocation estimation ({geoloc_model}): {e}")
            return None

    # -------------------------------------------------------------------------
    # STEP 4 – Mapbox satellite tiles download
    # -------------------------------------------------------------------------

    def download_satellite_tiles(self, lat: float, lon: float) -> bool:
        """
        Downloads satellite tiles from Mapbox and builds a GeoTIFF centered on
        the estimated coordinates.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.

        Returns:
            bool: True if download + merge succeed, False otherwise.
        """
        mapbox_key = os.getenv("MAPBOX_API_KEY")
        if not mapbox_key:
            logger.error("❌ MAPBOX_API_KEY not set. Cannot download satellite tiles.")
            return False

        area_size = int(getattr(self.args, "area_size", 500))
        zoom = int(getattr(self.args, "zoom", 18))

        logger.info("🛰️  [Step 4] Downloading Mapbox satellite tiles...")
        logger.debug(f"   center     : ({lat:.6f}, {lon:.6f})")
        logger.debug(f"   area_size  : {area_size} m")
        logger.debug(f"   zoom       : {zoom}")

        try:
            downloader = satelliteTileDownloader(
                center_lat=lat,
                center_lon=lon,
                area_size_m=area_size,
                zoom=zoom,
                output_folder=self.working_dir,
            )
            success = downloader.run_pipeline()
            if not success:
                logger.error("❌ Satellite tile download failed.")
                return False

            expected_tif = os.path.join(self.working_dir, f"{self.base_name}.tif")
            if os.path.exists(expected_tif):
                logger.info(f"✅ Satellite GeoTIFF ready: {expected_tif}")
            else:
                logger.warning("⚠️ Download reported success but GeoTIFF not found with expected name.")
            return True

        except Exception as e:
            logger.error(f"❌ Error downloading satellite tiles: {e}")
            return False

    # -------------------------------------------------------------------------
    # STEP 5 – DINO + crop + DIM
    # -------------------------------------------------------------------------

    def _run_command(self, command: list, cwd: Optional[str] = None) -> bool:
        """
        Runs a command quietly and logs errors only.
        """
        try:
            res = subprocess.run(
                command,
                cwd=cwd,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if res.returncode != 0:
                logger.error(f"❌ Command failed (code {res.returncode}): {' '.join(command)}")
                if res.stderr:
                    logger.error(res.stderr.strip())
                return False
            return True
        except Exception as e:
            logger.error(f"❌ Command execution error: {e}")
            return False

    def _prepare_dim_images_folder(self) -> Optional[tuple]:
        """
        Prepares /images folder with ortho + top_view for DIM.

        Returns:
            tuple: (images_dir, ortho_path, render_path) or None if missing files.
        """
        images_dir = os.path.join(self.working_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        ortho_src = os.path.join(self.working_dir, f"{self.base_name}.tif")
        render_src = os.path.join(self.working_dir, "top_view.png")

        if not os.path.exists(ortho_src):
            logger.error(f"❌ Orthophoto not found: {ortho_src}")
            return None
        if not os.path.exists(render_src):
            logger.error(f"❌ Render not found: {render_src}")
            return None

        ortho_path = os.path.join(images_dir, f"{self.base_name}.tif")
        render_path = os.path.join(images_dir, "top_view.png")

        shutil.copy2(ortho_src, ortho_path)
        shutil.copy2(render_src, render_path)

        return images_dir, ortho_path, render_path

    def _run_dino_and_crop(self, ortho_path: str, render_path: str) -> bool:
        """
        Runs DINO matching and crops orthophoto around matched center if successful.
        If DINO fails, pipeline continues with original ortho.
        """
        logger.info("🦕 [Step 5a] Running DINO matcher...")
        try:
            dino = DinoImageMatcher(dino_version="v2", downscale_factor=1.0)
            dino_output = os.path.join(self.working_dir, "dino_results")
            center_row, center_col = dino.match(
                base_image_path=ortho_path,
                template_image_path=render_path,
                output_dir=dino_output,
            )

            if center_row is None or center_col is None:
                logger.warning("⚠️ DINO returned no center. Skipping crop.")
                return True

            logger.info(f"✅ DINO center: row={center_row:.2f}, col={center_col:.2f}")
            logger.info("✂️ [Step 5b] Cropping orthophoto around DINO center...")

            cropper = OrthoCropper(scale_factor=2.0)
            cropped_path = os.path.join(os.path.dirname(ortho_path), f"{self.base_name}_cropped.tif")
            out = cropper.crop(
                ortho_path=ortho_path,
                reference_image_path=render_path,
                center_row=center_row,
                center_col=center_col,
                output_path=cropped_path,
            )

            if out and os.path.exists(out):
                os.remove(ortho_path)
                shutil.move(out, ortho_path)
                logger.info("✅ Orthophoto cropped and replaced.")
            else:
                logger.warning("⚠️ Crop output missing. Keeping original orthophoto.")

            return True

        except Exception as e:
            logger.warning(f"⚠️ DINO/Crop failed, continuing with original ortho: {e}")
            return True

    def _rotate_image_variants(self, file_path: str):
        """
        Creates 90, 180, 270 degree rotated copies of the image.
        """
        if not os.path.exists(file_path):
            return

        try:
            img = Image.open(file_path)
            folder, filename = os.path.split(file_path)
            name, ext = os.path.splitext(filename)

            for angle in [90, 180, 270]:
                rotated_img = img.rotate(-angle, expand=True)  # negative = clockwise
                output_path = os.path.join(folder, f"{name}_rot{angle}{ext}")
                rotated_img.save(output_path)
        except Exception as e:
            logger.error(f"⚠️ Error creating rotated variants for {file_path}: {e}")

    def _create_scaled_variants(self, image_path: str):
        """
        Creates scaled versions (25%, 50%, 75%) of the orthophoto.
        """
        if not os.path.exists(image_path):
            return

        try:
            img = Image.open(image_path)
            base_dir, filename = os.path.split(image_path)
            name, ext = os.path.splitext(filename)

            scales = [(0.25, "_s_0_25"), (0.5, "_s_0_50"), (0.75, "_s_0_75")]

            for scale, suffix in scales:
                new_size = (int(img.width * scale), int(img.height * scale))
                resized = img.resize(new_size, Image.Resampling.LANCZOS)
                new_filename = os.path.join(base_dir, f"{name}{suffix}{ext}")
                resized.save(new_filename, format="TIFF")
        except Exception as e:
            logger.error(f"⚠️ Error creating scaled variants for {image_path}: {e}")

    def _validate_image_variants(self, ortho_path: str, render_path: str) -> bool:
        """
        Validates that required rotated/scaled image variants exist before DIM.
        """
        render_base, render_ext = os.path.splitext(render_path)
        ortho_base, ortho_ext = os.path.splitext(ortho_path)

        required = [
            f"{render_base}_rot90{render_ext}",
            f"{render_base}_rot180{render_ext}",
            f"{render_base}_rot270{render_ext}",
            f"{ortho_base}_s_0_25{ortho_ext}",
            f"{ortho_base}_s_0_50{ortho_ext}",
            f"{ortho_base}_s_0_75{ortho_ext}",
        ]

        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            logger.error("❌ Missing image variants required by DIM:")
            for m in missing:
                logger.error(f"   - {m}")
            return False
        return True

    def _run_dim_and_georef(self, ortho_path: str, render_path: str) -> bool:
        """
        Runs DIM (LoFTR + SuperPoint/SuperGlue), merges DBs, and computes georef matrix.
        """
        logger.info("🧩 [Step 5c] Running DIM pipelines...")

        def run_dim(pair_type: str) -> bool:
            cmd = [
                "python3", self.DIM_SCRIPT_DEMO,
                "-p", pair_type,
                "-t", "none",
                "-s", "bruteforce",
                "--force",
                "--skip_reconstruction",
                "-q", "high",
                "-V",
                "-d", self.working_dir,
            ]
            return self._run_command(cmd, cwd=self.DIM_SCRIPT_DIR)

        ok_loftr = run_dim("loftr")
        ok_sp = run_dim("superpoint+superglue")
        if not (ok_loftr and ok_sp):
            logger.error("❌ DIM failed on one or more matchers.")
            return False

        merge_db_path = os.path.join(self.working_dir, "merge_db")
        os.makedirs(merge_db_path, exist_ok=True)

        db_loftr = os.path.join(self.working_dir, "results_loftr_bruteforce_quality_high", "database.db")
        db_sp = os.path.join(self.working_dir, "results_superpoint+superglue_bruteforce_quality_high", "database.db")

        if os.path.exists(db_loftr):
            shutil.copy2(db_loftr, os.path.join(merge_db_path, "database_loftr.db"))
        if os.path.exists(db_sp):
            shutil.copy2(db_sp, os.path.join(merge_db_path, "database_superpoint.db"))

        join_cmd = [
            "python3", self.DIM_SCRIPT_JOIN,
            "-i", merge_db_path,
            "-o", self.working_dir,
        ]
        if not self._run_command(join_cmd, cwd=os.path.join(self.DIM_SCRIPT_DIR, "scripts")):
            logger.error("❌ Failed to join DIM databases.")
            return False

        logger.info("🧭 [Step 5d] Computing georeferencing transform...")
        try:
            output_matrix_path = os.path.join(self.working_dir, "transformation.txt")
            joined_db = os.path.join(self.working_dir, "joined.db")
            processor = georef_dim(ortho_path, render_path, output_matrix_path, joined_db, debug=False)
            processor.run_pipeline()
            logger.info(f"✅ Georeferencing matrix ready: {output_matrix_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Georeferencing calculation failed: {e}")
            return False

    def run_step5_dino_crop_dim(self) -> bool:
        """
        Full Step 5 orchestration:
          1) Prepare images folder
          2) DINO matching
          3) Optional orthophoto crop
          4) DIM + DB join + georef matrix
        """
        prep = self._prepare_dim_images_folder()
        if prep is None:
            return False
        _, ortho_path, render_path = prep

        if not self._run_dino_and_crop(ortho_path, render_path):
            return False

        # Prepare image variants before DIM
        self._rotate_image_variants(render_path)
        self._create_scaled_variants(ortho_path)

        if not self._validate_image_variants(ortho_path, render_path):
            return False

        return self._run_dim_and_georef(ortho_path, render_path)

    # -------------------------------------------------------------------------
    # PIPELINE ENTRY POINT
    # -------------------------------------------------------------------------

    def run_pipeline(self) -> bool:
        """
        Main execution method for the pipeline.

        Returns:
            bool: True if successful, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("🚀 Starting Pipeline Execution")
        logger.info("=" * 60)

        # ------------------------------------------------------------------
        # STEP 1: Run Blender – synthetic views + model post-processing
        # ------------------------------------------------------------------
        streetviews = int(getattr(self.args, "streetviews", 3))
        ok, M_raw, _pivot_blender = self.generate_synthetic_views(streetviews=streetviews)

        if not ok:
            logger.error("⛔ Pipeline interrupted at Step 1 (Blender).")
            return False

        # ------------------------------------------------------------------
        # STEP 1b: Validate the Blender transformation matrix
        # ------------------------------------------------------------------
        M_blender = self._validate_blender_matrix(M_raw)

        if M_blender is None:
            logger.error("⛔ Pipeline interrupted: invalid Blender matrix.")
            return False

        # ------------------------------------------------------------------
        # STEP 2: Dimension estimation + metric scaling of the 3D model
        # ------------------------------------------------------------------
        dimension_m = self.estimate_scene_dimension()

        if dimension_m is None:
            logger.error("⛔ Pipeline interrupted at Step 2 (Dimension Estimation).")
            return False

        # After Blender post-processing: 1 Blender unit == 1 pixel of top_view.png.
        # We need 1 unit == 1 metre, so the scale factor is metres / pixels.
        top_view_path = os.path.join(self.working_dir, "top_view.png")
        with Image.open(top_view_path) as img:
            image_width_px = img.width

        metric_scale = dimension_m / image_width_px
        logger.info(f"✅ Metric scale computed: {metric_scale:.6f} m/px")
        logger.debug(f"   image width : {image_width_px} px")
        logger.debug(f"   scene width : {dimension_m:.2f} m")

        if not self.scale_3d_model(metric_scale):
            logger.error("⛔ Pipeline interrupted at Step 2 (Model Scaling).")
            return False

        # Resize top_view.png so that its width == dimension_m pixels (1 px = 1 m)
        target_width_px = int(round(dimension_m))
        with Image.open(top_view_path) as img:
            orig_w, orig_h = img.width, img.height
            target_height_px = int(round(orig_h * target_width_px / orig_w))
            resized = img.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)
            resized.save(top_view_path)
        logger.debug(f"✅ top_view.png resized: {orig_w}x{orig_h} → {target_width_px}x{target_height_px} px")

        # Store for downstream steps
        self.metric_scale = metric_scale

        # ------------------------------------------------------------------
        # STEP 3: Geolocation estimation
        # ------------------------------------------------------------------
        coords = self.estimate_geolocation()

        if coords is None:
            logger.error("⛔ Pipeline interrupted at Step 3 (Geolocation).") 
            return False

        self.latitude, self.longitude = coords

        # ------------------------------------------------------------------
        # STEP 4: Download satellite tiles from Mapbox
        # ------------------------------------------------------------------
        if not self.download_satellite_tiles(self.latitude, self.longitude):
            logger.error("⛔ Pipeline interrupted at Step 4 (Mapbox Tiles Download).")
            return False

        # ------------------------------------------------------------------
        # STEP 5: DINO + crop orthophoto + DIM
        # ------------------------------------------------------------------
        if not self.run_step5_dino_crop_dim():
            logger.error("⛔ Pipeline interrupted at Step 5 (DINO/Crop/DIM).")
            return False

        # ------------------------------------------------------------------
        # STEP 6: Apply DIM transform to *_scaled.glb (overwrite)
        # ------------------------------------------------------------------
        logger.info("🔧 [Step 6] Applying DIM transform to scaled mesh...")
        transformer = GeoTransformer(
            working_dir=self.working_dir,
            input_file=self.args.input_file,
            output_folder=self.args.output_folder,
            lat=self.latitude,
            lon=self.longitude,
            pipeline_scale=self.metric_scale,
        )
        if not transformer.run():
            logger.error("⛔ Pipeline interrupted at Step 6 (GeoTransformer).")
            return False

        # ------------------------------------------------------------------
        # STEP 6b: Read pivot submesh center from transformed *_scaled.glb
        # ------------------------------------------------------------------
        logger.info("📍 [Step 6b] Reading 'pivot' submesh center from *_scaled.glb...")
        glb_path = self._find_scaled_glb_path()
        if not glb_path:
            logger.error("❌ *_scaled.glb not found for pivot center extraction.")
            return False

        pivot_center = self._extract_pivot_center_from_glb(glb_path)
        if pivot_center is None:
            return False

        pivot_x, pivot_y, pivot_z = float(pivot_center[0]), float(pivot_center[1]), float(pivot_center[2])
        logger.debug(
            f"✅ Pivot center from GLB: X={pivot_x:.6f}, Y={pivot_y:.6f}, Z={pivot_z:.6f}"
        )

        latlon = self._epsg3857_to_latlon(pivot_x, pivot_y)
        if latlon is None:
            return False
        pivot_lat, pivot_lon = latlon
        logger.debug(f"✅ Pivot geographic coordinates: LAT={pivot_lat:.8f}, LON={pivot_lon:.8f}")

        pivot_height = ElevationService.get_elevation(pivot_lat, pivot_lon)
        if pivot_height is None:
            logger.error("❌ Cannot export metadata.xlsx without pivot elevation.")
            return False
        logger.debug(f"✅ Pivot height from LAT/LON: H={pivot_height:.3f} m")

        # ------------------------------------------------------------------
        # STEP 6c: Export metadata.xlsx with transforms for the input GLB
        # ------------------------------------------------------------------
        logger.info("🧾 [Step 6c] Exporting metadata.xlsx (rotation + scale for input GLB)...")
        m_total = self._compute_total_transform_matrix(
            blender_matrix=M_blender,
            metric_scale=self.metric_scale,
            transformer=transformer,
        )
        if m_total is None:
            return False

        try:
            trs = MatrixUtils.decompose_trs(m_total)
            r = trs["rotation_euler_deg_xyz"]
            s = trs["scale"]
        except Exception as e:
            logger.error(f"❌ Failed to decompose total matrix into TRS: {e}")
            return False

        # Use averaged Y scale to stabilize anisotropic scale in exported metadata.
        scale_x = float(s[0])
        scale_z = float(s[2])
        scale_y = (scale_x + scale_z) / 2.0
        logger.debug(
            f"Scale override for metadata export: sy=avg(sx,sz) => {scale_y:.6f} (sx={scale_x:.6f}, sz={scale_z:.6f})"
        )

        # UH4D Browser convention: Y axis is up, Euler order is YXZ.
        # Keep heading from internal Z and export as rotation around Y.
        viewer_rot_x = 0.0
        viewer_rot_y = float(r[2])
        viewer_rot_z = 0.0
        logger.debug(
            "UH4D rotation mapping (Y-up, YXZ): "
            f"internal XYZ=({float(r[0]):.6f},{float(r[1]):.6f},{float(r[2]):.6f}) -> "
            f"viewer=({viewer_rot_x:.6f},{viewer_rot_y:.6f},{viewer_rot_z:.6f})"
        )

        metadata = {
            "latitude": float(pivot_lat),
            "longitude": float(pivot_lon),
            "height": float(pivot_height),
            "place": "",
            "description": "",
            "scale": [scale_x, scale_y, scale_z],
            "rotation": [viewer_rot_x, viewer_rot_y, viewer_rot_z],
            "translation": [0.0, 0.0, 0.0],
        }

        json_path = self._export_metadata_json(metadata)
        if json_path is None:
            return False
        logger.info(f"✅ Metadata JSON exported: {json_path}")

        logger.info("⏹️  Pipeline completed successfully.")
        logger.debug(f"   M_blender stored  | shape={M_blender.shape}")
        logger.debug(f"   metric_scale      | {metric_scale:.6f} m/px")
        logger.debug(f"   coordinates       | ({self.latitude:.6f}, {self.longitude:.6f})")
        logger.debug(f"   ortho_map         | {os.path.join(self.working_dir, self.base_name + '.tif')}")
        logger.debug(f"   transform_matrix  | {os.path.join(self.working_dir, 'transformation.txt')}")
        sys.exit(0)
