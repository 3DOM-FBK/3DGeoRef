import os
import sys
import subprocess
import shutil
import gc
import logging
from typing import Optional, Tuple, Union

import trimesh
from PIL import Image

# Ensure the pipeline directory is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from pipeline.services import satelliteTileDownloader
from pipeline.georeferencing import georef_dim, GeoTransformer, ElevationService

# Logger configuration
LOG_LEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL), 
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class PipelineProcessor:
    """
    Manages the 3D georeferencing pipeline, including synthetic view generation,
    geolocation estimation, scene dimension scaling, and orthophoto processing.
    """

    # Constants for external script paths - Adjust these if execution environment changes
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
                - mode (str): Pipeline mode ('auto', 'geoloc', 'dim').
                - lat (float, optional): Latitude override.
                - lon (float, optional): Longitude override.
                - geoloc_model (str): 'gemini', 'geoclip', or 'ollama'.
                - nr_prediction (int): Number of predictions for geoloc.
                - area_size (float): Satellite area size.
                - zoom (int): Satellite zoom level.
                - ortho (str, optional): Path to user-provided orthophoto.
        """
        self.args = args
        self.base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        # Use a consistent temporary directory structure
        self.working_dir = os.path.join("/tmp", self.base_name)
        
        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)
        
        logger.info("Pipeline initialized.")
        logger.info(f"  Input File: {args.input_file}")
        logger.info(f"  Working Directory: {self.working_dir}")
        logger.info(f"  Log Level: {LOG_LEVEL}")

    def _free_memory(self):
        """
        Frees up memory by running garbage collection and clearing GPU cache if available.
        """
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("🧹 GPU cache cleared.")
        except ImportError:
            pass # Torch not installed or not needed
        logger.debug("🧹 Memory freed.")

    def _run_command(self, command: list, cwd: str = None, capture_output: bool = True) -> bool:
        """
        Helper to run subprocess commands with logging.
        
        Args:
            command: List of command strings.
            cwd: Working directory for the command.
            capture_output: Whether to capture stdout/stderr.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        cmd_str = " ".join(command)
        logger.debug(f"Executing: {cmd_str}")
        try:
            if capture_output:
                subprocess.run(command, check=True, capture_output=True, text=True, cwd=cwd)
            else:
                # Direct output to DEVNULL to suppress noise if requested
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=cwd)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Command failed: {cmd_str}")
            if e.stdout:
                logger.error(f"  Stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"  Stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"❌ Execution error: {e}")
            return False

    def generate_synthetic_views(self, streetviews: Optional[int] = None) -> bool:
        """
        Generates synthetic views from the input 3D model using Blender.
        """
        logger.info("🔧 Generating synthetic views...")
        
        command = [
            "blender", "-b",
            "--python", self.BLENDER_SCRIPT_PATH,
            "--", 
            "--input_file", self.args.input_file,
            "--output_folder", self.working_dir
        ]
        
        if streetviews:
            command += ["--streetviews", str(streetviews)]

        # Run without capturing output (silence blender noise), only catch errors
        return self._run_command(command, capture_output=False)

    def estimate_scene_dimension(self) -> Optional[float]:
        """
        Estimates the real-world dimension (in meters) of the scene using the top-view image.
        Uses lazy import for Gemini dependencies.
        """
        logger.info("📏 Estimating scene dimensions using Gemini...")
        top_view_path = os.path.join(self.working_dir, "top_view.png")
        
        if not os.path.exists(top_view_path):
            logger.warning(f"⚠️ Top view image not found at {top_view_path}. Skipping dimension estimation.")
            return None

        try:
            from pipeline.geolocation.gemini import GeminiDimensionEstimator
            estimator = GeminiDimensionEstimator()
            dimension = estimator.estimate_dimension(top_view_path)
            
            if dimension:
                logger.info(f"✅ Estimated dimension: {dimension} meters")
                return dimension
            else:
                logger.warning("⚠️ Could not estimate dimension.")
                return None
        except Exception as e:
            logger.error(f"❌ Error estimating dimension: {e}")
            return None

    def resize_image_to_dimension(self, image_name: str, dimension: float) -> Optional[float]:
        """
        Resizes the image such that its width matches the estimated dimension (1 px = 1 m).
        Returns the scale factor applied (target_width / original_width).
        """
        image_path = os.path.join(self.working_dir, image_name)
        if not os.path.exists(image_path):
             logger.warning(f"⚠️ Image {image_name} not found, cannot resize.")
             return None
        
        try:
            with Image.open(image_path) as img:
                original_width, original_height = img.size
                target_width = int(dimension)
                
                # Aspect ratio preservation
                aspect_ratio = original_height / original_width
                target_height = int(target_width * aspect_ratio)

                logger.info(f"🔄 Resizing {image_name} to {target_width}x{target_height} px (Width: {dimension}m)")
                
                resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                resized.save(image_path)
                
                scale_factor = target_width / original_width
                return scale_factor

        except Exception as e:
            logger.error(f"❌ Error resizing {image_name}: {e}")
            return None

    def scale_3d_model(self, scale_factor: float):
        """
        Scales the 3D model (ending in '_scaled.glb') by the given factor.
        """
        target_file = None
        
        # Find the scaled model file in working dir
        if os.path.exists(self.working_dir):
            for f in os.listdir(self.working_dir):
                if f.endswith("_scaled.glb"):
                    target_file = os.path.join(self.working_dir, f)
                    break
        
        # Fallback to input file path if relevant
        if not target_file and self.args.input_file.endswith("_scaled.glb") and os.path.exists(self.args.input_file):
            target_file = self.args.input_file

        if not target_file:
            logger.warning("⚠️ No model ending with '_scaled.glb' found to scale.")
            return

        logger.info(f"⚖️ Scaling 3D model {os.path.basename(target_file)} by factor {scale_factor:.4f}...")
        
        try:
            scene = trimesh.load(target_file)
            matrix = trimesh.transformations.scale_matrix(scale_factor)
            
            if isinstance(scene, trimesh.Scene):
                for geom in scene.geometry.values():
                    geom.apply_transform(matrix)
            else:
                scene.apply_transform(matrix)
                
            scene.export(target_file)
            logger.info("✅ Model scaled and saved.")
        except Exception as e:
            logger.error(f"❌ Error scaling 3D model: {e}")

    def _estimate_geoloc_generic(self, geoloc_type: str, nr_prediction: int) -> Tuple[float, float]:
        """
        Generic handler for different geolocation strategies (GeoCLIP, Ollama, Gemini).
        """
        logger.info(f"📍 Estimating geolocation using {geoloc_type}...")
        
        predictor = None
        try:
            if geoloc_type == "geoclip":
                from pipeline.geolocation.geoclip import GeoClipBatchPredictor
                predictor = GeoClipBatchPredictor(top_k=int(nr_prediction))
                most_common, _ = predictor.predict_folder(self.working_dir)
                
            elif geoloc_type == "ollama":
                from pipeline.geolocation.ollama import ImageToCoordinates
                predictor = ImageToCoordinates(ollama_model="llama3.2-vision")
                most_common, _ = predictor.run_pipeline(self.working_dir)
                
            else: # gemini
                from pipeline.geolocation.gemini import GeminiGeolocator
                predictor = GeminiGeolocator()
                most_common, _ = predictor.run_pipeline(self.working_dir)

            # Cleanup
            del predictor
            self._free_memory()

            if not most_common:
                raise RuntimeError("No predictions returned.")

            lat, lon = most_common
            logger.info(f"✅ Estimated coordinates ({geoloc_type}): Latitude={lat}, Longitude={lon}")
            return float(lat), float(lon)

        except Exception as e:
            logger.error(f"❌ Error estimating geolocation ({geoloc_type}): {e}")
            raise

    def download_satellite_imagery(self, lat: float, lon: float) -> bool:
        """
        Downloads satellite imagery using the TileDownloader service.
        """
        if not self.args.area_size or not self.args.zoom:
            logger.warning("⚠️ Missing area_size or zoom for satellite download.")
            return False

        logger.info("🛰️ Downloading satellite imagery...")
        try:
            downloader = satelliteTileDownloader(
                lat, lon, self.args.area_size, self.args.zoom, self.working_dir
            )
            return downloader.run_pipeline()
        except Exception as e:
            logger.error(f"❌ Error downloading satellite imagery: {e}")
            return False

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
                rotated_img = img.rotate(-angle, expand=True) # Negative for clockwise
                output_path = os.path.join(folder, f"{name}_rot{angle}{ext}")
                rotated_img.save(output_path)
        except Exception as e:
            logger.error(f"⚠️ Error creating rotated variants for {file_path}: {e}")

    def _create_scaled_variants(self, image_path: str):
        """
        Creates scaled versions (25%, 50%, 75%) of the image.
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

    def run_deep_image_matching_and_georef(self):
        """
        Executes Deep-Image-Matching (DIM) and georeferencing.
        """
        logger.info("🔧 Running Deep-Image-Matching & Georeferencing...")

        ortho_path = os.path.join(self.working_dir, "images", self.base_name + ".tif")
        render_path = os.path.join(self.working_dir, "images", "top_view.png")
        output_matrix_path = os.path.join(self.working_dir, "transformation.txt")

        # Prepare image variants
        self._rotate_image_variants(render_path)
        self._create_scaled_variants(ortho_path)

        # Helper to run DIM for a specific pair type
        def run_dim(pair_type: str):
            cmd = [
                "python3", self.DIM_SCRIPT_DEMO,
                "-p", pair_type,
                "-t", "none",
                "-s", "bruteforce",
                "--force",
                "--skip_reconstruction",
                "-q", "high",
                "-V",
                "-d", self.working_dir
            ]
            # Assumes DIM_SCRIPT_DIR is a valid path where DIM is installed
            self._run_command(cmd, cwd=self.DIM_SCRIPT_DIR)

        # 1. Run DIM with LoFTR
        run_dim("loftr")
        
        # 2. Run DIM with SuperPoint+SuperGlue
        run_dim("superpoint+superglue")

        # 3. Merge Databases
        merge_db_path = os.path.join(self.working_dir, "merge_db")
        os.makedirs(merge_db_path, exist_ok=True)
        
        db_loftr = os.path.join(self.working_dir, "results_loftr_bruteforce_quality_high", "database.db")
        db_sp = os.path.join(self.working_dir, "results_superpoint+superglue_bruteforce_quality_high", "database.db")

        if os.path.exists(db_loftr):
            shutil.copy(db_loftr, os.path.join(merge_db_path, "database_loftr.db"))
        if os.path.exists(db_sp):
            shutil.copy(db_sp, os.path.join(merge_db_path, "database_superpoint.db"))

        join_cmd = [
            "python3", self.DIM_SCRIPT_JOIN,
            "-i", merge_db_path,
            "-o", self.working_dir
        ]
        # Assumes join script is in the scripts subdirectory of DIM_SCRIPT_DIR
        self._run_command(join_cmd, cwd=os.path.join(self.DIM_SCRIPT_DIR, "scripts"))

        # 4. Compute Georeferencing Matrix
        try:
            joined_db = os.path.join(self.working_dir, "joined.db")
            processor = georef_dim(ortho_path, render_path, output_matrix_path, joined_db, debug=False)
            processor.run_pipeline()
        except Exception as e:
            logger.error(f"❌ Georeferencing calculation failed: {e}")

    def move_images_to_subfolder(self, ortho_map: Optional[str] = None):
        """
        Organizes images into an 'images' subfolder.
        """
        images_dir = os.path.join(self.working_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        for file_name in os.listdir(self.working_dir):
            if file_name == "top_view.png" or file_name.lower().endswith(".tif"):
                src = os.path.join(self.working_dir, file_name)
                dst = os.path.join(images_dir, file_name)
                try:
                    shutil.move(src, dst)
                except Exception as e:
                    logger.warning(f"  Failed to move {file_name}: {e}")
        
        if ortho_map:
            try:
                shutil.copy(ortho_map, os.path.join(images_dir, os.path.basename(ortho_map)))
            except Exception as e:
                logger.error(f"  Failed to copy user ortho map: {e}")

    def _finalize_output(self):
        """
        Handles copying of final results and backing up the working directory.
        """
        # Copy Orthophoto
        ortho_src = os.path.join(self.working_dir, "images", self.base_name + ".tif")
        if os.path.exists(ortho_src):
            try:
                dest = os.path.join(self.args.output_folder, self.base_name, self.base_name + ".tif")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(ortho_src, dest)
                logger.info(f"✅ Orthophoto saved to: {dest}")
            except Exception as e:
                logger.error(f"❌ Failed to copy orthophoto: {e}")

        # Backup Working Directory - Temporary
        try:
            backup_path = os.path.join(self.args.output_folder, self.base_name, "working_dir_backup")
            if os.path.exists(self.working_dir):
                shutil.copytree(self.working_dir, backup_path, dirs_exist_ok=True)
                logger.info(f"✅ Temporary working directory backed up to: {backup_path}")
        except Exception as e:
            logger.error(f"❌ Failed to backup temporary directory: {e}")

    def run_pipeline(self) -> bool:
        """
        Main execution method for the pipeline.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        logger.info("🚀 Starting Pipeline Execution")
        
        # Check API keys
        mapbox_key = os.getenv("MAPBOX_API_KEY")
        if not mapbox_key:
            logger.warning("⚠️ MAPBOX_API_KEY missing. Satellite download will fail.")
        if not os.getenv("GEMINI_API_KEY"):
            logger.warning("⚠️ GEMINI_API_KEY missing. Geolocation may fail.")

        mode = getattr(self.args, "mode", "auto")
        
        # --- Step 1: Synthetic Views ---
        if mode in ("auto", "geoloc", "dim"):
            # generate_synthetic_views handles logging, we just check return
            if not self.generate_synthetic_views(streetviews=3):
                logger.error("⛔ Pipeline interrupted at Step 1 (Synthetic Views).")
                return False

        # --- Step 2: Dimension Estimation ---
        if mode in ("auto", "geoloc", "dim"):
            estimated_dim = self.estimate_scene_dimension()
            if estimated_dim:
                scale_factor = self.resize_image_to_dimension("top_view.png", estimated_dim)
                if scale_factor:
                    self.scale_3d_model(scale_factor)
            else:
                logger.error("⛔ Pipeline interrupted at Step 2 (Dimension Estimation).")
                return False

        # --- Step 3: Geolocation ---
        lat, lon = self.args.lat, self.args.lon
        
        if mode in ("auto", "geoloc") and (lat is None or lon is None):
            geoloc_model = getattr(self.args, "geoloc_model", "gemini").lower()
            try:
                lat, lon = self._estimate_geoloc_generic(geoloc_model, self.args.nr_prediction)
            except Exception:
                logger.error("⛔ Pipeline interrupted at Step 3 (Geolocation).")
                return False
                
        elif mode == "dim" and (lat is None or lon is None):
            logger.error("❌ Mode 'dim' requires valid 'lat' and 'lon' arguments.")
            return False

        # --- Step 4: Elevation ---
        elevation = 0.0
        try:
            elevation = ElevationService.get_elevation(lat, lon)
            logger.info(f"🏔️  Elevation at ({lat}, {lon}): {elevation}m")
        except Exception as e:
            logger.error(f"❌ Failed to get elevation: {e}")
            logger.error("⛔ Pipeline interrupted at Step 4.")
            return False

        # Return early if only geolocation was requested by the actual Pipeline mode
        # (Though 'geoloc' mode usually still falls through if you proceed logic wise, 
        # previous code did return here. We'll stick to that behavior)
        if mode == "geoloc":
            return lat, lon

        # --- Step 5: Ortho / Satellite Imagery & Transformation ---
        ortho_provided = getattr(self.args, "ortho", None)
        success_img = False
        
        if ortho_provided:
            logger.info("--> Using user-provided ortho images")
            self.move_images_to_subfolder(ortho_provided)
            self.run_deep_image_matching_and_georef()
            success_img = True
            
        elif mapbox_key:
            if self.download_satellite_imagery(lat, lon):
                self.move_images_to_subfolder()
                self.run_deep_image_matching_and_georef()
                success_img = True
            else:
                logger.error("❌ Failed to download satellite imagery.")
        
        else:
             logger.warning("⚠️ No ortho image and no Mapbox key. Skipping DIM and Georef.")

        if success_img:
            # Apply final transform
            try:
                gt = GeoTransformer(self.working_dir, self.args.input_file, self.args.output_folder, lat, lon)
                gt.run()
            except Exception as e:
                logger.error(f"❌ Failed to apply GeoTransformer: {e}")
                return False

        # --- Step 6: Finalize / Backup ---
        self._finalize_output()
        
        logger.info("✅ Pipeline completed successfully.")
        return True