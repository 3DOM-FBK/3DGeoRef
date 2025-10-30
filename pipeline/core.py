import os
import sys
import subprocess
import shutil
import trimesh
import numpy as np
import logging
import requests
import json
from PIL import Image
from pyproj import Transformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from geoloc_geoclip import GeoClipBatchPredictor
from geoloc_ollama import ImageToCoordinates
from geoloc_geminiAI import GeminiGeolocator
from tiles_to_geotiff import satelliteTileDownloader
from georef_dim import georef_dim


# ===== Logger configuration =====
log_level = os.environ.get("LOGLEVEL", "INFO").upper()
log_format = '%(asctime)s - %(levelname)-8s - %(message)s'
logging.basicConfig(
    level=getattr(logging, log_level), 
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("Pipeline start")
logger.info(f"Log level set to: {log_level}")



class PipelineProcessor:
    # ===== Function: __init__ =====
    def __init__(self, args):
        self.args = args

        self.base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        self.working_dir = os.path.join("/tmp", self.base_name)

        os.makedirs(self.working_dir, exist_ok=True)


    # ===== Function: clear_tmp_directory =====
    def clear_tmp_directory(self, path="/tmp"):
        """
        Clears the contents of the specified directory by removing all files, symbolic links, and subdirectories.
        If the directory does not exist, an error is logged. Any exceptions encountered during deletion are
        caught and logged without interrupting execution.

        Args:
            path (str): The path of the directory to clear. Defaults to "/tmp".
        """
        if not os.path.exists(path):
            logger.error(f"  ❌ Folder {path} doesn't exist.")
            return

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"  ⚠️ Error during deliting file {file_path}: {e}")


    # ===== Function: generate_synthetic_views =====
    def generate_synthetic_views(self, input_file, streetviews=None):
        """
        Generates synthetic views from a given input file using Blender. Optionally, street view images can be
        provided to influence the generation. Executes a Blender script in background mode and logs any errors
        encountered during execution.

        Args:
            input_file (str): Path to the input file used for generating synthetic views.
            streetviews (optional, str or int): Path or identifier for street view images to incorporate in the generation. Defaults to None.
        """
        logger.info("🔧 Generating synthetic views...")

        # Base command
        blender_cmd = [
            "blender", "-b",
            "--python", "/app/pipeline/synthetic_imgs.py",
            "--", "--input_file", input_file,
            "--output_folder", self.working_dir
        ]

        # Add optional argument
        if streetviews:
            blender_cmd += ["--streetviews", str(streetviews)]

        try:
            subprocess.run(blender_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # subprocess.run(blender_cmd)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error during Blender execution: {e}")


    # ===== Function: estimate_geolocation =====
    def estimate_geoloc_geoclip(self, nr_prediction):
        """
        Estimates geolocation (latitude and longitude) using GeoCLIP directly,
        based on all images in the current working directory.

        Args:
            nr_prediction (int): Number of GPS predictions to consider for each image.

        Returns:
            tuple: (latitude, longitude) as floats.
        """
        logger.info("📍 Estimating geolocation using GeoCLIP...")

        try:
            # Create the batch predictor
            predictor = GeoClipBatchPredictor(top_k=int(nr_prediction))

            # Run predictions on the working directory
            most_common, counter = predictor.predict_folder(self.working_dir)

            if not most_common:
                logger.error("❌ No geolocation predictions could be computed.")
                raise RuntimeError("Failed to estimate geolocation from images.")

            lat, lon = most_common
            logger.info(f"✅ Estimated coordinates: Latitude={lat}, Longitude={lon}")
            return float(lat), float(lon)

        except Exception as e:
            logger.error(f"❌ Error estimating geolocation: {e}")
            raise


    # ===== Function: estimate_geoloc_ollama =====
    def estimate_geoloc_ollama(self, nr_prediction):
        """
        Estimates geolocation (latitude and longitude) using Ollama model,
        based on all images in the current working directory.

        Args:
            nr_prediction (int): Number of GPS predictions to consider for each image.
        """
        logger.info("📍 Estimating geolocation using Ollama...")

        try:
            # Create the batch predictor
            predictor = ImageToCoordinates(ollama_model="llama3.2-vision")

            most_common, counter = predictor.run_pipeline(self.working_dir)

            if not most_common:
                logger.error("❌ No geolocation predictions could be computed.")
                raise RuntimeError("Failed to estimate geolocation from images.")

            lat, lon = most_common
            logger.info(f"✅ Estimated coordinates: Latitude={lat}, Longitude={lon}")
            return float(lat), float(lon)

        except Exception as e:
            logger.error(f"❌ Error estimating geolocation: {e}")
            raise


    # ===== Function: estimate_geoloc_geminiAI =====
    def estimate_geoloc_geminiAI(self, nr_prediction):
        """
        Estimates geolocation (latitude and longitude) using Gemini AI model,
        based on all images in the current working directory.
        Args:
            nr_prediction (int): Number of GPS predictions to consider for each image.
        """
        logger.info("📍 Estimating geolocation using Gemini AI...")

        try:
            # Create the batch predictor
            predictor = GeminiGeolocator()

            most_common, counter = predictor.run_pipeline(self.working_dir)

            if not most_common:
                logger.error("❌ No geolocation predictions could be computed.")
                raise RuntimeError("Failed to estimate geolocation from images.")

            lat, lon = most_common
            logger.info(f"✅ Estimated coordinates: Latitude={lat}, Longitude={lon}")
            return float(lat), float(lon)

        except Exception as e:
            logger.error(f"❌ Error estimating geolocation: {e}")
            raise


    # ===== Function: download_satellite_imagery =====
    def download_satellite_imagery(self, lat, lon, area_size_m, zoom):
        """
        Downloads satellite imagery for a specified geographic location. Requires latitude, longitude,
        area size, zoom level, and an API key. Executes an external Python script as a subprocess and logs the
        progress and any errors encountered during the download.

        Args:
            lat (float): Latitude of the center point for the imagery.
            lon (float): Longitude of the center point for the imagery.
            area_size_m (float): Size of the area to download in meters.
            zoom (int): Zoom level for the satellite imagery.
            api_key (str): Valid API key for accessing satellite imagery. - Mapbox or Google Maps.

        Returns:
            bool: True if the download succeeds, False otherwise.
        """
        if not area_size_m or not zoom:
            logger.info("⚠️  Missing area_size_m or zoom arguments for satellite image download.")
            return False
        else:
            logger.info("🛰️  Downloading satellite imagery ...")

            downloader = satelliteTileDownloader(lat, lon, area_size_m, zoom, self.working_dir)

            result = downloader.run_pipeline()
            return result


    # ===== Function: rotate_image =====
    def rotate_image(self, file_path):
        """
        Rotates an image at the specified file path by 90, 180, and
        270 degrees clockwise and saves the rotated images with
        appropriate suffixes in the same directory.
        Args:
            file_path (str): Path to the image file to be rotated.
        """
        img = Image.open(file_path)
        
        folder, filename = os.path.split(file_path)
        name, ext = os.path.splitext(filename)
        
        angles = [90, 180, 270]
        
        for angle in angles:
            rotated_img = img.rotate(-angle, expand=True)  # negative for clockwise rotation
            output_path = os.path.join(folder, f"{name}_rot{angle}{ext}")
            rotated_img.save(output_path)


    def create_scaled_versions(self, image_path):
        """
        Creates scaled versions of the input image at 25%, 50%, and 75% of the original size.
        Args:
            image_path (str): Path to the input image file.
        """
        img = Image.open(image_path)
        base_dir, filename = os.path.split(image_path)
        name, ext = os.path.splitext(filename)

        scales = [(0.25, "_s_0_25"), (0.5, "_s_0_50"), (0.75, "_s_0_75")]

        for scale, suffix in scales:
            new_size = (int(img.width * scale), int(img.height * scale))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            new_filename = os.path.join(base_dir, f"{name}{suffix}{ext}")
            resized.save(new_filename, format="TIFF")
    

    # ===== Function: run_deep_image_matching_and_georef =====
    def run_deep_image_matching_and_georef(self, base_name):
        """
        Runs the Deep-Image-Matching (DIM) algorithm to find correspondences between images and then performs
        georeferencing using the resulting matches. Executes two subprocesses: one to run DIM and another to
        apply the georeferencing script, saving the transformation matrix.

        Args:
            base_name (str): The base filename (without extension) of the orthophoto to georeference.

        Returns:
            None
        """
        logger.info("🔧 Run Deep-Image-Matching Algorithm...")

        ortho_path = os.path.join(self.working_dir, "images", base_name + ".tif")
        render_path = os.path.join(self.working_dir, "images", "top_view.png")
        output_path = os.path.join(self.working_dir, "transformation.txt")
        database_path = os.path.join(self.working_dir, "results_loftr_bruteforce_quality_medium", "database.db")

        self.rotate_image(render_path)
        self.create_scaled_versions(ortho_path)

        # First command: Deep Image Matching
        dim_cmd_1 = [
            "python3", "demo.py",
            # "-p", "se2loftr",
            "-p", "loftr",
            "-t", "none",
            "-s", "bruteforce",
            "--force",
            "--skip_reconstruction",
            "-q", "medium",
            "-V",
            "-d", self.working_dir
        ]
        subprocess.run(dim_cmd_1, capture_output=True, text=True, cwd="/workspace/dim")

        # Second command: Deep Image Matching
        processor = georef_dim(ortho_path, render_path, output_path, database_path)
        processor.run_pipeline()


    # ===== Function: move_images_to_subfolder =====
    def move_images_to_subfolder(self, ortho_map=None):
        """
        Moves specific image files from the working directory into a dedicated "images" subfolder. Targets
        files named "top_view.png" and all files with a ".tif" extension, creating the subfolder if it does not
        already exist.

        Args:
            ortho_map (str, optional): Path to an orthophoto image to be copied into the images subfolder. Defaults to None.

        Returns:
            None
        """
        images_dir = os.path.join(self.working_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        for file_name in os.listdir(self.working_dir):
            if file_name == "top_view.png" or file_name.lower().endswith(".tif"):
                src_path = os.path.join(self.working_dir, file_name)
                dst_path = os.path.join(images_dir, file_name)
                shutil.move(src_path, dst_path)
        
        if ortho_map:
            shutil.copy(ortho_map, os.path.join(images_dir, os.path.basename(ortho_map)))
    

    # ===== Function: load_matrix_4x4 =====
    def load_matrix_4x4(self, file_path):
        """
        Loads a 4x4 matrix from a text file and verifies its shape.

        Args:
            file_path (str): Path to the text file containing the matrix.

        Returns:
            numpy.ndarray: The loaded 4x4 matrix.

        Raises:
            ValueError: If the matrix read from the file is not 4x4.
        """
        matrix = np.loadtxt(file_path, delimiter=None)
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, found {matrix.shape}")
        return matrix


    # ===== Function: load_json =====
    def load_json(self, file_path):
        """
        Loads and parses a JSON file.

        Args:
            file_path (str): Path to the JSON file to be loaded.

        Returns:
            dict or list: The parsed JSON data.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data


    # ===== Function: get_elevation =====
    def get_elevation(self, lat, lon, dataset="srtm30m"):
        """
        Fetches the elevation value for a given latitude and longitude from the OpenTopodata API.

        Args:
            lat (float): Latitude of the location.
            lon (float): Longitude of the location.
            dataset (str, optional): Elevation dataset to query (default is "srtm30m").

        Returns:
            float: Elevation in meters if available, otherwise 0.
        """
        logger.info("🔧 Get Elevation...")
        url = f"https://api.opentopodata.org/v1/{dataset}?locations={lat},{lon}"
        
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            if "results" in data and len(data["results"]) > 0:
                return data["results"][0].get("elevation", 0)
            else:
                return 0
        except Exception as e:
            return 0


    # ===== Function: blender_to_trimesh_transform =====
    def blender_to_trimesh_transform(self):
        """
        Creates a transformation matrix to convert coordinates from Blender's coordinate system
        to Trimesh's coordinate system. The transformation includes a -90 degree rotation around
        the X-axis and axis inversions for Y and Z.

        Returns:
            numpy.ndarray: A 4x4 transformation matrix for converting Blender coordinates to Trimesh coordinates.
        """
        R = trimesh.transformations.rotation_matrix(np.radians(-90), [1,0,0])
        M = np.eye(4)
        M[1,1] = -1
        M[2,2] = -1
        return M @ R


    # ===== Function: decompose_matrix =====
    def decompose_matrix(self, M):
        """
        Decomposes a 4x4 transformation matrix into its translation, scale, and rotation (in Euler angles).

        Args:
            M (numpy.ndarray): A 4x4 transformation matrix.

        Returns:
            dict: A dictionary containing:
                - "translation" (numpy.ndarray): The translation vector (x, y, z).
                - "scale" (numpy.ndarray): Scaling factors along each axis.
                - "euler_deg" (tuple): Rotation angles in degrees as Euler angles (sxyz convention).
                If the decomposition fails, returns ("n/a",).
        """
        M = np.asarray(M, dtype=float)
        assert M.shape == (4,4)

        R = M[:3, :3].copy()
        t = M[:3, 3].copy()

        scale = np.linalg.norm(R, axis=0)
        safe_scale = np.where(scale == 0, 1.0, scale)
        Rn = R / safe_scale

        rot4 = np.eye(4)
        rot4[:3, :3] = Rn

        try:
            euler = trimesh.transformations.euler_from_matrix(rot4, axes='sxyz')
            euler_deg = np.degrees(euler)
        except Exception:
            euler_deg = ("n/a",)

        return {
            "translation": t,
            "scale": scale,
            "euler_deg": euler_deg
        }


    # ===== Function: align_z_to_elevation =====
    def align_z_to_elevation(self, model, elevation, mid_point):
        """
        Aligns the Z-coordinate of a 3D model to a specified elevation by translating it vertically.
        Args:
            model (trimesh.Scene or trimesh.Trimesh): The 3D model to be aligned.
            elevation (float): The target elevation to align the model's Z-coordinate to.
            mid_point (numpy.ndarray): A point on the model used to determine the current Z-coordinate.
        Returns:
            trimesh.Scene or trimesh.Trimesh: The transformed 3D model with adjusted Z-coordinate.
        """
        dz = elevation - (mid_point[2])
        T = trimesh.transformations.translation_matrix([0,0,dz])

        if isinstance(model, trimesh.Scene):
            for geom in model.geometry.values():
                geom.apply_transform(T)
        else:
            model.apply_transform(T)

        return model


    # ===== Function: apply_transform =====
    def apply_transform(self):
        """
        Applies a transformation to a 3D model using a 4x4 matrix, aligning it to the correct
        position, orientation, and scale, and then adjusts its elevation. The function supports
        both `trimesh.Scene` and `trimesh.Trimesh` objects.

        Steps:
            1. Loads the transformation matrix from `transformation.txt`.
            2. Converts coordinates from Blender to Trimesh convention.
            3. Decomposes the transformation matrix to extract rotation, translation, and scale.
            4. Constructs and applies rotation, translation, and scale matrices.
            5. Aligns the model's minimum Z value to the given elevation.
            6. Calculates the elevation at the model's location.
            7. Exports the transformed model as an `.obj` file.

        Returns:
            None
        """
        if not os.path.exists(os.path.join(self.working_dir, "transformation.txt")):
            logger.error("⚠️  Missing transformation.txt file. Cannot apply transformation.")
            return
        logger.info("🔧 Refine Position...")
        matrix_4x4 = self.load_matrix_4x4(os.path.join(self.working_dir, "transformation.txt"))
        model = trimesh.load(os.path.join(self.working_dir, self.base_name + "_scaled.glb"))

        # Blender to trimesh conversion
        T = self.blender_to_trimesh_transform()
        if isinstance(model, trimesh.Scene):
            for geom in model.geometry.values():
                geom.apply_transform(T)
        else:
            model.apply_transform(T)

        # Decompose Matrix 4x4 an get values
        result = self.decompose_matrix(matrix_4x4)

        angle_deg = result["euler_deg"][2]
        translation = [result["translation"][0], result["translation"][1], 0]
        scale_factors = [result["scale"][0], result["scale"][1], 1.0]

        # --- Rotation Matrix ---
        angle_rad = np.radians(angle_deg)
        axis = [0, 0, 1]
        R = trimesh.transformations.rotation_matrix(angle_rad, axis)

        # --- Translation Matrix ---
        T = np.eye(4)
        T[:3, 3] = translation

        # --- Scale Matrix ---
        S = np.eye(4)
        mean_scale_factor = np.mean([scale_factors[0], scale_factors[1]])
        S[0,0] = mean_scale_factor
        S[1,1] = mean_scale_factor
        S[2,2] = mean_scale_factor

        transform = T @ R @ S

        # --- Apply transformation ---
        if isinstance(model, trimesh.Scene):
            for geom in model.geometry.values():
                geom.apply_transform(transform)
        else:
            model.apply_transform(transform)
        
        # --- Get elevation ---
        if isinstance(model, trimesh.Scene):
            points_list = []
            for geom in model.geometry.values():
                pts, _ = geom.sample(10000, return_index=True)
                points_list.append(pts)
            points = np.vstack(points_list)
        else:
            points, _ = model.sample(10000, return_index=True)

        centroid = model.bounds.mean(axis=0) if isinstance(model, trimesh.Scene) else model.center_mass
        distances = np.linalg.norm(points - centroid, axis=1)
        mid_point = points[np.argmin(distances)]

        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        s_lon, s_lat = transformer.transform(mid_point[0], mid_point[1])

        elevation = self.get_elevation(s_lat, s_lon)
        logger.info(f"Elevation at location: {elevation}m")

        self.align_z_to_elevation(model, elevation, mid_point)
        
        # Export model
        out_path = os.path.join(self.args.output_folder, self.base_name)
        file_path = os.path.join(out_path, self.base_name+"_georef.obj")

        os.makedirs(out_path, exist_ok=True)
        model.export(file_path, file_type="obj", digits=15, include_texture=False)


    # ===== Function: run_pipeline =====
    def run_pipeline(self):
        """
        Run the full pipeline:
        1. Generate synthetic views
        2. Estimate geolocation
        3. Get elevation
        4. Handle ortho/satellite images and apply transforms
        """

        logger.info("Start Pipeline")

        # Get mode (default: auto)
        mode = getattr(self.args, "mode", "auto")

        # Load API keys from environment
        mapbox_key = os.getenv("MAPBOX_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        if not mapbox_key:
            logger.warning("MAPBOX_API_KEY not found in environment variables. Some steps may fail.")
        if not gemini_key:
            logger.warning("GEMINI_API_KEY not found in environment variables. Gemini geolocation may not work.")

        # --------------------------
        # Step 1: Synthetic views
        # --------------------------
        if mode in ("auto", "geoloc", "dim"):
            self.generate_synthetic_views(
                input_file=self.args.input_file,
                streetviews=3
            )

        # --------------------------
        # Step 2: Geolocation
        # --------------------------
        lat, lon = None, None
        if mode in ("auto", "geoloc"):
            if self.args.lat is not None and self.args.lon is not None:
                lat, lon = self.args.lat, self.args.lon
                logger.info(f"Using provided Location: lat={lat}, lon={lon}")
            else:
                geoloc_model = getattr(self.args, "geoloc_model", "gemini").lower()
                logger.info(f"Estimating location using {geoloc_model} model")
                
                if geoloc_model == "geoclip":
                    lat, lon = self.estimate_geoloc_geoclip(self.args.nr_prediction)
                elif geoloc_model == "ollama":
                    lat, lon = self.estimate_geoloc_ollama(self.args.nr_prediction)
                else:
                    # Default: Gemini
                    lat, lon = self.estimate_geoloc_geminiAI(self.args.nr_prediction)
            
            logger.info(f"Estimated Location: lat={lat}, lon={lon}")

            if mode == "geoloc":
                return lat, lon

        elif mode == "dim":
            if self.args.lat is None or self.args.lon is None:
                logger.error("Mode 'dim' requires lat and lon to be provided.")
                return
            lat, lon = self.args.lat, self.args.lon
            logger.info(f"Using provided Location: lat={lat}, lon={lon}")

        # --------------------------
        # Step 3: Elevation
        # --------------------------
        elevation = self.get_elevation(lat, lon)
        logger.info(f"Orto Elevation at location: {elevation}m")

        # --------------------------
        # Step 4: Ortho / Satellite imagery
        # --------------------------
        ortho_provided = getattr(self.args, "ortho", None)

        if ortho_provided:
            logger.info("--> Using user-provided ortho images")
            self.move_images_to_subfolder(ortho_provided)
            self.run_deep_image_matching_and_georef(self.base_name)
            self.apply_transform()

        elif mapbox_key:
            logger.info("--> Downloading satellite imagery using MAPBOX API")
            if self.download_satellite_imagery(lat, lon, self.args.area_size_m, self.args.zoom):
                self.move_images_to_subfolder()
                self.run_deep_image_matching_and_georef(self.base_name)
                self.apply_transform()
            else:
                logger.warning("Failed to download satellite imagery.")

        else:
            # Fallback: no ortho or API key
            logger.info(f"Approximate Location: lat={lat}, lon={lon}, elevation={elevation}")
            logger.info("--> To refine the location, provide a valid MAPBOX API key or ortho images.")
        

        # Copy temporary data to output folder
        out_path = os.path.join(self.args.output_folder, self.base_name, "tmp")
        out_path_1 = os.path.join(self.args.output_folder, self.base_name, self.base_name+".tif")
        os.makedirs(out_path, exist_ok=True)

        shutil.copytree(self.working_dir, out_path, dirs_exist_ok=True)
        shutil.copy(os.path.join(self.working_dir, "images", self.base_name+".tif"), out_path_1)