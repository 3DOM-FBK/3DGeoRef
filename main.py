import os
import argparse
import shutil
import subprocess
import json
import sys
import trimesh
import numpy as np
import logging
from PIL import Image
import requests
from scipy.spatial.transform import Rotation as R


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


# ===== Function: parse_args =====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True, help="Path to input .obj/.ply/.fbx/.glb/.gltf file")
    parser.add_argument("-o", "--output_folder", required=True, help="Folder to save outputs")
    parser.add_argument("--streetviews", type=str, default="5", help="Number of streetview-style renderings around the model (default: 5)")
    parser.add_argument("--nr_prediction", type=str, default="3", help="Number of gps prediction (default: 3)")
    parser.add_argument("--api_key", type=str, help="Google Cloud Platform API key.")
    parser.add_argument("--area_size_m", type=str, default="200", help="Side length of the square area to download (in meters).")
    parser.add_argument("--zoom", type=str, default="18", help="Zoom level (e.g., 18 or 20).")
    parser.add_argument("--lat", type=str, default=None,help="Latitude of 3d model")
    parser.add_argument("--lon", type=str, default=None,help="Longitude of 3d mode")
    
    return parser.parse_args()


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
            "--python", "/app/python/synthetic_imgs.py",
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
    def estimate_geolocation(self, nr_prediction):
        """
        Estimates the geolocation (latitude and longitude) based on previously generated data in the working directory.
        Runs an external Python script as a subprocess and parses its JSON output. Logs errors and raises exceptions
        if the subprocess fails or the output cannot be parsed.

        Args:
            nr_prediction (int): The number of geolocation predictions to consider when estimating the final location.

        Returns:
            tuple: A tuple containing the latitude and longitude as floats.
        """
        logger.info("📍 Estimating geolocation...")

        geoloc_cmd = [
            "python3", "/app/python/geoloc_img.py",
            "-i", self.working_dir,
            "--nr_prediction", str(nr_prediction)
        ]

        res = subprocess.run(geoloc_cmd, capture_output=True, text=True)

        if res.returncode != 0:
            logger.error("❌ Geolocation failed")
            raise RuntimeError("Geolocation subprocess failed.")

        try:
            data = json.loads(res.stdout)
            lat, lon = data["lat"], data["lon"]
            return lat, lon
        except Exception as e:
            logger.error("❌ Failed to parse geolocation output")
            sys.exit(1)


    # ===== Function: download_satellite_imagery =====
    def download_satellite_imagery(self, api_key, lat, lon, area_size_m, zoom):
        """
        Downloads satellite imagery from Google for a specified geographic location. Requires latitude, longitude,
        area size, zoom level, and an API key. Executes an external Python script as a subprocess and logs the
        progress and any errors encountered during the download.

        Args:
            lat (float): Latitude of the center point for the imagery.
            lon (float): Longitude of the center point for the imagery.
            area_size_m (float): Size of the area to download in meters.
            zoom (int): Zoom level for the satellite imagery.
            api_key (str): Google API key for accessing satellite imagery.

        Returns:
            bool: True if the download succeeds, False otherwise.
        """
        if not area_size_m or not zoom:
            logger.info("⚠️  Missing area_size_m or zoom arguments for satellite image download.")
            return False
        else:
            logger.info("🛰️  Downloading satellite imagery from Google ...")
            ortho_cmd = [
                "python3", "/app/python/ortho.py",
                "--api_key", api_key,
                "--center_lat", str(lat),
                "--center_lon", str(lon),
                "--area_size_m", str(area_size_m),
                "--zoom", str(zoom),
                "-o", self.working_dir
            ]
            # result = subprocess.run(ortho_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result = subprocess.run(ortho_cmd)
            if result.returncode != 0:
                logger.error("--> ⚠️ Downloading satellite imagery from Google - Error.")
                return False
            else: 
                logger.info("--> ✅ Downloading satellite imagery from Google - Done.")
                return True


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
        dim_cmd_1 = [
            "python3", "demo.py",
            "-p", "se2loftr",
            "-t", "none",
            "-s", "bruteforce",
            "--force",
            "--skip_reconstruction",
            "-q", "medium",
            "-V",
            "-d", self.working_dir
        ]
        subprocess.run(dim_cmd_1, capture_output=True, text=True, cwd="/workspace/dim")

        dim_cmd_2 = [
            "python3", "/app/python/georef.py",
            "--ortho", os.path.join(self.working_dir, "images", base_name + ".tif"),
            "--render", os.path.join(self.working_dir, "images", "top_view.png"),
            "--output", os.path.join(self.working_dir, "transformation.txt"),
            "--database", os.path.join(self.working_dir, "results_se2loftr_bruteforce_quality_medium", "database.db")
        ]
        subprocess.run(dim_cmd_2, capture_output=True, text=True)


    # ===== Function: move_images_to_subfolder =====
    def move_images_to_subfolder(self):
        """
        Moves specific image files from the working directory into a dedicated "images" subfolder. Targets
        files named "top_view.png" and all files with a ".tif" extension, creating the subfolder if it does not
        already exist.

        Args:
            None

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


    # ===== Function: apply_transformation_to_model =====
    def apply_transformation_to_model(self, model_path, dst_dir, base_name):
        """
        Applies a 4x4 transformation matrix to a 3D model to align it with georeferenced coordinates and
        convert it from GIS to GLTF coordinate conventions. The transformed model is then saved to the
        specified destination directory.

        Args:
            model_path (str): Path to the original 3D model file.
            dst_dir (str): Directory where the transformed model and transformation matrix are located and where the output will be saved.
            base_name (str): Base name used to save the transformed model with a "_transformed" suffix.

        Returns:
            None
        """
        matrix_path = os.path.join(dst_dir, "transformation.txt")

        if (os.path.isfile(matrix_path)):
            with open(matrix_path, 'r') as f:
                lines = f.readlines()

            matrix = np.array([[float(val) for val in line.strip().split()] for line in lines])
            if matrix.shape != (4, 4):
                raise ValueError("The transformation matrix must be 4x4.")
            
            scale_x = np.linalg.norm(matrix[0, :3])
            scale_y = np.linalg.norm(matrix[1, :3])
            matrix[2, :3] *= scale_z
            
            gis_to_gltf = np.array([
                [-1,  0,  0, 0],  # X remains X
                [0,  0,  -1, 0],  # Z (up in GIS) -> Y (up in GLTF)
                [0,  1,  0, 0],   # Y (north in GIS) -> Z (forward in GLTF)
                [0,  0,  0, 1]
            ])
            adjusted_matrix = gis_to_gltf @ matrix

            model = trimesh.load(model_path)

            if isinstance(model, trimesh.Scene):
                for geom in model.geometry.values():
                    geom.apply_transform(adjusted_matrix)
            else:
                model.apply_transform(adjusted_matrix)
            
            bbox_min, bbox_max = model.bounds
            centroid = (bbox_min + bbox_max) / 2.0

            model.export(os.path.join(dst_dir, base_name + "_transformed.obj"))
    

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


    # ===== Function: align_min_z_to_elevation =====
    def align_min_z_to_elevation(self, model, elevation):
        """
        Aligns the lowest point of a 3D model to a specified elevation. Supports both trimesh.Scene and
        individual trimesh.Trimesh objects by translating the model along the Y-axis.

        Args:
            model (trimesh.Trimesh or trimesh.Scene): The 3D model to align.
            elevation (float): The target elevation to align the model's minimum Y coordinate.

        Returns:
            trimesh.Trimesh or trimesh.Scene: The transformed model with its minimum Y aligned to the elevation.
        """
        if isinstance(model, trimesh.Scene):
            min_y = max([geom.bounds[0,1] for geom in model.geometry.values()])
        else:
            min_y = model.bounds[0,1]

        dy = -(elevation - (-min_y))
        T = trimesh.transformations.translation_matrix([0,dy,0])

        if isinstance(model, trimesh.Scene):
            for geom in model.geometry.values():
                geom.apply_transform(T)
        else:
            model.apply_transform(T)

        return model


    # ===== Function: apply_transform =====
    def apply_transform(self, elevation):
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
            6. Exports the transformed model as an `.obj` file.

        Args:
            elevation (float): The target elevation to which the model should be aligned.

        Returns:
            None
        """
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

        angle_deg = -result["euler_deg"][2]
        translation = [result["translation"][0], 0, result["translation"][1]]
        scale_factors = [result["scale"][0], result["scale"][1], 1.0]

        # --- Rotation Matrix ---
        angle_rad = np.radians(angle_deg)
        axis = [0, 1, 0]
        R = trimesh.transformations.rotation_matrix(angle_rad, axis)

        # --- Translation Matrix ---
        T = np.eye(4)
        T[:3, 3] = translation

        # --- Scale Matrix ---
        S = np.eye(4)
        S[0,0] = scale_factors[0]
        S[1,1] = scale_factors[1]
        S[2,2] = scale_factors[2]

        transform = T @ R @ S

        # # Apply transformation
        if isinstance(model, trimesh.Scene):
            for geom in model.geometry.values():
                geom.apply_transform(transform)
        else:
            model.apply_transform(transform)
        
        # Apply elevation
        model = self.align_min_z_to_elevation(model, elevation)
        
        # Export model
        file_path = os.path.join(self.args.output_folder, self.base_name+"_georef.obj")
        model.export(file_path, file_type="obj", digits=15, include_texture=True)


    # ===== Function: run_pipeline =====
    def run_pipeline(self):
        logger.info(f"Start Pipeline")

        # Generate synthetic views
        self.generate_synthetic_views(input_file=self.args.input_file, streetviews=3)

        if (self.args.lat is None and self.args.lon is None):
            # Extimate model geolocation
            lat, lon = self.estimate_geolocation(self.args.nr_prediction)
        else:
            lat = self.args.lat
            lon = self-args.lon

        # Run Get Elevation Algorithm
        elevation = self.get_elevation(lat, lon)

        if (self.args.api_key):
            # Download satellite imagery from Google
            result = self.download_satellite_imagery(self.args.api_key, lat, lon, self.args.area_size_m, self.args.zoom)

            # if (result):
            #     # Moving DIM input images to subfolder
            #     self.move_images_to_subfolder()

            #     # Run Deep-Image-Matching Algorithm
            #     self.run_deep_image_matching_and_georef(self.base_name)

            #     # Apply 4x4 transformation matrix to the model
            #     self.apply_transform(elevation)
        
        else:
            logger.info(f"Approximate Location: lat = {lat}, lon = {lon}, elevation = {elevation}")
            logger.info("--> To refine the location, please provide a valid Google Map Tile API key.")


# ===== Function: main =====
if __name__ == "__main__":
    args = parse_args()

    processor = PipelineProcessor(args)
    processor.run_pipeline()