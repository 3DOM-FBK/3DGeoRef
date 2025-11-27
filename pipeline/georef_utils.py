import os
import sys
import numpy as np
import trimesh
from pyproj import Transformer
import logging
import requests
import json


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


class GeoTransformer:
    def __init__(self, working_dir, input_file, output_folder, lat, lon):
        self.working_dir = working_dir
        self.input_file = input_file
        self.output_folder = output_folder
        self.lat = float(lat)
        self.lon = float(lon)
        self.basename = self.base_name = os.path.splitext(os.path.basename(input_file))[0]

    @staticmethod
    def blender_to_trimesh_transform():
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
    

    @staticmethod
    def decompose_matrix(M):
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
    

    @staticmethod
    def get_elevation(lat, lon, dataset="srtm30m"):
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
    
    @staticmethod
    def compute_scale_factor_correct(lat):
        """
        Computes the scale factor (K) for the Web Mercator projection.

        Args:
            lat (float): The latitude in degrees.

        Returns:
            float: The scale factor K.
        """
        # 1. Convert latitude from degrees to radians
        lat_rad = np.radians(lat)
        
        # 2. Calculate K = 1 / cos(phi)
        # This factor K is the distortion (how many times is the map stretched)
        scale_factor_K = 1.0 / np.cos(lat_rad)
        
        return scale_factor_K
    

    @staticmethod
    def load_matrix_4x4(file_path):
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
    

    @staticmethod
    def get_mid_point(model):
        """
        Computes the midpoint of a 3D model.

        Args:
            model (trimesh.Scene or trimesh.Trimesh): The 3D model.

        Returns:
            numpy.ndarray: The midpoint of the model.
        """
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

        return mid_point, centroid
    

    @staticmethod
    def get_elevation_at_mid_point(mid_point):
        """
        Computes the elevation at the midpoint of a 3D model.

        Args:
            mid_point (numpy.ndarray): The midpoint of the model.

        Returns:
            float: The elevation at the midpoint of the model.
        """
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        s_lon, s_lat = transformer.transform(mid_point[0], mid_point[1])

        elevation = GeoTransformer.get_elevation(s_lat, s_lon)
        logger.info(f"Elevation at location: {elevation}m")

        return elevation
    

    @staticmethod
    def move_model_to_origin(model, centroid):
        """
        Moves the 3D model so that the centroid is in the origin of the scene.

        Args:
            model (trimesh.Scene or trimesh.Trimesh): The 3D model.
            centroid (numpy.ndarray): The centroid of the model.

        Returns:
            trimesh.Scene or trimesh.Trimesh: The moved 3D model.
        """
        translation = -centroid
        T = trimesh.transformations.translation_matrix(translation)
        
        if isinstance(model, trimesh.Scene):
            for geom in model.geometry.values():
                geom.apply_transform(T)
        else:
            model.apply_transform(T)

    
        return model, T
    
    
    @staticmethod
    def export_model(model, out_path):
        """
        Export model to glb file.

        Args:
            model (trimesh.Scene or trimesh.Trimesh): The 3D model.
            out_path (str): Path to the output directory.

        Returns:
            None
        """
        # os.makedirs(output_folder, exist_ok=True)
        # file_path = os.path.join(output_folder, basename + "_georef.glb")
        model.export(out_path, file_type="glb")
    

    @staticmethod
    def to_cesium(model):
        """
        Apply 90 degrees rotations around X and Z axes to the model, 
        to match the Cesium coordinate system.

        Args:
            model (trimesh.Scene or trimesh.Trimesh): The 3D model.

        Returns:
            None
        """
        # Apply 90 degrees rotation around X axis
        R_x = trimesh.transformations.rotation_matrix(np.radians(-90), [1,0,0])
        model.apply_transform(R_x)

        # Apply 90 degrees rotation around Z axis
        R_z = trimesh.transformations.rotation_matrix(np.radians(-90), [0,1,0])
        model.apply_transform(R_z)

        # M_rot_cesium = R_z @ R_x

        return model, R_x, R_z

    @staticmethod
    def to_heritage_data_processor(S_k, M_translation, matrix_4x4, matrix_blender_path):
        with open(matrix_blender_path, "r") as f:
            matrix_blender = np.array(json.load(f), dtype=np.float64)
        
        # Now all matrices are 4x4
        final_matrix = matrix_blender
        
        transformation_params = GeoTransformer.decompose_matrix(final_matrix)
        

        angle_deg_0 = transformation_params["euler_deg"][0]
        angle_deg_1 = transformation_params["euler_deg"][1]
        angle_deg_2 = transformation_params["euler_deg"][2]
        translation = [transformation_params["translation"][0], transformation_params["translation"][1], 0]
        scale_factors = [transformation_params["scale"][0], transformation_params["scale"][1], 1.0]

        print("Angle 1 (deg):", angle_deg_0)
        print("Angle 2 (deg):", angle_deg_1)
        print("Angle 3 (deg):", angle_deg_2)
        print("Translation:", translation)
        print("Scale Factors:", scale_factors)


    def run_GeoTransformer(self):
        """
        Runs the GeoTransformer, applying the transformation matrix to the model and exporting it to glb.

        Returns:
            None
        """
        if not os.path.exists(os.path.join(self.working_dir, "transformation.txt")):
            logger.error("⚠️  Missing transformation.txt file. Cannot apply transformation.")
            return
        matrix_4x4 = GeoTransformer.load_matrix_4x4(os.path.join(self.working_dir, "transformation.txt"))
        model = trimesh.load(os.path.join(self.working_dir, self.base_name + "_scaled.glb"))

        # Blender to trimesh conversion
        T = self.blender_to_trimesh_transform()
        if isinstance(model, trimesh.Scene):
            for geom in model.geometry.values():
                geom.apply_transform(T)
        else:
            model.apply_transform(T)

        # Decompose Matrix 4x4 an get values
        transformation_params = GeoTransformer.decompose_matrix(matrix_4x4)
        angle_deg_2 = transformation_params["euler_deg"][2]
        translation = [transformation_params["translation"][0], transformation_params["translation"][1], 0]
        scale_factors = [transformation_params["scale"][0], transformation_params["scale"][1], 1.0]

        # --- Rotation Matrix ---
        angle_rad = np.radians(angle_deg_2)
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

        matrix_dim = T @ R @ S

        # --- Apply transformation ---
        if isinstance(model, trimesh.Scene):
            for geom in model.geometry.values():
                geom.apply_transform(matrix_dim)
        else:
            model.apply_transform(matrix_dim)
        
        # --- Obtain the mid_point of the model ---
        mid_point, centroid = GeoTransformer.get_mid_point(model)

        # # --- Get elevation at mid_point ---
        # elevation_at_mid_point = GeoTransformer.get_elevation_at_mid_point(mid_point)

        # --- Move model to origin ---
        model, M_translation = GeoTransformer.move_model_to_origin(model, centroid)

        # # Compute scale factor correct
        # scale_factor_K = GeoTransformer.compute_scale_factor_correct(self.lat)

        # # --- Apply scale_factor_K to model ---
        # S_k = np.eye(4)
        # S_k[0,0] = S_k[1,1] = S_k[2,2] = 1 / scale_factor_K
        # model.apply_transform(S_k)
        
        # Convert the model to Cesium format with to_cesium()
        # model, R_x, R_z = GeoTransformer.to_cesium(model)
        
        # Export model
        out_path = os.path.join(self.output_folder, self.basename + "_georef.glb")
        GeoTransformer.export_model(model, out_path)

        # matrix_blender = os.path.join(self.working_dir, "matrix_blender.json")
        # GeoTransformer.to_heritage_data_processor(S_k, M_translation, matrix_dim, matrix_blender)



if __name__ == "__main__":
    gt = GeoTransformer(
        working_dir="/tmp/piazzaDuomoTrento/",
        input_file="/data/input/piazzaDuomoTrento.glb",
        output_folder="/data/output/",
        lat="46.067123",
        lon="11.121544"
    )
    gt.run_GeoTransformer()
