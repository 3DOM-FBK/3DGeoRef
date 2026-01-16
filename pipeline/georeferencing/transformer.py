"""
GeoTransformer Module

This module provides functionality for georeferencing 3D models by applying
coordinate transformations, scale corrections, and coordinate system conversions.
"""

import os
import sys
import json
import logging
from typing import Tuple, Dict, Optional, Union

import numpy as np
import trimesh
import requests
import rasterio
from rasterio.transform import Affine
from pyproj import Transformer


# Configure logging
log_level = os.environ.get("LOGLEVEL", "INFO").upper()
log_format = '%(asctime)s - %(levelname)-8s - %(message)s'
logging.basicConfig(
    level=getattr(logging, log_level),
    format=log_format,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class MatrixUtils:
    """Utility class for 4x4 transformation matrix operations."""
    

    @staticmethod
    def load_matrix(file_path: str) -> np.ndarray:
        """
        Load a 4x4 transformation matrix from a text file.
        
        Args:
            file_path: Path to the matrix file
            
        Returns:
            4x4 numpy array
            
        Raises:
            ValueError: If matrix is not 4x4
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Matrix file not found: {file_path}")
        
        matrix = np.loadtxt(file_path, delimiter=None)
        
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, found {matrix.shape}")
        
        return matrix
    

    @staticmethod
    def decompose(matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose a 4x4 transformation matrix into translation, scale, and rotation.
        
        Args:
            matrix: 4x4 transformation matrix
            
        Returns:
            Dictionary with 'translation', 'scale', and 'euler_deg' keys
        """
        matrix = np.asarray(matrix, dtype=float)
        assert matrix.shape == (4, 4), "Matrix must be 4x4"
        
        # Extract translation
        translation = matrix[:3, 3].copy()
        
        # Extract rotation and scale
        rotation_scale = matrix[:3, :3].copy()
        scale = np.linalg.norm(rotation_scale, axis=0)
        
        # Normalize rotation matrix
        safe_scale = np.where(scale == 0, 1.0, scale)
        rotation_normalized = rotation_scale / safe_scale
        
        # Convert to 4x4 for euler extraction
        rotation_4x4 = np.eye(4)
        rotation_4x4[:3, :3] = rotation_normalized
        
        # Extract Euler angles
        try:
            euler_rad = trimesh.transformations.euler_from_matrix(rotation_4x4, axes='sxyz')
            euler_deg = np.degrees(euler_rad)
        except Exception as e:
            logger.warning(f"Failed to extract Euler angles: {e}")
            euler_deg = np.array([0, 0, 0])
        
        return {
            "translation": translation,
            "scale": scale,
            "euler_deg": euler_deg
        }


    @staticmethod
    def blender_to_trimesh_matrix():
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
    def create_trs(translation: np.ndarray, rotation: np.ndarray, scale: Union[float, np.ndarray]) -> np.ndarray:
        """
        Create a 4x4 transformation matrix from translation, rotation, and scale.
        
        Args:
            translation: 3D translation vector
            rotation: Rotation matrix (3x3) or Euler angles (3D vector in radians)
            scale: Uniform scale factor or 3D scale vector
            
        Returns:
            4x4 transformation matrix
        """
        matrix = np.eye(4)
        
        # Apply scale
        if np.isscalar(scale):
            matrix[:3, :3] *= scale
        else:
            matrix[:3, :3] *= np.diag(scale)
        
        # Apply rotation
        if rotation.shape == (3, 3):
            matrix[:3, :3] = rotation @ matrix[:3, :3]
        elif rotation.shape == (3,):
            # Assume Euler angles
            rot_matrix = trimesh.transformations.euler_matrix(*rotation, axes='sxyz')
            matrix[:3, :3] = rot_matrix[:3, :3] @ matrix[:3, :3]
        
        # Apply translation
        matrix[:3, 3] = translation
        
        return matrix


class ElevationService:
    """Service for fetching elevation data from OpenTopoData API."""
    
    DEFAULT_DATASET = "srtm30m"
    API_URL = "https://api.opentopodata.org/v1/{dataset}"

    @staticmethod
    def get_elevation(lat: float, lon: float, dataset: str = DEFAULT_DATASET) -> float:
        """
        Fetch elevation for given coordinates.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            dataset: Elevation dataset name
            
        Returns:
            Elevation in meters, or 0 if unavailable
        """
        
        url = ElevationService.API_URL.format(dataset=dataset)
        params = {"locations": f"{lat},{lon}"}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "results" in data and len(data["results"]) > 0:
                elevation = data["results"][0].get("elevation", 0)
                return elevation
            else:
                logger.warning("⚠️ GeoRef_utils.py - No elevation data available")
                return 0
                
        except requests.RequestException as e:
            logger.error(f"❌ GeoRef_utils.py - Failed to fetch elevation: {e}")
            return 0



class ModelAnalyzer:
    """Utility class for analyzing 3D model geometry."""
    

    @staticmethod
    def get_centroid_and_midpoint(model: Union[trimesh.Scene, trimesh.Trimesh], 
                                   num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the centroid and midpoint of a 3D model.
        
        The centroid is the geometric center of the bounding box.
        The midpoint is the sampled point closest to the centroid.
        
        Args:
            model: 3D model (Scene or Trimesh)
            num_samples: Number of points to sample
            
        Returns:
            Tuple of (midpoint, centroid)
        """
        # Sample points from model
        if isinstance(model, trimesh.Scene):
            points_list = []
            for geom in model.geometry.values():
                pts, _ = geom.sample(num_samples, return_index=True)
                points_list.append(pts)
            points = np.vstack(points_list)
            centroid = model.bounds.mean(axis=0)
        else:
            points, _ = model.sample(num_samples, return_index=True)
            centroid = model.center_mass
        
        # Find point closest to centroid
        distances = np.linalg.norm(points - centroid, axis=1)
        midpoint = points[np.argmin(distances)]
        
        return midpoint, centroid
    

    @staticmethod
    def apply_transform(model: Union[trimesh.Scene, trimesh.Trimesh], 
                       matrix: np.ndarray) -> None:
        """
        Apply transformation matrix to model (in-place).
        
        Args:
            model: 3D model
            matrix: 4x4 transformation matrix
        """
        if isinstance(model, trimesh.Scene):
            for geom in model.geometry.values():
                geom.apply_transform(matrix)
        else:
            model.apply_transform(matrix)


class GeoTransformer:
    """Main class for georeferencing 3D models."""
    
    def __init__(self, working_dir: str, input_file: str, output_folder: str, 
                 lat: float, lon: float):
        """
        Initialize GeoTransformer.
        
        Args:
            working_dir: Working directory containing transformation files
            input_file: Path to input 3D model
            output_folder: Output directory for georeferenced model
            lat: Latitude of reference point
            lon: Longitude of reference point
        """
        self.working_dir = working_dir
        self.input_file = input_file
        self.lat = float(lat)
        self.lon = float(lon)
        self.basename = os.path.splitext(os.path.basename(input_file))[0]
        self.output_folder = os.path.join(output_folder, self.basename)
        
        # Validate directories
        os.makedirs(self.output_folder, exist_ok=True)
    

    @staticmethod
    def compute_web_mercator_scale_factor(lat: float) -> float:
        """
        Compute scale factor for Web Mercator projection at given latitude.
        
        The scale factor K = 1 / cos(latitude) represents the distortion
        introduced by the Web Mercator projection.
        
        Args:
            lat: Latitude in degrees
            
        Returns:
            Scale factor K
        """
        lat_rad = np.radians(lat)
        return 1.0 / np.cos(lat_rad)
    

    def _load_transformation_matrix(self) -> Optional[np.ndarray]:
        """Load transformation matrix from working directory."""
        matrix_path = os.path.join(self.working_dir, "transformation.txt")
        
        if not os.path.exists(matrix_path):
            logger.error("⚠️ GeoRef_utils.py - Missing transformation.txt file")
            return None
        
        try:
            return MatrixUtils.load_matrix(matrix_path)
        except Exception as e:
            logger.error(f"❌ GeoRef_utils.py - Failed to load transformation matrix: {e}")
            return None
    

    def _load_blender_matrix(self) -> Optional[np.ndarray]:
        """Load and adjust Blender transformation matrix."""
        matrix_path = os.path.join(self.working_dir, "matrix_blender.json")
        
        if not os.path.exists(matrix_path):
            logger.warning("⚠️ GeoRef_utils.py - Blender matrix not found, skipping")
            return None
        
        try:
            with open(matrix_path, "r") as f:
                matrix = np.array(json.load(f), dtype=np.float64)
            
            # Swap Y and Z translation components and negate Y
            matrix[1][3], matrix[2][3] = matrix[2][3], -matrix[1][3]
            
            return matrix
        except Exception as e:
            logger.error(f"❌ GeoRef_utils.py - Failed to load Blender matrix: {e}")
            return None
    

    def _get_elevation_at_model_midpoint(self, midpoint: np.ndarray) -> float:
        """Get elevation at model midpoint by converting to WGS84."""
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(midpoint[0], midpoint[1])
        return ElevationService.get_elevation(lat, lon), lon, lat
    

    def _apply_cesium_rotation(self, model: Union[trimesh.Scene, trimesh.Trimesh]) -> np.ndarray:
        """
        Apply rotation to match Cesium coordinate system.
        
        Returns:
            Rotation matrix applied
        """
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.radians(-90), [0, 1, 0]
        )
        return rotation_matrix
    

    def _export_heritage_data(self, matrices: Dict[str, np.ndarray], 
                             elevation: float,
                             longitude: float,
                             latitude: float) -> None:
        """
        Export transformation data for Heritage Data Processor.
        
        Args:
            matrices: Dictionary of transformation matrices
            elevation: Elevation at model location
        """
        # Combine all transformations
        final_matrix = (matrices['rotation_cesium'] @ 
                       matrices['rotation_x_90'] @ 
                       matrices['scale_matrix'] @ 
                       matrices['translation_matrix'] @ 
                       matrices['matrix_dim'] @ 
                       matrices['T_blender_to_trimesh'] @ 
                       matrices['blender_matrix'])
        
        # Decompose and log
        params = MatrixUtils.decompose(final_matrix)

        # Apply custom modifications to params for rotation and translation
        original_euler_deg = params['euler_deg'].copy()
        original_translation = params['translation'].copy()

        params['euler_deg'][0] = -original_euler_deg[0]
        params['euler_deg'][1] = -original_euler_deg[2]
        params['euler_deg'][2] = -original_euler_deg[1]

        params['translation'][0] = original_translation[0]
        params['translation'][1] = -original_translation[2]
        params['translation'][2] = original_translation[1]

        logger.info("=" * 60)
        logger.info("HERITAGE DATA PROCESSOR EXPORT")
        logger.info("=" * 60)
        logger.info(f"Euler Angles (deg): {params['euler_deg']}")
        logger.info(f"Translation: {params['translation']}")
        logger.info(f"Mean Scale Factor: {np.mean(params['scale']):.6f}")
        logger.info(f"Elevation: {elevation}m")
        logger.info("=" * 60)
        
        # Save to JSON
        output_path = os.path.join(self.output_folder, "heritage_data.json")
        export_data = {
            "lat": latitude,
            "lon": longitude,
            "place": "",
            "description": "",
            "scale": params['scale'].tolist(),
            "rotation": params['euler_deg'].tolist(),
            "translation": params['translation'].tolist()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    

    def run(self) -> bool:
        """
        Execute the complete georeferencing pipeline.
        
        Returns:
            True if successful, False otherwise
        """        
        # Step 1: Load transformation matrix
        matrix_dim = self._load_transformation_matrix()
        if matrix_dim is None:
            return False
        
        # Step 2: Load 3D model
        model_path = os.path.join(self.working_dir, f"{self.basename}_scaled.glb")
        
        if not os.path.exists(model_path):
            logger.error(f"❌ GeoRef_utils.py - Model not found: {model_path}")
            return False
        
        try:
            model = trimesh.load(model_path)
        except Exception as e:
            logger.error(f"❌ GeoRef_utils.py - Failed to load model: {e}")
            return False
        
        # Step 2: Apply Blender to Trimesh transformation
        T_blender_to_trimesh = MatrixUtils.blender_to_trimesh_matrix()
        ModelAnalyzer.apply_transform(model, T_blender_to_trimesh)
        
        # Step 3: Decompose and rebuild transformation matrix
        params = MatrixUtils.decompose(matrix_dim)
        
        # Reorder components (coordinate system conversion)
        angle_deg = [params["euler_deg"][0], 
                    params["euler_deg"][1], 
                    params["euler_deg"][2]]
        translation = [params["translation"][0], 
                      params["translation"][1], 
                      params["translation"][2]]
        mean_scale = np.mean([params["scale"][0], params["scale"][1]])
        
        # Rebuild matrix
        R = trimesh.transformations.euler_matrix(np.radians(angle_deg[0]), 
                                                 np.radians(angle_deg[1]), 
                                                 np.radians(angle_deg[2]), 
                                                 axes='sxyz')
        T = trimesh.transformations.translation_matrix(translation)
        S = np.eye(4)
        S[0, 0] = S[1, 1] = S[2, 2] = mean_scale
        
        matrix_dim = T @ R @ S
        
        # Step 4: Apply transformation
        ModelAnalyzer.apply_transform(model, matrix_dim)

        # Step 4.5: Save scaled model
        scaled_model_path = os.path.join(self.output_folder, f"{self.basename}_dim.obj")
        model.export(scaled_model_path, file_type="obj", include_texture=False)
        
        # Step 5: Calculate model center
        midpoint, centroid = ModelAnalyzer.get_centroid_and_midpoint(model)
        
        # Step 6: Get elevation
        elevation, lon, lat = self._get_elevation_at_model_midpoint(midpoint)
        
        # Step 7: Move to origin
        translation_matrix = trimesh.transformations.translation_matrix(-centroid)
        ModelAnalyzer.apply_transform(model, translation_matrix)
        
        # Step 8: Apply Web Mercator scale correction
        scale_factor = self.compute_web_mercator_scale_factor(self.lat)
        
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale_matrix[1, 1] = scale_matrix[2, 2] = 1 / scale_factor
        ModelAnalyzer.apply_transform(model, scale_matrix)

        # Step 9: Rotate model 90 degrees around X axis
        rotation_x_90 = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        ModelAnalyzer.apply_transform(model, rotation_x_90)
        
        # Step 10: Convert to Cesium coordinate system
        rotation_cesium = self._apply_cesium_rotation(model)
        ModelAnalyzer.apply_transform(model, rotation_cesium)

        # Step 11: Export georeferenced model
        output_path = os.path.join(self.output_folder, f"{self.basename}_georef.glb")

        # Step 12: Apply elevation to model
        # TO DO...
        
        try:
            model.export(output_path, file_type="glb")
        except Exception as e:
            logger.error(f"❌ GeoRef_utils.py - Failed to export model: {e}")
            return False
        
        # Step 11: Export heritage data (if Blender matrix available)
        blender_matrix = self._load_blender_matrix()
        if blender_matrix is not None:
            matrices = {
                'rotation_cesium': rotation_cesium,
                'rotation_x_90': rotation_x_90,
                'scale_matrix': scale_matrix,
                'translation_matrix': translation_matrix,
                'matrix_dim': matrix_dim,
                'T_blender_to_trimesh': T_blender_to_trimesh,
                'blender_matrix': blender_matrix
            }
            self._export_heritage_data(matrices, elevation, lon, lat)
        
        return True

# if __name__ == "__main__":
#     gt = GeoTransformer(
#         working_dir="/tmp/1d936fe09cc64376802b196157a73d16/",
#         input_file="/data/input/paper_january/1d936fe09cc64376802b196157a73d16.glb",
#         output_folder="/data/output/debug_tmp/",
#         lat="41.9022",
#         lon="12.4578"
#     )
#     gt.run()