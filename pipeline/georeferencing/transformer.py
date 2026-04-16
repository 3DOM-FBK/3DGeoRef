"""
GeoTransformer Module

This module exports georeferencing metadata for the UH4D Browser.
The final output is an XLSX file containing latitude, longitude, height,
translation, rotation, and scale referenced to the original input model.

Target viewer conventions:
- Y axis is height
- Euler rotation order is YXZ
- No Cesium-specific axis correction is applied
"""

import json
import logging
import os
import sys
from typing import Dict, Optional, Union

import numpy as np
import requests
import trimesh
from pyproj import CRS, Transformer


log_level = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class MatrixUtils:
    """Utility class for 4x4 transformation matrices."""

    @staticmethod
    def load_matrix(file_path: str) -> np.ndarray:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Matrix file not found: {file_path}")

        matrix = np.loadtxt(file_path, delimiter=None)
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, found {matrix.shape}")
        return matrix

    @staticmethod
    def decompose(matrix: np.ndarray, euler_axes: str = "ryxz") -> Dict[str, np.ndarray]:
        """
        Decompose a 4x4 matrix into translation, scale, and Euler angles.

        Args:
            matrix: 4x4 transformation matrix.
            euler_axes: Euler extraction convention used by trimesh.transformations.

        Returns:
            Dictionary with translation, scale, and euler_deg.
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")

        translation = matrix[:3, 3].copy()
        rotation_scale = matrix[:3, :3].copy()
        scale = np.linalg.norm(rotation_scale, axis=0)
        safe_scale = np.where(scale == 0, 1.0, scale)
        rotation_normalized = rotation_scale / safe_scale

        rotation_4x4 = np.eye(4)
        rotation_4x4[:3, :3] = rotation_normalized

        try:
            euler_rad = trimesh.transformations.euler_from_matrix(rotation_4x4, axes=euler_axes)
            euler_deg = np.degrees(euler_rad)
        except Exception as exc:
            logger.warning(f"Failed to extract Euler angles with {euler_axes}: {exc}")
            euler_deg = np.array([0.0, 0.0, 0.0])

        return {
            "translation": translation,
            "scale": scale,
            "euler_deg": euler_deg,
        }

    @staticmethod
    def viewer_to_blender_matrix() -> np.ndarray:
        """
        Convert from the GLTF/UH4D viewer frame (Y-up) to Blender frame (Z-up).

        This matches the conversion implicitly used by the current render/export path.
        """
        rotation = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        mirror = np.eye(4)
        mirror[1, 1] = -1
        mirror[2, 2] = -1
        return mirror @ rotation

    @staticmethod
    def blender_to_viewer_matrix() -> np.ndarray:
        """Convert from Blender frame (Z-up) back to the viewer frame (Y-up)."""
        return np.linalg.inv(MatrixUtils.viewer_to_blender_matrix())

    @staticmethod
    def uniform_scale_matrix(scale_factor: float) -> np.ndarray:
        matrix = np.eye(4)
        matrix[0, 0] = scale_factor
        matrix[1, 1] = scale_factor
        matrix[2, 2] = scale_factor
        return matrix


class ElevationService:
    """Service for fetching elevation data from OpenTopoData API."""

    DEFAULT_DATASET = "srtm30m"
    API_URL = "https://api.opentopodata.org/v1/{dataset}"

    @staticmethod
    def get_elevation(lat: float, lon: float, dataset: str = DEFAULT_DATASET) -> float:
        url = ElevationService.API_URL.format(dataset=dataset)
        params = {"locations": f"{lat},{lon}"}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "results" in data and data["results"]:
                return data["results"][0].get("elevation", 0)

            logger.warning("⚠️ No elevation data available")
            return 0
        except requests.RequestException as exc:
            logger.error(f"❌ Failed to fetch elevation: {exc}")
            return 0


class GeoidConverter:
    """Utility class for converting orthometric to ellipsoid heights."""

    @staticmethod
    def get_geoid_undulation(lat: float, lon: float) -> float:
        try:
            crs_wgs84_3d = CRS.from_epsg(4979)
            crs_geoid = CRS.compound_crs([
                CRS.from_epsg(4326),
                CRS.from_epsg(5773),
            ])
            transformer = Transformer.from_crs(crs_geoid, crs_wgs84_3d, always_xy=True)
            _, _, geoid_undulation = transformer.transform(lon, lat, 0.0)
            return geoid_undulation
        except Exception as exc:
            logger.warning(f"⚠️ Failed to compute geoid undulation via PROJ: {exc}")
            return GeoidConverter._approximate_geoid_undulation(lat, lon)

    @staticmethod
    def _approximate_geoid_undulation(lat: float, lon: float) -> float:
        if 35 <= lat <= 72 and -10 <= lon <= 40:
            return 45.0
        if 25 <= lat <= 55 and -130 <= lon <= -60:
            return -30.0
        if 10 <= lat <= 55 and 60 <= lon <= 150:
            return -20.0
        logger.warning("Using fallback global average geoid undulation (0m)")
        return 0.0

    @staticmethod
    def orthometric_to_ellipsoid(lat: float, lon: float, orthometric_height: float) -> float:
        geoid_undulation = GeoidConverter.get_geoid_undulation(lat, lon)
        ellipsoid_height = orthometric_height + geoid_undulation
        logger.info(
            f"Height conversion: {orthometric_height:.2f}m (orthometric) + "
            f"{geoid_undulation:.2f}m (geoid) = {ellipsoid_height:.2f}m (ellipsoid)"
        )
        return ellipsoid_height


class ModelAnalyzer:
    """Utility class for analyzing mesh geometry."""

    @staticmethod
    def get_centroid(model: Union[trimesh.Scene, trimesh.Trimesh]) -> np.ndarray:
        if isinstance(model, trimesh.Scene):
            return model.bounds.mean(axis=0)
        return model.center_mass

    @staticmethod
    def apply_transform(model: Union[trimesh.Scene, trimesh.Trimesh], matrix: np.ndarray) -> None:
        if isinstance(model, trimesh.Scene):
            for geom in model.geometry.values():
                geom.apply_transform(matrix)
        else:
            model.apply_transform(matrix)


class GeoTransformer:
    """Export UH4D-compatible georeferencing metadata as XLSX."""

    def __init__(
        self,
        working_dir: str,
        input_file: str,
        output_folder: str,
        lat: float,
        lon: float,
        pipeline_scale: float = 1.0,
        viewer_euler_axes: str = "ryxz",
    ):
        self.working_dir = working_dir
        self.input_file = input_file
        self.lat = float(lat)
        self.lon = float(lon)
        self.pipeline_scale = float(pipeline_scale)
        self.viewer_euler_axes = viewer_euler_axes
        self.basename = os.path.splitext(os.path.basename(input_file))[0]
        self.output_folder = os.path.join(output_folder, self.basename)
        os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def compute_web_mercator_scale_factor(lat: float) -> float:
        return 1.0 / np.cos(np.radians(lat))

    def _load_transformation_matrix(self) -> Optional[np.ndarray]:
        matrix_path = os.path.join(self.working_dir, "transformation.txt")
        if not os.path.exists(matrix_path):
            logger.error("⚠️ Missing transformation.txt file")
            return None
        try:
            return MatrixUtils.load_matrix(matrix_path)
        except Exception as exc:
            logger.error(f"❌ Failed to load transformation matrix: {exc}")
            return None

    def _load_blender_matrix(self) -> Optional[np.ndarray]:
        """Load matrix_blender.json exactly as generated by Blender."""
        matrix_path = os.path.join(self.working_dir, "matrix_blender.json")
        if not os.path.exists(matrix_path):
            logger.warning("⚠️ matrix_blender.json not found")
            return None

        try:
            with open(matrix_path, "r", encoding="utf-8") as file_handle:
                return np.array(json.load(file_handle), dtype=np.float64)
        except Exception as exc:
            logger.error(f"❌ Failed to load Blender matrix: {exc}")
            return None

    def _load_scaled_model(self) -> Optional[Union[trimesh.Scene, trimesh.Trimesh]]:
        model_path = os.path.join(self.working_dir, f"{self.basename}_scaled.glb")
        if not os.path.exists(model_path):
            logger.error(f"❌ Model not found: {model_path}")
            return None
        try:
            return trimesh.load(model_path)
        except Exception as exc:
            logger.error(f"❌ Failed to load model: {exc}")
            return None

    def _normalize_dim_matrix(self, matrix_dim: np.ndarray) -> np.ndarray:
        """
        Rebuild DIM transform as translation + rotation + uniform scale.

        The DIM output is 2D affine in the Blender map frame. Rebuilding it as a
        similarity transform avoids exporting shear into the viewer metadata.
        """
        params = MatrixUtils.decompose(matrix_dim, euler_axes="sxyz")
        rotation = trimesh.transformations.euler_matrix(
            np.radians(params["euler_deg"][0]),
            np.radians(params["euler_deg"][1]),
            np.radians(params["euler_deg"][2]),
            axes="sxyz",
        )
        translation = trimesh.transformations.translation_matrix(params["translation"])
        uniform_scale = MatrixUtils.uniform_scale_matrix(np.mean(params["scale"][:2]))
        return translation @ rotation @ uniform_scale

    def _compute_refinement_translation(self, matrix_dim: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute the refinement translation from the temporary scaled mesh.

        The temporary mesh is already the same one used by DIM, so we only need to:
        1. convert it to Blender frame,
        2. apply the DIM matrix,
        3. compute the centroid,
        4. bring that centroid to the origin.
        """
        model = self._load_scaled_model()
        if model is None:
            return None

        viewer_to_blender = MatrixUtils.viewer_to_blender_matrix()
        ModelAnalyzer.apply_transform(model, viewer_to_blender)
        ModelAnalyzer.apply_transform(model, matrix_dim)
        centroid = ModelAnalyzer.get_centroid(model)
        return trimesh.transformations.translation_matrix(-centroid)

    def _export_xlsx(
        self,
        latitude: float,
        longitude: float,
        height: float,
        translation: np.ndarray,
        euler_deg: np.ndarray,
        scale: np.ndarray,
        web_mercator_factor: float,
    ) -> None:
        try:
            from openpyxl import Workbook
        except ImportError as exc:
            logger.error("❌ openpyxl is not installed. Run: pip install openpyxl")
            raise exc

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Georeferencing"

        headers = [
            "model_basename",
            "latitude",
            "longitude",
            "height_m_ellipsoid",
            "translation_x",
            "translation_y",
            "translation_z",
            "rotation_x_deg",
            "rotation_y_deg",
            "rotation_z_deg",
            "scale_x",
            "scale_y",
            "scale_z",
            "euler_convention",
            "height_reference",
            "web_mercator_factor",
        ]
        sheet.append(headers)

        row = [
            self.basename,
            float(latitude),
            float(longitude),
            float(height),
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
            float(euler_deg[0]),
            float(euler_deg[1]),
            float(euler_deg[2]),
            float(scale[0]),
            float(scale[1]),
            float(scale[2]),
            "YXZ (intrinsic/local, trimesh axes='ryxz')",
            "WGS84 ellipsoid height (EGM96 geoid conversion applied)",
            float(web_mercator_factor),
        ]
        sheet.append(row)

        output_path = os.path.join(self.output_folder, f"{self.basename}_georef.xlsx")
        workbook.save(output_path)
        logger.info(f"✅ UH4D georeferencing XLSX saved to: {output_path}")

    def run(self) -> bool:
        """
        Compute UH4D-compatible georeferencing metadata.

        The exported transform is expressed in the viewer frame (Y-up) and uses
        Euler order YXZ. No Cesium-specific axis corrections are applied.
        """
        matrix_dim_raw = self._load_transformation_matrix()
        if matrix_dim_raw is None:
            return False

        matrix_dim = self._normalize_dim_matrix(matrix_dim_raw)
        blender_matrix = self._load_blender_matrix()

        translation_refinement = self._compute_refinement_translation(matrix_dim)
        if translation_refinement is None:
            return False

        viewer_to_blender = MatrixUtils.viewer_to_blender_matrix()
        blender_to_viewer = MatrixUtils.blender_to_viewer_matrix()

        pipeline_scale_matrix = MatrixUtils.uniform_scale_matrix(self.pipeline_scale)
        web_mercator_factor = self.compute_web_mercator_scale_factor(self.lat)
        web_mercator_scale = MatrixUtils.uniform_scale_matrix(1.0 / web_mercator_factor)

        # Temporary scaled model in Blender frame:
        #   S_pipeline @ M_blender @ C_viewer_to_blender @ original_model
        # Final exported viewer transform:
        #   C_blender_to_viewer @ S_webmercator @ T_refine @ M_dim @ (...above...)
        inner_chain = matrix_dim
        if blender_matrix is not None:
            inner_chain = inner_chain @ pipeline_scale_matrix @ blender_matrix @ viewer_to_blender
        else:
            logger.warning(
                "⚠️ Exporting TRS relative to the temporary scaled mesh because "
                "matrix_blender.json is missing."
            )
            inner_chain = inner_chain @ viewer_to_blender

        final_matrix = (
            blender_to_viewer @
            web_mercator_scale @
            translation_refinement @
            inner_chain
        )

        trs = MatrixUtils.decompose(final_matrix, euler_axes=self.viewer_euler_axes)

        elevation_orthometric = ElevationService.get_elevation(self.lat, self.lon)
        elevation_ellipsoid = GeoidConverter.orthometric_to_ellipsoid(
            self.lat,
            self.lon,
            elevation_orthometric,
        )

        try:
            self._export_xlsx(
                latitude=self.lat,
                longitude=self.lon,
                height=elevation_ellipsoid,
                translation=trs["translation"],
                euler_deg=trs["euler_deg"],
                scale=trs["scale"],
                web_mercator_factor=web_mercator_factor,
            )
        except Exception as exc:
            logger.error(f"❌ Failed to export XLSX: {exc}")
            return False

        return True