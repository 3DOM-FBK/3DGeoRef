"""
Minimal GeoTransformer

This module applies the DIM transformation matrix directly to the temporary
scaled model (*_scaled.glb) and overwrites the same file.
"""

import logging
import os
from typing import Optional, Union

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


class MatrixUtils:
    """Matrix helpers."""

    @staticmethod
    def load_matrix(file_path: str) -> np.ndarray:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Matrix file not found: {file_path}")

        matrix = np.loadtxt(file_path, delimiter=None)
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, found {matrix.shape}")
        return matrix.astype(np.float64)

    @staticmethod
    def decompose_trs(matrix: np.ndarray) -> dict:
        """
        Decompose a 4x4 transform into translation, rotation (Euler XYZ deg), and scale.
        """
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, found {m.shape}")

        translation = m[:3, 3].copy()

        rs = m[:3, :3].copy()
        scale = np.linalg.norm(rs, axis=0)
        safe = np.where(scale == 0.0, 1.0, scale)
        rotation = rs / safe

        r4 = np.eye(4, dtype=np.float64)
        r4[:3, :3] = rotation
        try:
            euler_rad = trimesh.transformations.euler_from_matrix(r4, axes="sxyz")
            euler_deg = np.degrees(euler_rad)
        except Exception:
            euler_deg = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        return {
            "translation": translation,
            "rotation_euler_deg_xyz": euler_deg,
            "scale": scale,
        }


class ModelAnalyzer:
    """Geometry helpers."""

    @staticmethod
    def apply_transform(model: Union[trimesh.Scene, trimesh.Trimesh], matrix: np.ndarray) -> None:
        if isinstance(model, trimesh.Scene):
            for geom in model.geometry.values():
                geom.apply_transform(matrix)
        else:
            model.apply_transform(matrix)


class PivotCalculator:
    """
    Computes the full transform chain (original mesh → final georeferenced position)
    and returns the X, Y position of the model pivot after the transformation.

    Transform chain:
        M_total = R_x(-90°) @ P @ M_dim @ P⁻¹ @ S(metric_scale) @ M_blender

    where:
        - M_blender     : 4x4 matrix captured from Blender (Step 1)
        - metric_scale  : uniform scale factor applied in Step 2 (metres/pixel)
        - M_dim         : raw DIM matrix loaded from transformation.txt
        - P             : axis-permutation matrix (DIM → trimesh basis)
        - R_x(-90°)     : fixed -90° rotation around X applied in Step 6

    The pivot (X, Y) is the centre of the model bounding box projected onto the
    XY plane after M_total has been applied to the input mesh vertices.
    """

    def __init__(
        self,
        input_mesh_path: str,
        blender_matrix: np.ndarray,
        metric_scale: float,
        dim_matrix: np.ndarray,
    ):
        """
        Args:
            input_mesh_path : Path to the original (pre-pipeline) mesh file.
            blender_matrix  : 4x4 matrix from Blender Step 1.
            metric_scale    : Uniform scale factor from Step 2 (metres per pixel).
            dim_matrix      : Raw 4x4 DIM matrix from transformation.txt.
        """
        self.input_mesh_path = input_mesh_path
        self.blender_matrix = np.asarray(blender_matrix, dtype=np.float64)
        self.metric_scale = float(metric_scale)
        self.dim_matrix = np.asarray(dim_matrix, dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal helpers (mirrors GeoTransformer logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_matrix(s: float) -> np.ndarray:
        """Uniform scale 4x4 matrix."""
        m = np.eye(4, dtype=np.float64)
        m[0, 0] = m[1, 1] = m[2, 2] = s
        return m

    @staticmethod
    def _axis_permutation_matrix() -> np.ndarray:
        """P: DIM basis → trimesh basis  (X→X, Y→Z, Z→Y)."""
        p = np.eye(4, dtype=np.float64)
        p[:3, :3] = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        return p

    @staticmethod
    def _rotation_x_minus90() -> np.ndarray:
        """4x4 rotation matrix: -90° around X."""
        return trimesh.transformations.rotation_matrix(np.radians(-90.0), [1.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_total_matrix(self) -> np.ndarray:
        """
        Returns the 4x4 matrix that maps the original input mesh to its final
        georeferenced position:

            M_total = R_x(-90°) @ (P @ M_dim @ P⁻¹) @ S(metric_scale) @ M_blender
        """
        S = self._scale_matrix(self.metric_scale)
        P = self._axis_permutation_matrix()
        P_inv = np.linalg.inv(P)
        M_dim_trimesh = P @ self.dim_matrix @ P_inv
        R = self._rotation_x_minus90()

        return R @ M_dim_trimesh @ S @ self.blender_matrix

    def compute_pivot_xy(self) -> tuple:
        """
        Loads the input mesh, applies M_total, and returns the (X, Y) coordinates
        of the bounding-box centre in the final frame.

        Returns:
            (pivot_x, pivot_y, M_total) — floats + 4x4 ndarray
        """
        if not os.path.exists(self.input_mesh_path):
            raise FileNotFoundError(f"Input mesh not found: {self.input_mesh_path}")

        M_total = self.compute_total_matrix()

        mesh = trimesh.load(self.input_mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            meshes = list(mesh.geometry.values())
            vertices = np.vstack([m.vertices for m in meshes])
        else:
            vertices = mesh.vertices

        # Apply M_total to all vertices
        ones = np.ones((len(vertices), 1), dtype=np.float64)
        verts_h = np.hstack([vertices, ones])          # (N, 4)
        transformed = (M_total @ verts_h.T).T          # (N, 4)
        xyz = transformed[:, :3]

        # Bounding box centre XY
        bb_min = xyz.min(axis=0)
        bb_max = xyz.max(axis=0)
        centre = (bb_min + bb_max) / 2.0

        pivot_x = float(centre[0])
        pivot_y = float(centre[1])

        logger.debug("📍 Pivot calculation result")
        logger.debug(f"   M_total (shape): {M_total.shape}")
        trs = MatrixUtils.decompose_trs(M_total)
        t = trs["translation"]
        r = trs["rotation_euler_deg_xyz"]
        s = trs["scale"]
        logger.debug(f"   Total Translation: tx={t[0]:.4f}, ty={t[1]:.4f}, tz={t[2]:.4f}")
        logger.debug(f"   Total Rotation°  : rx={r[0]:.2f}, ry={r[1]:.2f}, rz={r[2]:.2f}")
        logger.debug(f"   Total Scale      : sx={s[0]:.6f}, sy={s[1]:.6f}, sz={s[2]:.6f}")
        logger.debug(f"   Bounding box min : {bb_min}")
        logger.debug(f"   Bounding box max : {bb_max}")
        logger.debug(f"   ➜  Pivot X = {pivot_x:.6f},  Pivot Y = {pivot_y:.6f}")


        return pivot_x, pivot_y, M_total


class GeoTransformer:
    """
    Loads *_scaled.glb from working_dir, applies DIM transform from
    transformation.txt, and overwrites *_scaled.glb.
    """

    def __init__(
        self,
        working_dir: str,
        input_file: str,
        output_folder: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        pipeline_scale: float = 1.0,
        **_: dict,
    ):
        self.working_dir = working_dir
        self.input_file = input_file
        self.output_folder = output_folder
        self.lat = lat
        self.lon = lon
        self.pipeline_scale = pipeline_scale

    def _find_scaled_model(self) -> Optional[str]:
        for name in sorted(os.listdir(self.working_dir)):
            if name.endswith("_scaled.glb"):
                return os.path.join(self.working_dir, name)
        return None

    def _load_dim_matrix(self) -> Optional[np.ndarray]:
        matrix_path = os.path.join(self.working_dir, "transformation.txt")
        try:
            return MatrixUtils.load_matrix(matrix_path)
        except Exception as exc:
            logger.error(f"❌ Failed to load DIM matrix: {exc}")
            return None

    @staticmethod
    def _dim_to_trimesh_basis_matrix() -> np.ndarray:
        """
        Axis mapping deduced from DIM -> trimesh observations:

            X_trimesh = X_dim
            Y_trimesh = Z_dim
            Z_trimesh = Y_dim

        The full transform is converted with conjugation:

            M_trimesh = P @ M_dim @ P^{-1}
        """
        p = np.eye(4, dtype=np.float64)
        p[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        return p

    def _dim_to_trimesh_matrix(self, dim_matrix: np.ndarray) -> np.ndarray:
        """Convert DIM 4x4 transform into trimesh frame via axis permutation."""
        p = self._dim_to_trimesh_basis_matrix()
        p_inv = np.linalg.inv(p)
        return p @ dim_matrix @ p_inv

    @staticmethod
    def _rotation_x_minus_90_matrix() -> np.ndarray:
        """Return a 4x4 rotation matrix for -90 degrees around X axis."""
        return trimesh.transformations.rotation_matrix(np.radians(-90.0), [1.0, 0.0, 0.0])

    def run(self) -> bool:
        """Apply DIM transform converted to trimesh frame and overwrite *_scaled.glb."""
        model_path = self._find_scaled_model()
        if model_path is None:
            logger.error("❌ No *_scaled.glb found in working directory.")
            return False

        dim_matrix = self._load_dim_matrix()
        if dim_matrix is None:
            return False

        # Print DIM matrix as Translation / Rotation / Scale
        try:
            trs = MatrixUtils.decompose_trs(dim_matrix)
            t = trs["translation"]
            r = trs["rotation_euler_deg_xyz"]
            s = trs["scale"]
            logger.debug("📐 DIM matrix decomposition (raw transformation.txt)")
            logger.debug(f"   Translation: tx={t[0]:.6f}, ty={t[1]:.6f}, tz={t[2]:.6f}")
            logger.debug(f"   Rotation   : rx={r[0]:.6f}°, ry={r[1]:.6f}°, rz={r[2]:.6f}°  (Euler XYZ)")
            logger.debug(f"   Scale      : sx={s[0]:.6f}, sy={s[1]:.6f}, sz={s[2]:.6f}")
        except Exception as exc:
            logger.warning(f"⚠️ Failed to decompose DIM matrix into TRS: {exc}")

        logger.info(f"🔧 Applying DIM transform to: {model_path}")
        try:
            model = trimesh.load(model_path)

            # 1) Convert DIM matrix to trimesh basis using explicit axis mapping, then apply
            matrix_model = self._dim_to_trimesh_matrix(dim_matrix)

            try:
                trs_mapped = MatrixUtils.decompose_trs(matrix_model)
                t2 = trs_mapped["translation"]
                r2 = trs_mapped["rotation_euler_deg_xyz"]
                s2 = trs_mapped["scale"]
                logger.debug("📐 Converted matrix decomposition (DIM -> trimesh)")
                logger.debug(f"   Translation: tx={t2[0]:.6f}, ty={t2[1]:.6f}, tz={t2[2]:.6f}")
                logger.debug(f"   Rotation   : rx={r2[0]:.6f}°, ry={r2[1]:.6f}°, rz={r2[2]:.6f}°  (Euler XYZ)")
                logger.debug(f"   Scale      : sx={s2[0]:.6f}, sy={s2[1]:.6f}, sz={s2[2]:.6f}")
            except Exception as exc:
                logger.warning(f"⚠️ Failed to decompose converted matrix into TRS: {exc}")

            ModelAnalyzer.apply_transform(model, matrix_model)
            logger.info("✅ Applied DIM transform with DIM->trimesh axis conversion")

            # 2) Apply additional fixed rotation around X axis (-90 deg)
            rot_x_m90 = self._rotation_x_minus_90_matrix()
            ModelAnalyzer.apply_transform(model, rot_x_m90)
            logger.info("✅ Applied extra rotation: X axis = -90°")

            model.export(model_path)
            logger.info(f"✅ Overwritten transformed model: {model_path}")
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to transform/export model: {exc}")
            return False
