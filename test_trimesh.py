"""
test_trimesh.py

Carica una nuvola di punti (PLY/OBJ/GLB/ecc.), applica traslazione,
rotazione ed scala specificati dall'utente, e salva il risultato come GLB
in una cartella di output.

Uso:
    python test_trimesh.py \
        --input  path/to/pointcloud.ply \
        --output path/to/output_dir \
        --tx 1.0 --ty 2.0 --tz 0.5 \
        --rx 0   --ry 90  --rz 0 \
        --sx 1.0 --sy 1.0 --sz 1.0

I parametri di rotazione sono in gradi (Euler XYZ, applicati nell'ordine X→Y→Z).
"""

import argparse
import os
import sys

import numpy as np
import trimesh
import trimesh.transformations as tf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_transform(tx: float, ty: float, tz: float,
                    rx_deg: float, ry_deg: float, rz_deg: float,
                    sx: float, sy: float, sz: float) -> np.ndarray:
    """
    Costruisce la matrice 4×4 TRS (Traslazione × Rotazione × Scala).
    Ordine di composizione: prima Scala, poi Rotazione XYZ, poi Traslazione.
    """
    # Scala
    S = np.diag([sx, sy, sz, 1.0])

    # Rotazione Euler XYZ (in radianti)
    rx, ry, rz = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
    R = tf.euler_matrix(rx, ry, rz, axes='sxyz')  # 4×4

    # Traslazione
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]

    # TRS = T @ R @ S
    M = T @ R @ S
    return M


def decompose_and_print(M: np.ndarray, label: str = "Transform") -> None:
    """Stampa la decomposizione TRS della matrice per debug."""
    # Traslazione
    t = M[:3, 3]

    # Scala (norme colonne della sotto-matrice 3×3)
    RS = M[:3, :3]
    s = np.linalg.norm(RS, axis=0)

    # Matrice di rotazione pura
    R_pure = RS / s[np.newaxis, :]

    # Angoli Euler XYZ in gradi
    R4 = np.eye(4)
    R4[:3, :3] = R_pure
    angles_rad = tf.euler_from_matrix(R4, axes='sxyz')
    angles_deg = np.degrees(angles_rad)

    print(f"\n[{label}]")
    print(f"  Traslazione : tx={t[0]:.4f}  ty={t[1]:.4f}  tz={t[2]:.4f}")
    print(f"  Rotazione°  : rx={angles_deg[0]:.2f}  ry={angles_deg[1]:.2f}  rz={angles_deg[2]:.2f}")
    print(f"  Scala       : sx={s[0]:.6f}  sy={s[1]:.6f}  sz={s[2]:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carica una nuvola di punti, applica TRS e salva come GLB."
    )
    parser.add_argument("--input",  required=True, help="File di input (PLY, OBJ, GLB, ecc.)")
    parser.add_argument("--output", required=True, help="Cartella di output")

    parser.add_argument("--tx", type=float, default=0.0, help="Traslazione X")
    parser.add_argument("--ty", type=float, default=0.0, help="Traslazione Y")
    parser.add_argument("--tz", type=float, default=0.0, help="Traslazione Z")

    parser.add_argument("--rx", type=float, default=0.0, help="Rotazione attorno X (gradi)")
    parser.add_argument("--ry", type=float, default=0.0, help="Rotazione attorno Y (gradi)")
    parser.add_argument("--rz", type=float, default=0.0, help="Rotazione attorno Z (gradi)")

    parser.add_argument("--sx", type=float, default=1.0, help="Scala X")
    parser.add_argument("--sy", type=float, default=1.0, help="Scala Y")
    parser.add_argument("--sz", type=float, default=1.0, help="Scala Z")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Caricamento
    # ------------------------------------------------------------------
    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"[ERRORE] File non trovato: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[1] Caricamento: {input_path}")
    scene_or_mesh = trimesh.load(input_path, force="scene")

    # Supporta sia Scene che Mesh/PointCloud
    if isinstance(scene_or_mesh, trimesh.Scene):
        scene = scene_or_mesh
        geom_count = len(scene.geometry)
        print(f"    Scene con {geom_count} geometri{'a' if geom_count == 1 else 'e'}")
    else:
        # Wrappa in una Scene per uniformità
        scene = trimesh.Scene(geometry={"mesh": scene_or_mesh})
        print(f"    Geometria singola ({type(scene_or_mesh).__name__}) wrappata in Scene")

    # Bounding box prima della trasformazione
    bounds = scene.bounds
    if bounds is not None:
        size = bounds[1] - bounds[0]
        print(f"    Bounding box originale: {size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}")

    # ------------------------------------------------------------------
    # 2. Costruzione matrice TRS
    # ------------------------------------------------------------------
    M = build_transform(
        args.tx, args.ty, args.tz,
        args.rx, args.ry, args.rz,
        args.sx, args.sy, args.sz,
    )
    print("\n[2] Matrice di trasformazione (4×4):")
    print(np.array2string(M, precision=6, suppress_small=True))
    decompose_and_print(M, label="TRS decomposizione")

    # ------------------------------------------------------------------
    # 3. Applicazione trasformazione
    # ------------------------------------------------------------------
    print("\n[3] Applicazione trasformazione...")
    scene.apply_transform(M)

    bounds_after = scene.bounds
    if bounds_after is not None:
        size_after = bounds_after[1] - bounds_after[0]
        print(f"    Bounding box dopo: {size_after[0]:.4f} x {size_after[1]:.4f} x {size_after[2]:.4f}")

    # ------------------------------------------------------------------
    # 4. Salvataggio GLB
    # ------------------------------------------------------------------
    os.makedirs(args.output, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(args.output, f"{base_name}_transformed.glb")

    print(f"\n[4] Salvataggio: {output_path}")
    scene.export(output_path)
    print(f"    Fatto! ({os.path.getsize(output_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
