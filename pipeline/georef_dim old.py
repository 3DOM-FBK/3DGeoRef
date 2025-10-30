import cv2
import argparse
import pycolmap
import numpy as np
import rasterio as rio

from transformations import affine_matrix_from_points
from pathlib import Path


class georef_dim():

    def __init__(self, ortho_path, render_path, output_path, database_path, debug=False):
        self.ortho_path = Path(ortho_path)
        self.render_path = Path(render_path)
        self.output_path = Path(output_path)
        self.database_path = Path(database_path)
        self.debug = debug


    # ==== Method: run_pipeline =====
    def run_pipeline(self):
        images ={}

        # Load the ortho image to get its georeferencing information
        with rio.open(self.ortho_path) as ortho:
            transform = np.array(ortho.transform).reshape((3, 3))
            crs = ortho.crs

        # Load keypoints and matches from the colmap database
        print (f"Database: '{self.database_path}'")
        db = pycolmap.Database(self.database_path)
        for i in db.read_all_images():
            if i.name == self.render_path.name:
                images[i.name] = {
                    "image_id": i.image_id,
                    "camera_id": i.camera_id,
                    "name": i.name,
                    "keypoints": db.read_keypoints(i.camera_id),
                }
            elif i.name == self.ortho_path.name:
                images[i.name] = {
                    "image_id": i.image_id,
                    "camera_id": i.camera_id,
                    "name": i.name,
                    "keypoints": db.read_keypoints(i.camera_id),
                }
            else:
                quit()

        matches = db.read_two_view_geometry(
            images[self.render_path.name]["image_id"],
            images[self.ortho_path.name]["image_id"]
        )
        if matches is None:
            matches = db.read_two_view_geometry(
                images[self.ortho_path.name]["image_id"],
                images[self.render_path.name]["image_id"]
            )
            match_indexes_render = matches.inlier_matches[:, 1]
            match_indexes_ortho = matches.inlier_matches[:, 0]
            if matches is None:
                quit()
        else:
            match_indexes_render = matches.inlier_matches[:, 0]
            match_indexes_ortho = matches.inlier_matches[:, 1]

        # Estimate homography
        keypoints_render = images[self.render_path.name]["keypoints"]
        keypoints_ortho = images[self.ortho_path.name]["keypoints"]

        pts_render = keypoints_render[match_indexes_render][:, :2].astype(np.float32)
        pts_ortho = keypoints_ortho[match_indexes_ortho][:, :2].astype(np.float32)

        H, maskH = cv2.findHomography(pts_render, pts_ortho, cv2.RANSAC)
        A, maskA = cv2.estimateAffine2D(pts_render, pts_ortho, method=cv2.RANSAC)
        if A is not None:
            A = np.vstack([A, [0, 0, 1]])
        else:
            quit()

        pts_render = pts_render[maskA.ravel() == 0]
        pts_ortho = pts_ortho[maskA.ravel() == 0]

        img_render = cv2.imread(str(self.render_path))
        img_ortho = cv2.imread(str(self.ortho_path))

        # Export original tiepoints for self.debugging
        if self.debug:
            with open("./ortho_tiepoints.txt", 'w') as f:
                for pt in pts_ortho:
                    f.write(f"{pt[0]}, {pt[1]}\n")
            with open("./render_tiepoints.txt", 'w') as f:
                for pt in pts_render:
                    f.write(f"{pt[0]}, {pt[1]}\n")

        # Apply transformation to ortho image
        for i in range(pts_ortho.shape[0]):
            point = np.array([pts_ortho[i, 0], pts_ortho[i, 1], 1]).reshape((3, 1))
            transformed_point = np.dot(transform, point)
            pts_ortho[i, 0] = transformed_point[0, 0]
            pts_ortho[i, 1] = transformed_point[1, 0]

        if self.debug:
            with open("./transformed_ortho_tiepoints.txt", 'w') as f:
                for pt in pts_ortho:
                    f.write(f"{pt[0]}, {pt[1]}\n")

        # Transformation to be applied to rendered view
        T = np.dot(transform, A)

        with open(self.output_path, 'w') as f:
            f.write(f"{T[0, 0]} {T[0, 1]} 0.0 {T[0, 2]}\n")
            f.write(f"{T[1, 0]} {T[1, 1]} 0.0 {T[1, 2]}\n")
            f.write(f"0.0 0.0 1.0 0.0\n")
            f.write(f"0.0 0.0 0.0 1.0\n")

        for i in range(pts_render.shape[0]):
            point = np.array([pts_render[i, 0], pts_render[i, 1], 1]).reshape((3, 1))
            transformed_point = np.dot(T, point)
            pts_render[i, 0] = transformed_point[0, 0]
            pts_render[i, 1] = transformed_point[1, 0]

        if self.debug:
            with open("./transformed_render_tiepoints.txt", 'w') as f:
                for pt in pts_render:
                    f.write(f"{pt[0]}, {pt[1]}\n")