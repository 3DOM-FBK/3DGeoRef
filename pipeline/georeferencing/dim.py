import cv2, os
import argparse
import pycolmap
import numpy as np
import rasterio as rio

from pipeline.utils.transformations import affine_matrix_from_points
from pathlib import Path


class georef_dim():

    def __init__(self, ortho_path, render_path, output_path, database_path, debug=False):
        self.ortho_path = Path(ortho_path)
        self.render_path = Path(render_path)
        self.output_path = Path(output_path)
        self.database_path = Path(database_path)
        self.debug = debug


    # ===== Function: SelectPair =====
    def SelectPair(self, database: pycolmap.Database = None, render_name: str = "", ortho_name: str = ""):
        """
        Selects the best matching pair of images (rendered view and orthophoto) based on the number of inlier matches.
        Args:
            database (pycolmap.Database): The COLMAP database containing image and match information.
            render_name (str): The filename of the rendered view image.
            ortho_name (str): The filename of the orthophoto image.
        Returns:
            tuple: The filenames of the best matching render and ortho images.
        """
        # Prepare stems and extensions
        render_stem = Path(render_name).stem
        render_ext = render_name.split(".")[-1]
        ortho_stem = Path(ortho_name).stem
        ortho_ext = ortho_name.split(".")[-1]

        # Define possible
        rotations = ["", "_rot90", "_rot180", "_rot270"]
        scales = ["", "_s_0_25", "_s_0_50", "_s_0_75"]

        all_pairs = [
            (f"{render_stem}{r}.{render_ext}", f"{ortho_stem}{s}.{ortho_ext}")
            for r in rotations
            for s in scales
        ]

        # Read all images from database
        images_dict = {img.name: img.image_id for img in database.read_all_images()}

        # Count inlier matches for each pair
        all_matches = []
        for render_file, ortho_file in all_pairs:
            matches = database.read_two_view_geometry(
                images_dict[render_file],
                images_dict[ortho_file]
            )
            all_matches.append(matches.inlier_matches.shape[0])

        # Select the pair with the maximum inlier matches
        best_index = int(np.argmax(all_matches))

        # print (f"Selected pair: '{all_pairs[best_index][0]}' and '{all_pairs[best_index][1]}'")

        return all_pairs[best_index]


    # ==== Method: run_pipeline =====
    def run_pipeline(self):
        """
        Runs the georeferencing pipeline to align a rendered view with an orthophoto using feature matching.
        """
        images = {}

        # Load ortho image metadata
        with rio.open(self.ortho_path) as ortho:
            transform = np.array(ortho.transform).reshape((3, 3))
            crs = ortho.crs

        # Load database and select best pair
        db = pycolmap.Database(self.database_path)
        pair = self.SelectPair(db, self.render_path.name, self.ortho_path.name)
        self.render_path = self.render_path.parent / pair[0]
        self.ortho_path = self.ortho_path.parent / pair[1]

        # print(f"Selected pair: '{self.render_path}' and '{self.ortho_path}'")

        # Determine rotation and scale from filenames
        rotation = 0
        for rot_str, rot_deg in [("rot90", 90), ("rot180", 180), ("rot270", 270)]:
            if rot_str in self.render_path.stem:
                rotation = rot_deg
                break

        scale = 1.0
        for s_str, s_val in [("s_0_25", 0.25), ("s_0_50", 0.5), ("s_0_75", 0.75)]:
            if s_str in self.ortho_path.stem:
                scale = s_val
                break

        # Load keypoints for selected images
        image_names = [self.render_path.name, self.ortho_path.name]
        for img in db.read_all_images():
            if img.name in image_names:
                images[img.name] = {
                    "image_id": img.image_id,
                    "camera_id": img.camera_id,
                    "name": img.name,
                    "keypoints": db.read_keypoints(img.camera_id),
                }

        # Read matches between the selected images
        matches = db.read_two_view_geometry(
            images[self.render_path.name]["image_id"],
            images[self.ortho_path.name]["image_id"]
        )
        if matches is None:
            matches = db.read_two_view_geometry(
                images[self.ortho_path.name]["image_id"],
                images[self.render_path.name]["image_id"]
            )
            if matches is None:
                raise RuntimeError("No matches found between the two images.")
            match_indexes_render = matches.inlier_matches[:, 1]
            match_indexes_ortho = matches.inlier_matches[:, 0]
        else:
            match_indexes_render = matches.inlier_matches[:, 0]
            match_indexes_ortho = matches.inlier_matches[:, 1]

        # Load images
        img_render = cv2.imread(str(self.render_path))
        img_ortho = cv2.imread(str(self.ortho_path))
        keypoints_render = images[self.render_path.name]["keypoints"]
        img_h, img_w = img_render.shape[:2]

        # Apply rotation to render keypoints
        if rotation == 90:
            keypoints_render = np.array([[y, img_w - x] for x, y in keypoints_render])
        elif rotation == 180:
            keypoints_render = np.array([[img_w - x, img_h - y] for x, y in keypoints_render])
        elif rotation == 270:
            # keypoints_render = np.array([[img_w - y, x] for x, y in keypoints_render])
            keypoints_render = np.array([[img_h - y, x] for x, y in keypoints_render])

        # Scale ortho keypoints
        keypoints_ortho = np.rint(images[self.ortho_path.name]["keypoints"] / scale).astype(np.int32)

        # Select matching points
        pts_render = keypoints_render[match_indexes_render][:, :2].astype(np.float32)
        pts_ortho = keypoints_ortho[match_indexes_ortho][:, :2].astype(np.float32)

        # Compute transformations
        H, _ = cv2.findHomography(pts_render, pts_ortho, cv2.RANSAC)
        A, maskA = cv2.estimateAffine2D(pts_render, pts_ortho, method=cv2.RANSAC)
        if A is None:
            raise RuntimeError("Could not compute affine transformation")
        A = np.vstack([A, [0, 0, 1]])

        # Filter out points rejected by affine estimation
        pts_render = pts_render[maskA.ravel() == 0]
        pts_ortho = pts_ortho[maskA.ravel() == 0]

        # Export original tiepoints for self.debugging
        if self.debug:
            with open("./ortho_tiepoints.txt", 'w') as f:
                for pt in pts_ortho:
                    f.write(f"{pt[0]}, {pt[1]}\n")
            with open("./render_tiepoints.txt", 'w') as f:
                for pt in pts_render:
                    f.write(f"{pt[0]}, {pt[1]}\n")

        # Apply ortho image geotransform
        pts_ortho_h = np.hstack([pts_ortho, np.ones((pts_ortho.shape[0], 1))])
        pts_ortho = (transform @ pts_ortho_h.T).T[:, :2]

        # Compute final transformation to apply to rendered view
        T = transform @ A

        with open(self.output_path, 'w') as f:
            f.write(f"{T[0, 0]} {T[0, 1]} 0.0 {T[0, 2]}\n")
            f.write(f"{T[1, 0]} {T[1, 1]} 0.0 {T[1, 2]}\n")
            f.write("0.0 0.0 1.0 0.0\n")
            f.write("0.0 0.0 0.0 1.0\n")

        # Apply transformation to render points
        pts_render_h = np.hstack([pts_render, np.ones((pts_render.shape[0], 1))])
        pts_render = (T @ pts_render_h.T).T[:, :2]

        # Optionally save debug tiepoints
        if self.debug:
            for name, pts in [("ortho", pts_ortho), ("render", pts_render)]:
                with open(f"./transformed_{name}_tiepoints.txt", 'w') as f:
                    for pt in pts:
                        f.write(f"{pt[0]}, {pt[1]}\n")


# if __name__ == "__main__":
#     working_dir = "/tmp/8ce20ac5fc3a4af7ab223cdc0caa7d27"
#     base_name = "8ce20ac5fc3a4af7ab223cdc0caa7d27"
#     ortho_path = os.path.join(working_dir, "images", base_name + ".tif")
#     render_path = os.path.join(working_dir, "images", "top_view.png")
#     output_path = os.path.join(working_dir, "transformation.txt")
#     database_path = os.path.join(working_dir, "results_loftr_bruteforce_quality_medium", "database.db")
#     processor = georef_dim(ortho_path, render_path, output_path, database_path)

#     processor.run_pipeline()