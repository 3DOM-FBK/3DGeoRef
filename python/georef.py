import cv2
import argparse
import pycolmap
import numpy as np
import rasterio as rio

from transformations import affine_matrix_from_points
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Georeference a raster image using a world file.")
    parser.add_argument("--ortho", type=Path, help="Path to ortho image.", default="./colosseo.tif")
    parser.add_argument("--render", type=Path, help="Path to render image.", default="./top_view.png")
    parser.add_argument("--output", type=Path, help="Output file.", default="./output.txt")
    parser.add_argument("--database", type=Path, help="Path to database file.", default="./database.db")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(f"Georeferencing '{args.render}' with '{args.ortho}' and saving to '{args.output}'. Database: '{args.database}'")

    ortho_path = args.ortho
    render_path = args.render
    output_path = args.output
    database_path = args.database
    debug = args.debug
    images ={}


    # Load the ortho image to get its georeferencing information
    with rio.open(ortho_path) as ortho:
        transform = np.array(ortho.transform).reshape((3,3))
        crs = ortho.crs
        print(f"Ortho image CRS: {crs}")
        print(f"Transform: \n{transform}")
    

    # Load keypoints and matches from the colmap database
    db = pycolmap.Database(database_path)
    for i in db.read_all_images():
        if i.name == render_path.name:
            images[i.name] = {
                "image_id": i.image_id,
                "camera_id": i.camera_id,
                "name": i.name,
                "keypoints": db.read_keypoints(i.camera_id),
            }
        elif i.name == ortho_path.name:
            images[i.name] = {
                "image_id": i.image_id,
                "camera_id": i.camera_id,
                "name": i.name,
                "keypoints": db.read_keypoints(i.camera_id),
            }
        else:
            print("Images passed with argparse do not match images in the database.")
            quit()

    matches = db.read_two_view_geometry(images[render_path.name]["image_id"], images[ortho_path.name]["image_id"])
    if matches is None:
        matches = db.read_two_view_geometry(images[ortho_path.name]["image_id"], images[render_path.name]["image_id"])
        match_indexes_render = matches.inlier_matches[:,1]
        match_indexes_ortho = matches.inlier_matches[:,0]
        if matches is None:
            print("No matches found between the render and ortho images.")
            quit()
    else:
        match_indexes_render = matches.inlier_matches[:,0]
        match_indexes_ortho = matches.inlier_matches[:,1]


    # Estimate homograpy
    keypoints_render = images[render_path.name]["keypoints"]
    keypoints_ortho = images[ortho_path.name]["keypoints"]

    pts_render = keypoints_render[match_indexes_render][:, :2].astype(np.float32)
    pts_ortho  = keypoints_ortho[match_indexes_ortho][:, :2].astype(np.float32)

    H, maskH = cv2.findHomography(pts_render, pts_ortho, cv2.RANSAC)
    A, maskA = cv2.estimateAffine2D(pts_render, pts_ortho, method=cv2.RANSAC)
    if A is not None:
        A = np.vstack([A, [0, 0, 1]])
    else:
        print("Affine transformation could not be estimated.")
        quit()
    pts_render = pts_render[maskA.ravel() == 0]
    pts_ortho = pts_ortho[maskA.ravel() == 0]
    # print(f"Estimated homography:\n{H}")
    # print(f"Estimated affine:\n{A}")

    img_render = cv2.imread(str(render_path))
    img_ortho = cv2.imread(str(ortho_path))

    if debug:
        height, width = img_ortho.shape[:2]
        warped = cv2.warpPerspective(img_render, A, (width, height))
        cv2.imwrite('./warped.png', warped)


    # Export original tiepoints for debugging
    if debug:
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
    
    if debug:
        with open("./transformed_ortho_tiepoints.txt", 'w') as f:
            for pt in pts_ortho:
                f.write(f"{pt[0]}, {pt[1]}\n")



    # Transformation to be apllied to rendered view
    T = np.dot(transform, A)
    print(f"Transformation matrix:\n{T}")
    
    with open(output_path, 'w') as f:
        f.write(f"{T[0, 0]} {T[0, 1]} 0.0 {T[0, 2]}\n")
        f.write(f"{T[1, 0]} {T[1, 1]} 0.0 {T[1, 2]}\n")
        f.write(f"0.0 0.0 1.0 0.0\n")
        f.write(f"0.0 0.0 0.0 1.0\n")

    for i in range(pts_render.shape[0]):
        point = np.array([pts_render[i, 0], pts_render[i, 1], 1]).reshape((3, 1))
        transformed_point = np.dot(T, point)
        pts_render[i, 0] = transformed_point[0, 0]
        pts_render[i, 1] = transformed_point[1, 0]

    if debug:
        with open("./transformed_render_tiepoints.txt", 'w') as f:
            for pt in pts_render:
                f.write(f"{pt[0]}, {pt[1]}\n")


if __name__ == "__main__":
    main()