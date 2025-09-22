# 3DGeoRef
The project automates the process of georeferencing 3D models using synthetic images and satellite data. Given a 3D model, the system generates synthetic views, estimates the geographic location, downloads satellite images, performs image matching, and computes the transformation to align the 3D model with real-world geographic coordinates. The final result is a georeferenced 3D model, ready to be used in GIS systems or 3D viewers.


## Modules and their functions

### main.py
-   **PipelineProcessor**: manages the entire pipeline, from importing the 3D model to the final georeferencing.
    -   Generates synthetic views using Blender.
    -   Estimates geolocation using images and GeoCLIP.
    -   Downloads satellite images via Google Maps API.
    -   Performs image matching (Deep Image Matching).
    -   Computes and applies the transformation matrix to the 3D model.
    -   Aligns the model to the correct elevation.
    -   Exports the georeferenced model.
-   Handles logging, argument parsing, and temporary directories.

### synthetic_imgs.py
-   Script executed in Blender to:
    -   Script executed in Blender to:
    -   Generate orthographic and “streetview” views of the model.
    -   Compute the bounding box and rendering parameters.
    -   Apply HDRI lighting.
    -   Post-process and scale the model.
    -   Export images and the scaled model.

### geoloc_img.py
-   Uses GeoCLIP to estimate the geographic position (latitude, longitude) from images.
-   Analyzes a folder of images, excludes top-down views, and returns the most likely location.
-   Outputs in JSON format.

### ortho.py
-   Downloads satellite tiles from the Google Maps API for a specific area.
-   Reconstructs a georeferenced mosaic (GeoTIFF) from the tiles.
-   Computes conversions between geographic coordinates and tiles, ground resolution, etc.
-   Provides helper functions for downloading, coordinate conversions, and GeoTIFF generation.

### georef.py
-   Performs the actual georeferencing:
    -   Uses keypoints and image matching (render and orthophoto) with pycolmap.
    -   Computes the affine transformation matrix between the images.
    -   Saves the transformation matrix to be applied to the 3D model.

### transformations.py
-   Utility library for manipulating 3D transformation matrices:
    -   Rotations, translations, scaling, homographies, quaternions, decomposition and composition of matrices.
    -   Used to compute and apply geometric transformations to models.

## Typical pipeline flow
1.  **Input**: 3D model.
2.  **Synthetic view generation**:  synthetic_imgs.py  Through Blender.
3.  **Geolocation estimation**:  geoloc_img.py  on synthic images.
4.  **Satellite image download**:  ortho.py.
5.  **Image matching and transformation computation**: Deep Image Matching + georef.py.
6.  **Transformation application and alignment**:  main.py  e  transformations.py.
7.  **Output**: georeferenced 3D model.