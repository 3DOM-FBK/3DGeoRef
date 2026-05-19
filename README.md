# 3DGeoRef

**3DGeoRef** is an automated pipeline for georeferencing 3D models using synthetic rendering, AI-powered geolocation, and satellite imagery. The system transforms an arbitrary 3D model into a georeferenced asset precisely aligned with real-world geographic coordinates, ready for integration into GIS systems and 3D web viewers.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Docker Setup (Recommended)](#docker-setup-recommended)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Docker Usage Examples](#docker-usage-examples)
  - [Command-Line Arguments](#command-line-arguments)
- [Pipeline Workflow](#pipeline-workflow)
- [Module Documentation](#module-documentation)
- [Requirements](#requirements)
- [License](#license)
- [Future Updates](#future-updates)
- [Changelog](#changelog)



---

## Overview

Given a 3D model (GLB or GLTF format), **3DGeoRef** executes the following operations:

1. **Synthetic View Generation**: Renders an orthographic top-down view and multiple street-level perspective images of the model using Blender, and exports a post-processed, axis-aligned scaled model.
2. **Scene Dimension Estimation**: Estimates the real-world width of the scene from the top-down render using the Gemini multimodal model, then applies a metric scale to the 3D mesh so that 1 unit = 1 metre.
3. **Geolocation Estimation**: Estimates geographic coordinates (latitude, longitude) from the street-view renders using an AI model (GeoCLIP, Ollama, or Google Gemini).
4. **Satellite Imagery Download**: Retrieves high-resolution satellite tiles from Mapbox and assembles them into a georeferenced GeoTIFF centered on the estimated location.
5. **Image Matching**: Optionally pre-aligns the scene with DINOv2 feature matching, then runs Deep Image Matching (LoFTR + SuperPoint/SuperGlue) to compute a precise 2-D transformation between the synthetic render and the satellite orthophoto.
6. **Georeferencing and Metadata Export**: Applies the full transformation chain to the scaled mesh, extracts the pivot submesh geographic coordinates and elevation, and writes a `heritage_data.json` metadata file suitable for 3D web viewers.

---

## Features

- **Multi-Model AI Geolocation**: Choose between GeoCLIP (offline), Ollama (`llama3.2-vision`), or Google Gemini for location estimation.
- **Automated Metric Scaling**: Gemini-based scene dimension estimation converts the model to real-world metric units before matching.
- **DINOv2 Pre-alignment**: Optional DINOv2-based cross-correlation pass refines the search area before Deep Image Matching.
- **Robust Feature Matching**: Dual-matcher strategy (LoFTR + SuperPoint/SuperGlue) with multi-rotation and multi-scale image variants for maximum inlier coverage.
- **Fully Containerized**: Docker image with CUDA, Blender 4.4, and Deep Image Matching pre-installed.
- **Structured Metadata Output**: Exports `heritage_data.json` with geographic coordinates, elevation, scale, and rotation ready for UH4D / Cesium integration.
- **Flexible Configuration**: API keys, zoom levels, area size, log verbosity, and geolocation model are all configurable at runtime.

---

## Project Structure

```
3DGeoRef/
├── main.py                             # Command-line entry point
├── Dockerfile                          # Docker image (CUDA 12.1 + Blender 4.4 + DIM)
├── docker-compose.yml                  # Compose file for the pipeline + Ollama services
├── pipeline/
│   ├── __init__.py
│   ├── core.py                         # PipelineProcessor — main orchestration class
│   ├── geolocation/
│   │   ├── __init__.py
│   │   ├── geoclip.py                  # GeoCLIP batch predictor
│   │   ├── ollama.py                   # Ollama (llama3.2-vision) geolocator
│   │   └── gemini.py                   # Google Gemini geolocator + dimension estimator
│   ├── georeferencing/
│   │   ├── __init__.py
│   │   ├── dim.py                      # Deep Image Matching integration (georef_dim)
│   │   ├── dino.py                     # DINOv2 image matcher and orthophoto cropper
│   │   └── transformer.py              # GeoTransformer, MatrixUtils, PivotCalculator
│   ├── rendering/
│   │   ├── __init__.py
│   │   └── multiview.py                # Blender multi-view rendering script
│   ├── services/
│   │   ├── __init__.py
│   │   └── satellite_downloader.py     # Mapbox tile downloader → GeoTIFF assembler
│   └── utils/
│       ├── __init__.py
│       ├── coordinate_transforms.py    # EPSG:3857 ↔ WGS84 and geographic utilities
│       ├── elevation.py                # OpenTopoData elevation service + geoid converter
│       └── transformations.py         # Low-level affine/homography matrix utilities
└── README.md
```

---

## Installation

### Docker Setup (Recommended)

Docker is the recommended way to run **3DGeoRef**. The pre-built image bundles all system dependencies including Blender, CUDA, and Deep Image Matching.

#### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA GPU with CUDA 12.1+ support (optional but strongly recommended)
- NVIDIA Container Toolkit (for GPU access inside Docker)

#### Build and Run

```bash
# Clone the repository
git clone https://github.com/3DOM-FBK/3DGeoRef.git
cd 3DGeoRef

# Pull the pre-built Docker image
docker pull 3domfbk/3d-georef:v1.0.0-beta.1

# Or build locally
docker build -t 3domfbk/3d-georef:<TAG> .

# Run the container interactively
docker run --rm -it \
  --gpus all \
  -v /path/to/your/data:/data \
  3domfbk/3d-georef:v1.0.0-beta.1 bash
```

---

## Usage

### Basic Usage

Run the complete georeferencing pipeline on a 3D model:

```bash
python main.py \
  -i /path/to/model.glb \
  -o /path/to/output \
  --geoloc_model gemini \
  --gemini_api_key "YOUR_GEMINI_API_KEY" \
  --mapbox_api_key "YOUR_MAPBOX_API_KEY"
```

### Docker Usage Examples

#### Example 1: Full Pipeline with Gemini Geolocation

```bash
docker run --rm -it \
  --gpus all \
  -v /path/to/data:/data \
  3domfbk/3d-georef:v1.0.0-beta.1 \
    -i /data/input/model.glb \
    -o /data/output \
    --geoloc_model gemini \
    --gemini_model gemini-2.5-flash \
    --gemini_api_key "YOUR_GEMINI_API_KEY" \
    --mapbox_api_key "YOUR_MAPBOX_API_KEY" \
    --streetviews 8 \
    --area_size 500 \
    --zoom 18 \
    --cleanup
```

#### Example 2: GeoCLIP Geolocation (No API Key Required)

```bash
docker run --rm -it \
  --gpus all \
  -v /path/to/data:/data \
  3domfbk/3d-georef:v1.0.0-beta.1 \
    -i /data/input/building.glb \
    -o /data/output \
    --geoloc_model geoclip \
    --nr_prediction 3 \
    --mapbox_api_key "YOUR_MAPBOX_API_KEY" \
    --cleanup
```

#### Example 3: DINOv2 Pre-alignment Enabled

Enable the DINOv2 pre-alignment step to crop the orthophoto around the detected model center before running Deep Image Matching:

```bash
docker run --rm -it \
  --gpus all \
  -v /path/to/data:/data \
  3domfbk/3d-georef:v1.0.0-beta.1 \
    -i /data/input/monument.glb \
    -o /data/output \
    --geoloc_model gemini \
    --gemini_api_key "YOUR_GEMINI_API_KEY" \
    --mapbox_api_key "YOUR_MAPBOX_API_KEY" \
    --use_dino \
    --zoom 20
```

#### Example 4: Ollama Geolocation via Docker Compose

For Ollama-based geolocation, use Docker Compose to orchestrate both the pipeline container and the Ollama model server:

```bash
# Start both services (Ollama model download happens automatically on first run)
docker-compose up -d

# Monitor the Ollama service
docker-compose logs -f ollama

# Run the pipeline (in a separate terminal)
docker exec -it 3dgeoref_python python main.py \
  -i /data/input/model.glb \
  -o /data/output \
  --geoloc_model ollama \
  --mapbox_api_key "YOUR_MAPBOX_API_KEY"
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-i, --input_file` | str | *required* | Path to input 3D model (`.glb` / `.gltf`) |
| `-o, --output_folder` | str | *required* | Directory where output files are written |
| `--streetviews` | int | `5` | Number of street-view perspective renderings around the model |
| `--nr_prediction` | int | `1` | Number of top GPS candidates to retrieve per image (GeoCLIP only) |
| `--area_size` | int | `500` | Side length of the satellite download area in metres |
| `--zoom` | int | `18` | Mapbox tile zoom level (18–20 recommended for urban scenes) |
| `--lat` | float | `None` | Known latitude — reserved for future use |
| `--lon` | float | `None` | Known longitude — reserved for future use |
| `--ortho` | str | `None` | Path to a pre-existing orthophoto (skips satellite download) |
| `--use_dino` | flag | `False` | Enable DINOv2 pre-alignment and orthophoto cropping before DIM |
| `--geoloc_model` | str | `gemini` | Geolocation backend: `geoclip`, `ollama`, or `gemini` |
| `--gemini_model` | str | `gemini-3.1-flash-lite-preview` | Gemini model name (see [Gemini API docs](https://ai.google.dev/gemini-api/docs/models)) |
| `--gemini_api_key` | str | `None` | Google Gemini API key (or set `GEMINI_API_KEY` env var) |
| `--mapbox_api_key` | str | `None` | Mapbox API key (or set `MAPBOX_API_KEY` env var) |
| `--cleanup` | flag | `False` | Remove the temporary working directory in `/tmp` after execution |
| `--loglevel` | str | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, or `ERROR` |

---

## Pipeline Workflow

The pipeline is orchestrated by `PipelineProcessor` in `pipeline/core.py` and proceeds through the following stages:

### Step 1 — Synthetic View Generation (`pipeline/rendering/multiview.py`)

Blender is launched in headless mode to:
- Import the 3D model (GLB/GLTF).
- Compute the mesh bounding box and set up an orthographic top-down camera (`top_view.png`).
- Place `N` street-level perspective cameras uniformly around the model perimeter and render each view.
- Apply HDRI studio lighting for realistic appearance.
- Translate and scale the mesh so that its bottom-left corner aligns with the origin and 1 Blender unit = 1 pixel of `top_view.png`.
- Export the post-processed mesh as `*_scaled.glb`.
- Emit the 4×4 transformation matrix (`MATRIX_BLENDER`) and the pivot position (`PIVOT_BLENDER`) to stdout for downstream consumption.

### Step 2 — Metric Scaling

**2a. Scene dimension estimation** (`pipeline/geolocation/gemini.py`): The `GeminiDimensionEstimator` queries the Gemini API with `top_view.png` and returns the estimated real-world width of the scene in metres.

**2b. Model scaling** (`pipeline/core.py`): A uniform scale factor `metric_scale = dimension_m / image_width_px` is applied to `*_scaled.glb` via `trimesh` so that 1 unit = 1 metre. `top_view.png` is simultaneously resampled to `dimension_m × (proportional height)` pixels so that pixel coordinates directly correspond to metric distances.

### Step 3 — Geolocation Estimation (`pipeline/geolocation/`)

Street-view renders (excluding the nadir `top_view.png`) are forwarded to the selected geolocation backend:

- **GeoCLIP** (`geoclip.py`): Queries the GeoCLIP model with each image and selects the most frequently predicted GPS cluster.
- **Ollama** (`ollama.py`): Sends each image to a locally hosted `llama3.2-vision` model via the Ollama HTTP API; place-name responses are resolved to coordinates through the Nominatim geocoder.
- **Gemini** (`gemini.py`): Sends each image to the Gemini multimodal API with a precise geolocation prompt; the most consistent `(lat, lon)` pair across all images is returned.

### Step 4 — Satellite Tile Download (`pipeline/services/satellite_downloader.py`)

`SatelliteTileDownloader` queries the Mapbox Raster Tiles API to:
- Compute the XYZ tile extent that covers a square area of `area_size` metres centered on the estimated coordinates.
- Download all required tiles at the requested zoom level.
- Assemble them into a single GeoTIFF (`<basename>.tif`) with a correct Web Mercator (EPSG:3857) affine transform.

### Step 5 — Image Matching and Georeferencing Matrix (`pipeline/georeferencing/`)

**5a. DINOv2 pre-alignment** (`dino.py`, optional — `--use_dino`): `DinoImageMatcher` extracts patch-level features from the satellite orthophoto and the top-down render using `DINOv2 ViT-S/14`, then performs cross-correlation across a grid of scales and rotations to estimate the model center within the ortho.

**5b. Orthophoto crop** (`dino.py`): If a valid center is found, `OrthoCropper` crops a sub-region of the GeoTIFF around the predicted location (2× the render dimensions), reducing the search space for subsequent matching.

**5c. Image variant preparation** (`core.py`): 90°/180°/270° rotated copies of the top-down render and 25%/50%/75% downsampled copies of the orthophoto are generated to improve DIM robustness.

**5d. Deep Image Matching** (`dim.py`): Two independent DIM runs are executed — one with LoFTR and one with SuperPoint+SuperGlue. The resulting COLMAP databases are merged via `join_databases.py`. `georef_dim` then queries the merged database to select the image pair with the highest inlier count (accounting for all rotation/scale variants), extracts matched keypoints, and solves for the affine transformation matrix that maps render pixel coordinates to ortho georeferenced coordinates, saving the result to `transformation.txt`.

### Step 6 — Transformation Application and Metadata Export

**6a. GeoTransformer** (`transformer.py`): The full transform chain is:
```
M_total = R_x(−90°) @ (P @ M_dim @ P⁻¹) @ S(metric_scale) @ M_blender
```
where `P` is the axis-permutation matrix mapping DIM conventions to trimesh conventions. This matrix is applied to `*_scaled.glb`, which is overwritten in place.

**6b. Pivot extraction** (`core.py`): The center of the submesh named `pivot` is extracted from the transformed GLB in world space. Its X/Y coordinates (EPSG:3857) are converted to WGS84 latitude/longitude, and the elevation is fetched from the OpenTopoData SRTM30m API (`pipeline/utils/elevation.py`).

**6c. Metadata export** (`core.py`): The total transformation matrix is decomposed into Translation/Rotation/Scale (TRS) components. Scale is exported as `[sx, avg(sx,sz), sz]` and rotation is mapped to the UH4D/Y-up convention (YXZ Euler order). The output file `<output_folder>/<basename>/heritage_data.json` contains:

```json
{
  "latitude":    <pivot latitude>,
  "longitude":   <pivot longitude>,
  "height":      <elevation in metres (orthometric)>,
  "place":       "",
  "description": "",
  "scale":       [sx, sy, sz],
  "rotation":    [rx, ry, rz],
  "translation": [0.0, 0.0, 0.0]
}
```

---

## Module Documentation

### `pipeline/core.py` — `PipelineProcessor`

Main orchestration class. Manages the working directory under `/tmp/<basename>/`, sequences all pipeline stages, and handles inter-stage data flow (Blender matrix, metric scale, coordinates, transformer instance).

### `pipeline/geolocation/`

| Module | Class | Description |
|--------|-------|-------------|
| `geoclip.py` | `GeoClipBatchPredictor` | Offline geolocation using CLIP image embeddings; processes all street-view images and returns the most common GPS prediction. |
| `ollama.py` | `ImageToCoordinates` | Vision-language geolocation via a locally hosted Ollama model; uses `NominatimGeocoder` for place-name → coordinates resolution. |
| `gemini.py` | `GeminiGeolocator` | Google Gemini API geolocation; sends each image with a structured JSON prompt and aggregates results. |
| `gemini.py` | `GeminiDimensionEstimator` | Queries Gemini with the nadir top-view render to estimate the real-world scene width in metres. |

### `pipeline/georeferencing/`

| Module | Class / Function | Description |
|--------|-----------------|-------------|
| `dim.py` | `georef_dim` | Wraps the Deep Image Matching output: loads the merged COLMAP database, selects the best-matching image pair, extracts inlier keypoints, computes the georeferencing affine matrix, and writes `transformation.txt`. |
| `dino.py` | `DinoImageMatcher` | DINOv2 (or DINOv3) feature-based cross-correlation matcher; returns the predicted (row, col) center of the render within the orthophoto. |
| `dino.py` | `OrthoCropper` | Crops the GeoTIFF around a given center at a configurable scale factor. |
| `transformer.py` | `GeoTransformer` | Loads `transformation.txt`, converts from DIM to trimesh coordinate conventions, and applies the full transform chain to `*_scaled.glb`. |
| `transformer.py` | `MatrixUtils` | Static helpers: load a 4×4 matrix from a text file, decompose a 4×4 transform into TRS components. |
| `transformer.py` | `PivotCalculator` | Applies the full transform chain to the original input mesh and returns the bounding-box center (pivot X, Y) in the final georeferenced frame. |

### `pipeline/rendering/`

| Module | Description |
|--------|-------------|
| `multiview.py` | Blender Python script executed in headless mode. Imports the model, computes optimal camera placements, renders top-down and street-level views with HDRI lighting, applies post-processing (translation + scale), exports `*_scaled.glb`, and prints `MATRIX_BLENDER` and `PIVOT_BLENDER` to stdout. |

### `pipeline/services/`

| Module | Class | Description |
|--------|-------|-------------|
| `satellite_downloader.py` | `SatelliteTileDownloader` | Downloads Mapbox satellite tiles for a geographic bounding box and merges them into a GeoTIFF with proper EPSG:3857 affine transform. |

### `pipeline/utils/`

| Module | Class | Description |
|--------|-------|-------------|
| `coordinate_transforms.py` | `CoordinateTransforms` | Static methods for EPSG:3857 ↔ WGS84 conversion, Web Mercator scale factor, meters/pixel calculation, and spherical displacement. |
| `elevation.py` | `ElevationService` | Fetches orthometric elevation for a lat/lon pair from the OpenTopoData REST API (SRTM30m dataset). |
| `elevation.py` | `GeoidConverter` | Utility for converting orthometric heights (MSL) to ellipsoidal heights for use in Cesium/WGS84. |
| `transformations.py` | `affine_matrix_from_points` | Computes a 2-D affine transformation matrix from matched point correspondences (used by `georef_dim`). |

---

## Requirements

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 22.04 / Windows WSL2 | Ubuntu 22.04 LTS |
| RAM | 16 GB | 32 GB |
| GPU | — | NVIDIA GPU, 8+ GB VRAM |
| Disk | 10 GB (Docker image) | 20 GB+ |

### Software Dependencies

- **Blender** 4.4.0 (headless mode)
- **Python** 3.10+
- **CUDA** 12.1+ (optional, for GPU acceleration)
- **Deep Image Matching** (`dev` branch) — [3DOM-FBK/deep-image-matching](https://github.com/3DOM-FBK/deep-image-matching)

### Python Packages

Key dependencies (see `Dockerfile` for the complete list):

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | DINOv2 inference and GeoCLIP |
| `geoclip` | GeoCLIP geolocation model |
| `pycolmap` | COLMAP database interface for DIM |
| `trimesh` | 3D mesh loading and transformation |
| `rasterio` | GeoTIFF read/write |
| `pyproj` | Coordinate reference system transformations |
| `google-genai` | Google Gemini API client |
| `ollama` | Ollama API client |
| `requests` | HTTP requests (Mapbox, OpenTopoData, Nominatim) |
| `Pillow` | Image processing and I/O |
| `numpy` | Numerical computation |

---

## Future Updates

- **Improved Elevation Integration**: Refine elevation handling to correctly account for the difference between orthometric (MSL) and ellipsoidal heights when positioning models in Cesium.
- **Additional 3D Format Support**: Extend the input pipeline to support point clouds and other common 3D formats beyond GLB/GLTF.

---

## Changelog

### 2026-01-16

- **Dual-Matcher DIM**: Integrated LoFTR and SuperPoint+SuperGlue in parallel within the Deep Image Matching step for more robust feature correspondence across diverse scene types.
- **Gemini Dimension Estimation**: Implemented automatic estimation of real-world scene width from the nadir render using the Gemini multimodal model, enabling proper metric scaling prior to image matching.

---

## Acknowledgments

- **Deep Image Matching**: [3DOM-FBK/deep-image-matching](https://github.com/3DOM-FBK/deep-image-matching)
- **GeoCLIP**: Geolocation estimation using CLIP embeddings
- **DINOv2**: Self-supervised vision features — Meta AI Research
- **Blender**: Open-source 3D creation suite
- **Google Gemini**: Multimodal AI for geolocation and dimension estimation
- **Ollama**: Local large language model inference
- **OpenTopoData**: Open elevation API

---

## Contact

For questions, issues, or contributions, please open an issue on the GitHub repository or contact:

**3DOM-FBK**  
Fondazione Bruno Kessler  
Via Sommarive 18, 38123 Trento, Italy  
[https://3dom.fbk.eu](https://3dom.fbk.eu)
