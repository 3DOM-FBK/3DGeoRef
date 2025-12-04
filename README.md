# 3DGeoRef

**3DGeoRef** is an automated pipeline for georeferencing 3D models using synthetic rendering, AI-powered geolocation, and satellite imagery. The system transforms an arbitrary 3D model into a georeferenced asset aligned with real-world geographic coordinates, ready for integration into GIS systems or 3D viewers.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Docker Setup (Recommended)](#docker-setup-recommended)
  - [Local Installation](#local-installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Docker Usage Examples](#docker-usage-examples)
  - [Command-Line Arguments](#command-line-arguments)
  - [Execution Modes](#execution-modes)
- [Pipeline Workflow](#pipeline-workflow)
- [Module Documentation](#module-documentation)
- [Requirements](#requirements)
- [License](#license)

---

## Overview

Given a 3D model (GLB or GLTF format), **3DGeoRef** performs the following operations:

1. **Synthetic View Generation**: Creates multiple rendered views of the model using Blender
2. **Geolocation Estimation**: Estimates geographic coordinates using AI models (GeoCLIP, Ollama, or Gemini)
3. **Satellite Image Download**: Retrieves high-resolution satellite imagery from Mapbox API
4. **Image Matching**: Performs feature matching between synthetic renders and satellite images using Deep Image Matching
5. **Transformation Computation**: Calculates the affine transformation matrix to align the model
6. **Georeferencing**: Applies the transformation and elevation alignment to produce the final georeferenced model

---

## Features

- **Multi-Model AI Geolocation**: Choose between GeoCLIP, Ollama (llama3.2-vision), or Google Gemini for location estimation
- **Automated Pipeline**: End-to-end processing from raw 3D model to georeferenced output
- **Flexible Execution Modes**: Run the full pipeline, geolocation only, or image matching only
- **Docker Support**: Fully containerized environment with GPU support for CUDA acceleration
- **High-Quality Rendering**: Blender-based synthetic view generation with HDRI lighting
- **Robust Image Matching**: Integration with Deep Image Matching for accurate feature correspondence
- **Comprehensive Transformation Library**: Advanced 3D transformation utilities for precise alignment

---

## Project Structure

```
3DGeoRef/
├── main.py                             # Main entry point for the pipeline
├── Dockerfile                          # Docker image for the main pipeline (with Blender & CUDA)
├── docker-compose.yml                  # Docker Compose configuration
├── hdri/                               # HDRI environment maps for rendering
├── pipeline/                           # Core pipeline modules
│   ├── __init__.py
│   ├── core.py                         # PipelineProcessor - main orchestration logic
│   ├── geolocation/                    # Geolocation estimation modules
│   │   ├── __init__.py
│   │   ├── geoclip.py                  # GeoCLIP-based geolocation
│   │   ├── ollama.py                   # Ollama AI-based geolocation
│   │   └── gemini.py                   # Google Gemini-based geolocation
│   ├── georeferencing/                 # Georeferencing and transformation modules
│   │   ├── __init__.py
│   │   ├── dim.py                      # Deep Image Matching integration
│   │   └── transformer.py              # 3D model transformation and alignment
│   ├── rendering/                      # 3D rendering modules
│   │   ├── __init__.py
│   │   └── multiview.py                # Blender-based multi-view synthetic rendering
│   ├── services/                       # External service integrations
│   │   ├── __init__.py
│   │   └── satellite_downloader.py     # Mapbox satellite imagery download
│   └── utils/                          # Utility modules
│       ├── __init__.py
│       └── transformations.py          # 3D transformation matrix utilities
└── README.md                           # This file
```

---

## Installation

### Docker Setup (Recommended)

The easiest way to run **3DGeoRef** is using Docker, which provides a pre-configured environment with all dependencies including Blender, CUDA, and Deep Image Matching.

#### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA GPU with CUDA support (optional but recommended)
- NVIDIA Container Toolkit (for GPU acceleration)

#### Build and Run

```bash
# Clone the repository
git clone https://github.com/3DOM-FBK/3DGeoRef.git
cd 3DGeoRef

# Build the Docker image
docker build -t 3dgeoref:latest .

# Run the container interactively
docker run --rm -it \
  --gpus all \
  -v $(pwd):/app \
  -v /path/to/your/data:/data \
  3dgeoref:latest bash
```

---

## Usage

### Basic Usage

Run the complete georeferencing pipeline on a 3D model:

```bash
python main.py \
  -i /path/to/model.glb \
  -o /path/to/output \
  --geoloc_model gemini
```

### Docker Usage Examples

#### Example 1: Full Pipeline with Automatic Geolocation

Process a 3D model with automatic geolocation using Gemini AI:

```bash
docker run --rm -it \
  --gpus all \
  -v $(pwd):/app \
  -v /path/to/data:/data \
  -e GOOGLE_API_KEY=your_gemini_api_key \
  -e MAPBOX_API_KEY=your_mapbox_api_key \
  3dgeoref:latest \
  python main.py \
    -i /data/input/model.glb \
    -o /data/output \
    --geoloc_model gemini \
    --streetviews 8 \
    --area_size_m 500 \
    --zoom 18
```

#### Example 2: Using GeoCLIP for Geolocation

Use the GeoCLIP model for faster geolocation (no API key required):

```bash
docker run --rm -it \
  --gpus all \
  -v $(pwd):/app \
  -v /path/to/data:/data \
  3dgeoref:latest \
  python main.py \
    -i /data/input/building.obj \
    -o /data/output \
    --geoloc_model geoclip \
    --nr_prediction 3
```

#### Example 3: Manual Coordinates with DIM Mode

Skip geolocation and run only Deep Image Matching with known coordinates:

```bash
docker run --rm -it \
  --gpus all \
  -v $(pwd):/app \
  -v /path/to/data:/data \
  3dgeoref:latest \
  python main.py \
    -i /data/input/monument.ply \
    -o /data/output \
    --mode dim \
    --lat 46.0669 \
    --lon 11.1216 \
    --area_size_m 300 \
    --zoom 20
```

#### Example 4: Using Custom Orthophoto

Use your own orthophoto instead of downloading satellite imagery:

```bash
docker run --rm -it \
  --gpus all \
  -v $(pwd):/app \
  -v /path/to/data:/data \
  3dgeoref:latest \
  python main.py \
    -i /data/input/site.fbx \
    -o /data/output \
    --ortho /data/orthophoto.tif \
    --lat 48.8582 \
    --lon 2.2945
```

#### Example 5: Using Docker Compose with Ollama

For using Ollama-based geolocation, use Docker Compose to run both services:

```bash
# Start the services
docker-compose up -d

# Wait for Ollama to download the model (first run only)
docker-compose logs -f ollama

# Run the pipeline (in a new terminal)
docker exec -it 3dgeoref_python python main.py \
  -i /data/input/model.glb \
  -o /data/output \
  --geoloc_model ollama
```

#### Example 6: Interactive Development Mode

Run the container interactively for development and debugging:

```bash
docker run --rm -it \
  --gpus all \
  -v $(pwd):/app \
  -v /path/to/data:/data \
  --entrypoint bash \
  3dgeoref:latest

# Inside the container, you can run commands manually:
python main.py -i /data/input/test.glb -o /data/output --mode geoloc
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-i, --input_file` | str | *required* | Path to input 3D model (.glb/.gltf) |
| `-o, --output_folder` | str | *required* | Output folder for results |
| `--streetviews` | int | 5 | Number of street-view style renderings around the model |
| `--nr_prediction` | int | 1 | Number of GPS predictions (GeoCLIP only) |
| `--area_size_m` | int | 500 | Side length of square area to download (meters) |
| `--zoom` | int | 18 | Satellite imagery zoom level (18-20 recommended) |
| `--lat` | float | None | Manual latitude (skips geolocation if provided with --lon) |
| `--lon` | float | None | Manual longitude (skips geolocation if provided with --lat) |
| `--ortho` | str | None | Path to custom orthophoto (skips satellite download) |
| `--mode` | str | auto | Execution mode: `auto`, `geoloc`, or `dim` |
| `--geoloc_model` | str | gemini | Geolocation model: `geoclip`, `ollama`, or `gemini` |

### Execution Modes

- **`auto`** (default): Full pipeline from 3D model to georeferenced output
- **`geoloc`**: Only perform geolocation estimation and stop
- **`dim`**: Skip geolocation, perform only Deep Image Matching (requires `--lat` and `--lon`)

---

## Pipeline Workflow

The complete pipeline follows these steps:

### 1. **Synthetic View Generation** (`pipeline/rendering/multiview.py`)
   - Loads the 3D model into Blender
   - Computes bounding box and optimal camera positions
   - Generates orthographic top-down view
   - Creates multiple street-view perspective renderings
   - Applies HDRI lighting for realistic appearance
   - Exports rendered images and scaled model

### 2. **Geolocation Estimation** (`pipeline/geolocation/`)
   - **GeoCLIP** (`geoclip.py`): Fast, offline geolocation using CLIP embeddings
   - **Ollama** (`ollama.py`): Vision-language model (llama3.2-vision) for location reasoning
   - **Gemini** (`gemini.py`): Google's multimodal AI for high-accuracy geolocation
   - Filters out top-down views for better accuracy
   - Returns GPS coordinates (latitude, longitude)

### 3. **Satellite Image Download** (`pipeline/services/satellite_downloader.py`)
   - Queries Mapbox Static API for satellite tiles
   - Downloads tiles at specified zoom level
   - Stitches tiles into georeferenced mosaic
   - Exports as GeoTIFF with proper coordinate system

### 4. **Image Matching** (`pipeline/georeferencing/dim.py`)
   - Integrates with Deep Image Matching (DIM)
   - Extracts and matches keypoints between synthetic renders and satellite imagery
   - Uses pycolmap for robust feature matching
   - Computes homography and affine transformation matrices

### 5. **Model Transformation** (`pipeline/georeferencing/transformer.py`)
   - Applies computed transformation to 3D model
   - Aligns model to correct elevation using DEM data
   - Handles coordinate system conversions (WGS84, UTM, local)
   - Exports georeferenced model in original format

### 6. **Output Generation** (`pipeline/core.py`)
   - Saves georeferenced 3D model
   - Exports transformation matrices
   - Generates debug visualizations
   - Creates processing logs and metadata

---

## Module Documentation

### Core Module (`pipeline/core.py`)

**`PipelineProcessor`**: Main orchestration class that manages the entire pipeline.

- Handles logging and temporary directory management
- Coordinates all pipeline stages
- Manages error handling and recovery
- Provides progress tracking

### Geolocation Modules (`pipeline/geolocation/`)

- **`geoclip.py`**: GeoCLIP-based geolocation using CLIP embeddings
- **`ollama.py`**: Ollama AI integration for vision-language geolocation
- **`gemini.py`**: Google Gemini API integration for multimodal geolocation

### Georeferencing Modules (`pipeline/georeferencing/`)

- **`dim.py`**: Deep Image Matching integration for feature correspondence
- **`transformer.py`**: 3D transformation and georeferencing utilities

### Rendering Module (`pipeline/rendering/`)

- **`multiview.py`**: Blender-based synthetic view generation with HDRI lighting

### Services Module (`pipeline/services/`)

- **`satellite_downloader.py`**: Mapbox satellite imagery download and mosaicking

### Utils Module (`pipeline/utils/`)

- **`transformations.py`**: Comprehensive 3D transformation library
  - Rotation matrices (Euler angles, quaternions, axis-angle)
  - Translation and scaling
  - Homography and affine transformations
  - Matrix decomposition and composition
  - Coordinate system conversions

---

## Requirements

### System Requirements

- **OS**: Linux (Ubuntu 22.04+ recommended), Windows with WSL2
- **RAM**: 16 GB minimum, 32 GB recommended
- **GPU**: NVIDIA GPU with 8+ GB VRAM (optional but highly recommended)
- **Storage**: 10 GB for Docker images, additional space for data

### Software Dependencies

- **Blender** 4.4.0+
- **Python** 3.9+
- **CUDA** 12.1+ (for GPU acceleration)
- **Deep Image Matching** (dev branch)

### Python Packages

See `Dockerfile` for complete list. Key dependencies:
- `torch`, `torchvision` (PyTorch)
- `geoclip` (geolocation)
- `pycolmap` (image matching)
- `trimesh`, `open3d` (3D processing)
- `rasterio` (geospatial data)
- `google-genai` (Gemini API)
- `ollama` (Ollama integration)

---

## License

This project is developed by **3DOM-FBK** (Fondazione Bruno Kessler, 3D Optical Metrology unit).

For licensing information, please contact the authors.

---

## Acknowledgments

- **Deep Image Matching**: [3DOM-FBK/deep-image-matching](https://github.com/3DOM-FBK/deep-image-matching)
- **GeoCLIP**: Geolocation estimation using CLIP
- **Blender**: Open-source 3D creation suite
- **Google Gemini**: Multimodal AI for geolocation
- **Ollama**: Local AI model inference

---

## Contact

For questions, issues, or contributions, please open an issue on the GitHub repository or contact:

**3DOM-FBK**  
Fondazione Bruno Kessler  
Via Sommarive 18, 38123 Trento, Italy  
[https://3dom.fbk.eu](https://3dom.fbk.eu)