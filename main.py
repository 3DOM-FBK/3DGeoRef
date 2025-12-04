import argparse
from pipeline.core import PipelineProcessor



# ===== Function: parse_args =====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True, help="Path to input .glb/.gltf file")
    parser.add_argument("-o", "--output_folder", required=True, help="Folder to save outputs")
    parser.add_argument("--streetviews", type=str, default="5", help="Number of streetview-style renderings around the model (default: 5)")
    parser.add_argument("--nr_prediction", type=str, default="1", help="Number of gps prediction (default: 1) - GeoClip only")
    parser.add_argument("--area_size_m", type=str, default="500", help="Side length of the square area to download (in meters).")
    parser.add_argument("--zoom", type=str, default="18", help="Zoom level (e.g., 18 or 20).")
    parser.add_argument("--lat", type=str, default=None, help="Latitude of 3d model")
    parser.add_argument("--lon", type=str, default=None, help="Longitude of 3d mode")
    parser.add_argument("--ortho", type=str, default=None, help="Orthophoto image to use for georeferencing")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "geoloc", "dim"],
        default="auto",
        help="Pipeline execution mode: "
            "'auto' = full pipeline, "
            "'geoloc' = only Geolocalize step, "
            "'dim' = only Deep Image Matching (requires provided lat/lon)"
    )

    parser.add_argument(
        "--geoloc_model",
        type=str,
        choices=["geoclip", "ollama", "gemini"],
        default="gemini",
        help="Model to use for geolocation: "
            "'geoclip' = use GeoCLIP model, "
            "'ollama' = use Ollama AI model, "
            "'gemini' = use Gemini AI model (default)"
    )
    
    return parser.parse_args()


# ===== Function: main =====
if __name__ == "__main__":
    args = parse_args()

    processor = PipelineProcessor(args)
    processor.run_pipeline()