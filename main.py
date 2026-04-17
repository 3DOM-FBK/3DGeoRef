import argparse
from pipeline.core import PipelineProcessor



# ===== Function: parse_args =====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True, help="Path to input .glb/.gltf file")
    parser.add_argument("-o", "--output_folder", required=True, help="Folder to save outputs")
    parser.add_argument("--streetviews", type=str, default="5", help="Number of streetview-style renderings around the model (default: 5)")
    parser.add_argument("--nr_prediction", type=str, default="1", help="Number of gps prediction (default: 1) - GeoClip only")
    parser.add_argument("--area_size", type=str, default="500", help="Side length of the square area to download.")
    parser.add_argument("--zoom", type=str, default="18", help="Zoom level (e.g., 18 or 20).")
    parser.add_argument("--lat", type=str, default=None, help="Latitude of 3d model")
    parser.add_argument("--lon", type=str, default=None, help="Longitude of 3d mode")
    parser.add_argument("--ortho", type=str, default=None, help="Orthophoto image to use for georeferencing")
    parser.add_argument(
        "--use_dino",
        action="store_true",
        default=False,
        help="Enable DINO-based image alignment before Deep Image Matching (default: disabled)"
    )

    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     choices=["auto", "geoloc", "dim"],
    #     default="auto",
    #     help="Pipeline execution mode: "
    #         "'auto' = full pipeline, "
    #         "'geoloc' = only Geolocalize step, "
    #         "'dim' = only Deep Image Matching (requires provided lat/lon)"
    # )

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

    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-3.1-flash-lite-preview",
        help="Gemini model version to use (e.g., 'gemini-3-flash', 'gemini-2.5-flash' ...)."
    )

    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="API Key for Google Gemini (can also be set via GEMINI_API_KEY environment variable)."
    )

    parser.add_argument(
        "--mapbox_api_key",
        type=str,
        default=None,
        help="API Key for Mapbox (can also be set via MAPBOX_API_KEY environment variable)."
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Delete temporary working directory in /tmp after execution (default: keep)."
    )
    
    return parser.parse_args()


# ===== Function: main =====
if __name__ == "__main__":
    args = parse_args()

    processor = PipelineProcessor(args)
    processor.run_pipeline()
