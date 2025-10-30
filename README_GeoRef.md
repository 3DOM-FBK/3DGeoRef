# INPUT
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", required=True, help="Path to input .obj/.ply/.fbx/.glb/.gltf file")
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

# GLABAL VARIABLES
$GEMINI_API_KEY = API key for accessing Gemini AI services. You can generate this key from your Gemini AI Studio account.
$MAPBOX_API_KEY = API key for Mapbox, used to download and process map tiles.

# OUTPUT
The output of the pipeline is a 3D model in OBJ format, which has been scaled, rotated, and translated to correctly georeference it in the target coordinate system.