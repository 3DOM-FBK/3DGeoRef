import requests
from urllib.parse import urlencode
import numpy as np
import os
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.transform import from_origin
import math
import argparse
import platform
import shutil
from pyproj import Transformer
import time


# ===== Function: parse_args =====
def parse_args():
    parser = argparse.ArgumentParser(description="Download satellite tiles from Google Maps and generate a georeferenced GeoTIFF.")

    parser.add_argument("--api_key", required=True, help="Google Cloud Platform API key.")
    parser.add_argument("--center_lat", type=float, required=True, help="Latitude of the center of the area.")
    parser.add_argument("--center_lon", type=float, required=True, help="Longitude of the center of the area.")
    parser.add_argument("--area_size_m", type=float, required=True, help="Side length of the square area to download (in meters).")
    parser.add_argument("--zoom", type=int, required=True, help="Zoom level (e.g., 18 or 20).")
    parser.add_argument("-o", "--output_folder", required=True, help="Folder to save the downloaded tiles and the final GeoTIFF.")

    return parser.parse_args()


# ===== Function: create_google_maps_session =====
def create_google_maps_session(api_key, map_type="satellite", max_retries=5, retry_delay=5):
    """
    Creates a Google Maps tile session and retrieves a session token for accessing
    tile resources with the Maps Tile API, with automatic retry on 403 errors.

    Args:
        api_key (str): Your Google Cloud Platform API key.
        map_type (str, optional): Type of the map tiles (e.g., 'satellite', 'roadmap').
                                  Defaults to 'satellite'.
        max_retries (int, optional): Max number of retries for 403 errors. Defaults to 5.
        retry_delay (int, optional): Seconds to wait between retries. Defaults to 5.

    Returns:
        str: A session token string used to authorize tile requests.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request fails after retries.
    """
    url = f"https://tile.googleapis.com/v1/createSession?key={api_key}"
    payload = {
        "mapType": map_type,
        "language": "en-US",
        "region": "IT"
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # genera eccezione per errori HTTP
            data = response.json()
            session_token = data.get("session")
            if not session_token:
                raise ValueError("No session token returned in response")
            print("Session token:", session_token)
            return session_token

        except requests.exceptions.HTTPError as e:
            if response.status_code == 403 and attempt < max_retries:
                print(f"403 Forbidden received, retrying in {retry_delay}s... (attempt {attempt}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print("HTTP error:", e)
                raise
        except Exception as e:
            print("Error creating session:", e)
            raise


# ===== Function: download_image =====
def download_image(url, filename):
    """
    Downloads an image from the specified URL and saves it to the given filename.

    Args:
        url (str): The URL of the image to download.
        filename (str): The local file path where the image will be saved.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.

    Description:
        The function streams the content from the URL to avoid loading the entire
        file into memory at once. It writes the data in chunks of 8192 bytes to the
        specified local file. On success, it prints a confirmation message.
        If any HTTP or connection error occurs, it catches the exception and prints an error.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")


# ===== Function: lat_lng_to_tile =====
def lat_lng_to_tile(lat, lng, zoom):
    """
    Converts geographic latitude and longitude into Google Maps tile coordinates
    (x, y) at a specified zoom level.

    Args:
        lat (float): Latitude in decimal degrees.
        lng (float): Longitude in decimal degrees.
        zoom (int): Zoom level (typically 0 to 21).

    Returns:
        tuple: (x, y) tile coordinates as integers.

    Description:
        This function projects the given latitude and longitude into the Web Mercator
        tile coordinate system used by Google Maps and similar tile providers.
        The output tile coordinates correspond to the tile indices at the given zoom level.
    """
    n = 1 << zoom
    x = int((lng + 180.0) / 360.0 * n)
    lat_rad = np.radians(lat)
    y = int((1.0 - np.log(np.tan(lat_rad) + 1 / np.cos(lat_rad)) / np.pi) / 2.0 * n)
    return x, y


# ===== Function: tile_to_lat_lng =====
def tile_to_lat_lng(x, y, zoom):
    """
    Converts Google Maps tile coordinates (x, y) at a given zoom level
    into geographic latitude and longitude in decimal degrees.

    Args:
        x (int): Tile coordinate in the horizontal direction.
        y (int): Tile coordinate in the vertical direction.
        zoom (int): Zoom level (typically 0 to 21).

    Returns:
        tuple: (latitude, longitude) in decimal degrees.

    Description:
        The function converts tile XY coordinates from the Web Mercator 
        tile numbering scheme into geographic coordinates (latitude, longitude).
        Latitude is computed by inverting the Mercator projection.
    """
    n = 1 << zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
    lat = np.degrees(lat_rad)
    return lat, lon


# ===== Function: move_point_by_meters =====
def move_point_by_meters(lat, lng, bearing, distance_m):
    """
    Calculates the new geographic coordinates by moving from an initial point
    a certain distance along a specified bearing.

    Args:
        lat (float): Initial latitude in decimal degrees.
        lng (float): Initial longitude in decimal degrees.
        bearing (float): Direction of movement in degrees (0 = North, 90 = East, etc.).
        distance_m (float): Distance to move in meters.

    Returns:
        tuple: (new_latitude, new_longitude) in decimal degrees.

    Description:
        The function uses the great-circle distance formula on a spherical Earth
        (approximated as a sphere with mean radius R = 6378137 m) to calculate the
        new position after moving `distance_m` meters from (lat, lng) along the
        direction `bearing`.
    """
    R = 6378137  # Earth's mean radius in meters

    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)
    bearing_rad = np.radians(bearing)

    lat_new_rad = np.arcsin(np.sin(lat_rad) * np.cos(distance_m / R) +
                            np.cos(lat_rad) * np.sin(distance_m / R) * np.cos(bearing_rad))

    lng_new_rad = lng_rad + np.arctan2(np.sin(bearing_rad) * np.sin(distance_m / R) * np.cos(lat_rad),
                                        np.cos(distance_m / R) - np.sin(lat_rad) * np.sin(lat_new_rad))

    return np.degrees(lat_new_rad), np.degrees(lng_new_rad)


# ===== Function: download_area_tiles =====
def download_area_tiles(center_lat, center_lng, area_side_meters, zoom_level, api_key, output_folder="map_tiles", map_type="satellite"):
    """
    Download individual tiles (256x256) for a specified area using the Google Maps Tiles API.

    Args:
        center_lat (float): Latitude of the centre of the area.
        center_lng (float): Longitude of the centre of the area.
        area_side_meters (float): Length of the side of the square area in metres.
        zoom_level (int): Zoom level.
        api_key (str): Your Google Cloud Platform API key.
        output_folder (str): Folder where you save your tiles.
        map_type (str): Map type (e.g., 'satellite', 'roadmap', 'terrain', 'hybrid').
    """
    half_side_m = area_side_meters / 2

    north_lat, _ = move_point_by_meters(center_lat, center_lng, 0, half_side_m)
    south_lat, _ = move_point_by_meters(center_lat, center_lng, 180, half_side_m)
    _, east_lng = move_point_by_meters(center_lat, center_lng, 90, half_side_m)
    _, west_lng = move_point_by_meters(center_lat, center_lng, 270, half_side_m)

    min_lat = south_lat
    max_lat = north_lat
    min_lng = west_lng
    max_lng = east_lng

    min_tile_x, max_tile_y = lat_lng_to_tile(max_lat, min_lng, zoom_level)
    max_tile_x, min_tile_y = lat_lng_to_tile(min_lat, max_lng, zoom_level)

    min_x = min(min_tile_x, max_tile_x)
    max_x = max(min_tile_x, max_tile_x)
    min_y = min(min_tile_y, max_tile_y)
    max_y = max(min_tile_y, max_tile_y)

    os.makedirs(output_folder, exist_ok=True)

    downloaded_count = 0
    total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)

    # Use Google Maps Tiles API (commented out due to potential 403 errors)
    # for x in range(min_x, max_x + 1):
    #     for y in range(min_y, max_y + 1):
    #         session_token = create_google_maps_session(api_key)
    #         tile_url = f"https://tile.googleapis.com/v1/2dtiles/{zoom_level}/{x}/{y}?session={session_token}&key={api_key}"
    #         filename = os.path.join(output_folder, f"tile_z{zoom_level}_x{x}_y{y}.png")
    #         download_image(tile_url, filename)
    #         downloaded_count += 1

    # Use Mapbox as alternative source
    for x in range(min_x, max_x+1):
        for y in range(min_y, max_y+1):
            url = f"https://api.mapbox.com/v4/mapbox.satellite/{zoom_level}/{x}/{y}@1x.png?access_token={api_key}"
            filename = os.path.join(output_folder, f"tile_z{zoom_level}_x{x}_y{y}.png")
            r = requests.get(url)
            if r.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(r.content)
            else:
                print(f"Failed to download tile {x},{y}")
    
    return min_x, min_y


def meters_per_pixel(zoom, latitude):
    """
    Returns the ground resolution in meters/pixel at the given latitude and zoom level
    for Web Mercator projection (EPSG:3857).

    Args:
        zoom (int): Zoom level.
        latitude (float): Latitude in decimal degrees.

    Returns:
        float: Ground resolution in meters per pixel.
    """
    initial_resolution = 156543.03392804097  # meters/pixel at equator, zoom 0
    return initial_resolution * math.cos(math.radians(latitude)) / (2 ** zoom)


# ===== Function: merge_tiles_to_geotiff =====
def merge_tiles_to_geotiff(tile_folder, tile_size_px, zoom_level, output_filename, min_x, min_y):
    """
    Merges PNG tiles and creates a georeferenced GeoTIFF using centre and resolution.

    Args:
        tile_folder (str): Folder of the PNG tiles.
        tile_size_px (int): Size in pixels of each tile (e.g. 256).
        zoom_level (int): Zoom level (not used here, only for file naming).
        output_filename (str): GeoTIFF file path.
        min_x (int): Minimum X tile index (leftmost).
        min_y (int): Minimum Y tile index (topmost).
    """
    tiles = []
    for filename in os.listdir(tile_folder):
        if filename.endswith(".png") and f"z{zoom_level}_" in filename:
            parts = filename.replace(".png", "").split("_")
            x = int(parts[2][1:])
            y = int(parts[3][1:])
            tiles.append((x, y, filename))

    if not tiles:
        print("No tiles found.")
        return

    xs = sorted(set(t[0] for t in tiles))
    ys = sorted(set(t[1] for t in tiles))
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    tiles_x = max_x - min_x + 1
    tiles_y = max_y - min_y + 1

    width_px = tiles_x * tile_size_px
    height_px = tiles_y * tile_size_px

    mosaic = Image.new('RGB', (width_px, height_px))

    for x, y, filename in tiles:
        img = Image.open(os.path.join(tile_folder, filename))
        offset_x = (x - min_x) * tile_size_px
        offset_y = (y - min_y) * tile_size_px
        mosaic.paste(img, (offset_x, offset_y))

    tile_lat, tile_lon = tile_to_lat_lng(min_x, min_y, zoom_level)

    # Transform in EPSG:3857
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    top_left_lon, top_left_lat = transformer.transform(tile_lon, tile_lat)

    pixel_size  = meters_per_pixel(zoom_level, 0.0)
    transform = from_origin(top_left_lon, top_left_lat, pixel_size , pixel_size)

    r, g, b = mosaic.split()
    with rasterio.open(
        output_filename,
        'w',
        driver='GTiff',
        height=height_px,
        width=width_px,
        count=3,
        dtype='uint8',
        crs='EPSG:3857',
        transform=transform
    ) as dst:
        dst.write(np.array(r), 1)
        dst.write(np.array(g), 2)
        dst.write(np.array(b), 3)


def latlon_to_webmercator(lat, lon):
    """
    Converts geographic coordinates from WGS84 (EPSG:4326) to Web Mercator (EPSG:3857).

    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.

    Returns:
        tuple: (x, y) coordinates in meters in the Web Mercator projection.
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


# ===== Function: main =====
if __name__ == "__main__":
    args = parse_args()

    API_KEY = args.api_key
    CENTER_LATITUDE = args.center_lat
    CENTER_LONGITUDE = args.center_lon
    AREA_SIDE_METERS = args.area_size_m
    DOWNLOAD_ZOOM_LEVEL = args.zoom

    tile_tmp_dir = os.path.join(args.output_folder, "tile")
    min_x, min_y = download_area_tiles(CENTER_LATITUDE, CENTER_LONGITUDE, AREA_SIDE_METERS, DOWNLOAD_ZOOM_LEVEL, API_KEY, tile_tmp_dir)

    base_name = os.path.basename(os.path.normpath(args.output_folder))
    out_path = os.path.join(args.output_folder, f"{base_name}.tif")

    merge_tiles_to_geotiff(
        tile_folder=tile_tmp_dir,
        tile_size_px=256,
        zoom_level=DOWNLOAD_ZOOM_LEVEL,
        output_filename=out_path,
        min_x=min_x,
        min_y=min_y
    )