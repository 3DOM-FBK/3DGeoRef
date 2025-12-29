import requests
from urllib.parse import urlencode
import numpy as np
import os
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.transform import from_origin
import math
import platform
import shutil
from pyproj import Transformer
import time
import sys


class satelliteTileDownloader():
    def __init__(self, center_lat, center_lon, area_size_m, zoom, output_folder):
        self.api_key = os.environ["MAPBOX_API_KEY"]
        self.map_type = "satellite"
        self.center_lat = float(center_lat)
        self.center_lon = float(center_lon)
        self.area_size_m = int(area_size_m)
        self.zoom = int(zoom)
        self.output_folder = output_folder


    # ===== Helper Functions: clamp_lat =====
    def clamp_lat(self, lat):
        # Limiti validi per Web Mercator
        return max(min(lat, 85.05112878), -85.05112878)


    # ===== Helper Functions: normalize_lng =====
    def normalize_lng(self, lng):
        while lng < -180:
            lng += 360
        while lng > 180:
            lng -= 360
        return lng


    # ===== Function: lat_lng_to_tile =====
    def lat_lng_to_tile(self, lat, lng, zoom):
        lat = self.clamp_lat(lat)
        lng = self.normalize_lng(lng)

        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x_tile = int((lng + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)

        x_tile = max(0, min(int(n) - 1, x_tile))
        y_tile = max(0, min(int(n) - 1, y_tile))
        return x_tile, y_tile


    # ===== Function: tile_to_lat_lng =====
    def tile_to_lat_lng(self, x, y, zoom):
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
    def move_point_by_meters(self, lat, lng, bearing, distance_m):
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
    def download_area_tiles(self, center_lat, center_lng, area_side_meters, zoom_level, output_folder="map_tiles", map_type="satellite"):
        """
        Download individual tiles (256x256) for a specified area using the MapBox Raster API.

        Args:
            center_lat (float): Latitude of the centre of the area.
            center_lng (float): Longitude of the centre of the area.
            area_side_meters (float): Length of the side of the square area in metres.
            zoom_level (int): Zoom level.
            output_folder (str): Folder where you save your tiles.
            map_type (str): Map type (e.g., 'satellite', 'roadmap', 'terrain', 'hybrid').
        """
        half_side_m = float(area_side_meters) / 2

        north_lat, _ = self.move_point_by_meters(center_lat, center_lng, 0, half_side_m)
        south_lat, _ = self.move_point_by_meters(center_lat, center_lng, 180, half_side_m)
        _, east_lng = self.move_point_by_meters(center_lat, center_lng, 90, half_side_m)
        _, west_lng = self.move_point_by_meters(center_lat, center_lng, 270, half_side_m)

        min_lat = south_lat
        max_lat = north_lat
        min_lng = west_lng
        max_lng = east_lng

        min_tile_x, min_tile_y = self.lat_lng_to_tile(max_lat, min_lng, zoom_level)
        max_tile_x, max_tile_y = self.lat_lng_to_tile(min_lat, max_lng, zoom_level)

        min_x = min(min_tile_x, max_tile_x)
        max_x = max(min_tile_x, max_tile_x)
        min_y = min(min_tile_y, max_tile_y)
        max_y = max(min_tile_y, max_tile_y)

        os.makedirs(output_folder, exist_ok=True)

        downloaded_count = 0
        total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)

        # Use Mapbox as alternative source
        success = True
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                url = f"https://api.mapbox.com/v4/mapbox.satellite/{zoom_level}/{x}/{y}@1x.png?access_token={self.api_key}"
                filename = os.path.join(output_folder, f"tile_z{zoom_level}_x{x}_y{y}.png")
                with requests.get(url, stream=True) as r:
                    if r.status_code == 200:
                        with open(filename, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    else:
                        success = False
        
        return min_x, min_y, success


    # ===== Function: meters_per_pixel =====
    def meters_per_pixel(self, zoom, latitude):
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
    def merge_tiles_to_geotiff(self, tile_folder, tile_size_px, zoom_level, output_filename, min_x, min_y):
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
            return

        xs = sorted(set(t[0] for t in tiles))
        ys = sorted(set(t[1] for t in tiles))
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        tiles_x = max_x - min_x + 1
        tiles_y = max_y - min_y + 1

        width_px = tiles_x * tile_size_px
        height_px = tiles_y * tile_size_px

        # --- SAFETY CHECK: Prevent Decompression Bomb DOS Attack ---
        # Calculate expected size based on area_size_m
        # Ground resolution (m/px)
        res_m_px = self.meters_per_pixel(zoom_level, self.center_lat)
        
        # Expected pixels (pixels per side)
        expected_px_side = self.area_size_m / res_m_px
        
        # Max reasonable pixels (area) with safety factor (e.g., 3x buffer for tile alignment/padding)
        # Using area because width/height can vary depending on shape
        expected_area_px = expected_px_side * expected_px_side
        max_allowed_area_px = expected_area_px * 9  # 3x linear dimension = 9x area
        
        # Absolute hard limit (e.g., 25000x25000 = 625M pixels, well below typical DOS limits of 1.3B)
        HARD_LIMIT_PIXELS = 500_000_000  # 500 Megapixels

        current_area_px = width_px * height_px

        # Check relative to expected size
        if current_area_px > max_allowed_area_px and current_area_px > 10_000_000: # Only check if image is reasonably big (>10MP)
             # But if it's smaller than the hard limit, we might allow it if user asked for it? 
             # The user asked to filter bad images that are obviously wrong.
             pass

        if current_area_px > HARD_LIMIT_PIXELS:
            error_msg = (
                f"Generated image too large: {width_px}x{height_px} ({current_area_px} px). "
                f"Limit is {HARD_LIMIT_PIXELS} px. "
                f"Expected approx {int(expected_px_side)}x{int(expected_px_side)} based on area_size_m={self.area_size_m}."
            )
            print(f"❌ ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Check against expected size (if > 10x expected, it's likely a bug)
        if current_area_px > (expected_area_px * 25): # 5x linear dimension mismatch
             error_msg = (
                f"Generated image anomaly: {width_px}x{height_px} ({current_area_px} px) "
                f"is way larger than expected {int(expected_px_side)}x{int(expected_px_side)}. "
                f"Check lat/lon or tile calculation."
            )
             print(f"❌ ERROR: {error_msg}")
             raise ValueError(error_msg)
        # -----------------------------------------------------------

        try:
            mosaic = Image.new('RGB', (width_px, height_px))
        except Image.DecompressionBombError:
             print(f"❌ ERROR: Image too large, exceeded PIL limit.")
             raise

        for x, y, filename in tiles:
            img = Image.open(os.path.join(tile_folder, filename))
            offset_x = (x - min_x) * tile_size_px
            offset_y = (y - min_y) * tile_size_px
            mosaic.paste(img, (offset_x, offset_y))

        tile_lat, tile_lon = self.tile_to_lat_lng(min_x, min_y, zoom_level)

        # Transform in EPSG:3857
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        top_left_lon, top_left_lat = transformer.transform(tile_lon, tile_lat)

        pixel_size = self.meters_per_pixel(zoom_level, 0.0)
        transform = from_origin(top_left_lon, top_left_lat, pixel_size, pixel_size)

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


    # ===== Function: run_pipeline =====
    def run_pipeline(self):
        """
        Main pipeline to download tiles and create GeoTIFF.
        """
        tile_tmp_dir = os.path.join(self.output_folder, "tile")
        success = False

        for zoom in range(int(self.zoom), 13, -1):
            min_x, min_y, success = self.download_area_tiles(
                self.center_lat,
                self.center_lon,
                self.area_size_m,
                zoom,
                tile_tmp_dir
            )
            if success:
                self.zoom = zoom
                break

        if not success:
            return False

        base_name = os.path.basename(os.path.normpath(self.output_folder))
        out_path = os.path.join(self.output_folder, f"{base_name}.tif")

        self.merge_tiles_to_geotiff(
            tile_folder=tile_tmp_dir,
            tile_size_px=256,
            zoom_level=self.zoom,
            output_filename=out_path,
            min_x=min_x,
            min_y=min_y
        )

        return True