"""Satellite tile download and GeoTIFF assembly utilities."""

import logging
import math
import os

import numpy as np
import rasterio
import requests
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin

from pipeline.utils.coordinate_transforms import CoordinateTransforms


logger = logging.getLogger(__name__)


class SatelliteTileDownloader:
    """Download satellite tiles from Mapbox and merge them into a GeoTIFF."""

    def __init__(self, center_lat: float, center_lon: float, area_size_m: int, zoom: int, output_folder: str):
        """Initialize downloader parameters.

        Args:
            center_lat: Center latitude in decimal degrees.
            center_lon: Center longitude in decimal degrees.
            area_size_m: Side length of the target area in meters.
            zoom: Preferred zoom level.
            output_folder: Directory where intermediate and output files are written.
        """
        self.api_key = os.environ.get("MAPBOX_API_KEY")
        self.map_type = "satellite"
        self.center_lat = float(center_lat)
        self.center_lon = float(center_lon)
        self.area_size_m = int(area_size_m)
        self.zoom = int(zoom)
        self.output_folder = output_folder

    def clamp_lat(self, lat: float) -> float:
        """Clamp latitude to the valid Web Mercator range."""
        return max(min(lat, 85.05112878), -85.05112878)

    def normalize_lng(self, lng: float) -> float:
        """Normalize longitude to the [-180, 180] range."""
        while lng < -180:
            lng += 360
        while lng > 180:
            lng -= 360
        return lng

    def lat_lng_to_tile(self, lat: float, lng: float, zoom: int) -> tuple[int, int]:
        """Convert latitude/longitude to XYZ tile coordinates at a given zoom."""
        lat = self.clamp_lat(lat)
        lng = self.normalize_lng(lng)

        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x_tile = int((lng + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)

        x_tile = max(0, min(int(n) - 1, x_tile))
        y_tile = max(0, min(int(n) - 1, y_tile))
        return x_tile, y_tile


    def tile_to_lat_lng(self, x: int, y: int, zoom: int) -> tuple[float, float]:
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


    def move_point_by_meters(self, lat, lng, bearing, distance_m):
        """Delegated to CoordinateTransforms centralized function."""
        return CoordinateTransforms.move_point_by_meters(lat, lng, bearing, distance_m)


    # ===== Function: download_area_tiles =====
    def download_area_tiles(self, center_lat, center_lng, area_side_meters, zoom_level, output_folder="map_tiles", map_type="satellite"):
        """
        Download individual tiles (256x256) for a specified area using the MapBox Raster API.

        Args:
            center_lat (float): Latitude of the center of the area.
            center_lng (float): Longitude of the center of the area.
            area_side_meters (float): Length of the side of the square area in meters.
            zoom_level (int): Zoom level.
            output_folder (str): Folder where tiles are saved.
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


    def meters_per_pixel(self, zoom, latitude):
        """Delegated to CoordinateTransforms centralized function."""
        return CoordinateTransforms.meters_per_pixel(zoom, latitude)

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

        # --- Safety checks to prevent pathological image sizes ---
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
        if current_area_px > max_allowed_area_px and current_area_px > 10_000_000:
            # Preserve backward-compatible behavior while keeping this check explicit.
            pass

        if current_area_px > HARD_LIMIT_PIXELS:
            error_msg = (
                f"Generated image too large: {width_px}x{height_px} ({current_area_px} px). "
                f"Limit is {HARD_LIMIT_PIXELS} px. "
                f"Expected approx {int(expected_px_side)}x{int(expected_px_side)} based on area_size_m={self.area_size_m}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check against expected size (if > 10x expected, it's likely a bug)
        if current_area_px > (expected_area_px * 25):  # 5x linear dimension mismatch
            error_msg = (
                f"Generated image anomaly: {width_px}x{height_px} ({current_area_px} px) "
                f"is way larger than expected {int(expected_px_side)}x{int(expected_px_side)}. "
                f"Check lat/lon or tile calculation."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        # ----------------------------------------------------------

        try:
            mosaic = Image.new('RGB', (width_px, height_px))
        except Image.DecompressionBombError:
            logger.error("Image too large: PIL DecompressionBombError triggered.")
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


    def run_pipeline(self) -> bool:
        """
        Run tile download and merge steps to produce the output GeoTIFF.

        Returns:
            True on success, False otherwise.
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


# Backward compatibility alias used across the project.
satelliteTileDownloader = SatelliteTileDownloader