"""
Coordinate Transformation Utilities

This module provides centralized coordinate transformation functions:
- Web Mercator (EPSG:3857) ↔ WGS84 (EPSG:4326) conversions
- Scale factor calculations at different latitudes
- Geographic distance calculations
"""

import logging
import math
from typing import Optional, Tuple

from pyproj import Transformer

logger = logging.getLogger(__name__)


class CoordinateTransforms:
    """Utility class for geographic coordinate transformations."""
    
    # Web Mercator constants
    EARTH_RADIUS_M = 6378137.0  # WGS84 ellipsoid semi-major axis
    
    @staticmethod
    def epsg3857_to_latlon(x: float, y: float) -> Optional[Tuple[float, float]]:
        """
        Convert Web Mercator (EPSG:3857) coordinates to WGS84 (EPSG:4326).
        
        Args:
            x: X coordinate in Web Mercator (meters from central meridian)
            y: Y coordinate in Web Mercator (meters from equator)
            
        Returns:
            Tuple of (latitude, longitude) in WGS84, or None on error
        """
        try:
            transformer = Transformer.from_crs(
                "EPSG:3857",  # Web Mercator
                "EPSG:4326",  # WGS84
                always_xy=True
            )
            lon, lat = transformer.transform(x, y)
            logger.debug(f"Web Mercator ({x:.2f}, {y:.2f}) → WGS84 ({lat:.8f}, {lon:.8f})")
            return lat, lon
        except Exception as e:
            logger.error(f"❌ Failed to convert EPSG:3857 to lat/lon: {e}")
            return None
    
    @staticmethod
    def latlon_to_epsg3857(lat: float, lon: float) -> Optional[Tuple[float, float]]:
        """
        Convert WGS84 (EPSG:4326) to Web Mercator (EPSG:3857).
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            Tuple of (x, y) in Web Mercator, or None on error
        """
        try:
            transformer = Transformer.from_crs(
                "EPSG:4326",  # WGS84
                "EPSG:3857",  # Web Mercator
                always_xy=True
            )
            x, y = transformer.transform(lon, lat)
            logger.debug(f"WGS84 ({lat:.8f}, {lon:.8f}) → Web Mercator ({x:.2f}, {y:.2f})")
            return x, y
        except Exception as e:
            logger.error(f"❌ Failed to convert lat/lon to EPSG:3857: {e}")
            return None
    
    @staticmethod
    def web_mercator_scale_factor(lat: float) -> float:
        """
        Compute ground resolution scale factor for Web Mercator at given latitude.
        
        Web Mercator has scaling that depends on latitude due to the projection.
        Scale factor = cos(lat_radians)
        
        Args:
            lat: Latitude in degrees
            
        Returns:
            Scale factor (0.0 to 1.0)
        """
        return math.cos(math.radians(lat))
    
    @staticmethod
    def meters_per_pixel(zoom: int, latitude: float) -> float:
        """
        Calculate ground resolution in meters/pixel at given zoom level and latitude.
        
        This follows Google Maps/Web Mercator convention.
        
        Args:
            zoom: Zoom level (0-28)
            latitude: Latitude in degrees
            
        Returns:
            Ground resolution in meters per pixel
        """
        initial_resolution = 156543.03392804097  # meters/pixel at equator, zoom 0
        scale = math.cos(math.radians(latitude))
        return initial_resolution * scale / (2 ** zoom)
    
    @staticmethod
    def move_point_by_meters(
        lat: float,
        lon: float,
        bearing: float,
        distance_m: float
    ) -> Tuple[float, float]:
        """
        Calculate new lat/lon by moving a point by distance in a given direction.
        
        Uses Haversine formula for geospatial distance calculations.
        
        Args:
            lat: Starting latitude in degrees
            lon: Starting longitude in degrees
            bearing: Direction in degrees (0=North, 90=East, 180=South, 270=West)
            distance_m: Distance to move in meters
            
        Returns:
            Tuple of (new_latitude, new_longitude)
        """
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing)
        
        # Angular distance in radians
        angular_distance = distance_m / CoordinateTransforms.EARTH_RADIUS_M
        
        # Calculate new position
        new_lat = math.asin(
            math.sin(lat_rad) * math.cos(angular_distance) +
            math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
        )
        
        new_lon = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
            math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat)
        )
        
        return math.degrees(new_lat), math.degrees(new_lon)


__all__ = ['CoordinateTransforms']
