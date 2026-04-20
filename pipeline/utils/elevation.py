"""
Elevation Service Module

This module provides centralized elevation data fetching and geoid conversion services.
- ElevationService: Fetch orthometric heights from OpenTopoData API
- GeoidConverter: Convert between orthometric (geoid-based) and ellipsoid heights
"""

import logging
import os
import sys
from typing import Optional

import requests
from pyproj import CRS, Transformer

# Configure logging
log_level = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ElevationService:
    """
    Service for fetching elevation data from OpenTopoData API.
    
    Returns orthometric heights (height above mean sea level/geoid),
    which must be converted to ellipsoid heights for use with Cesium/WGS84.
    """
    
    DEFAULT_DATASET = "srtm30m"
    API_URL = "https://api.opentopodata.org/v1/{dataset}"

    @staticmethod
    def get_elevation(lat: float, lon: float, dataset: str = DEFAULT_DATASET) -> Optional[float]:
        """
        Fetch elevation for given coordinates.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            dataset: Elevation dataset name (default: srtm30m)
            
        Returns:
            Elevation in meters (orthometric height), or None if unavailable
        """
        
        url = ElevationService.API_URL.format(dataset=dataset)
        params = {"locations": f"{lat},{lon}"}
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                logger.warning(f"⚠️ Elevation API returned no results for ({lat}, {lon})")
                return None
            
            elevation = results[0].get("elevation", None)
            if elevation is None:
                logger.error("❌ Elevation API result missing 'elevation' field.")
                return None
            
            return float(elevation)
                
        except requests.RequestException as e:
            logger.error(f"❌ Failed to fetch elevation: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching elevation: {e}")
            return None


class GeoidConverter:
    """
    Utility class for converting between geoid (orthometric) and ellipsoid heights.
    
    OpenTopoData returns orthometric heights (referenced to the geoid/mean sea level).
    Cesium and WGS84 use ellipsoid heights. The difference is the geoid undulation (N).
    
    Formula: h_ellipsoid = H_orthometric + N
    Where N is the geoid undulation (geoid height above the ellipsoid).
    """
    
    # EGM96 geoid grid file (must be available in PROJ data directory)
    GEOID_GRID = "egm96_15.gtx"
    
    @staticmethod
    def get_geoid_undulation(lat: float, lon: float) -> float:
        """
        Get the geoid undulation (N) at given coordinates using EGM96 model.
        
        The geoid undulation is the height of the geoid above the WGS84 ellipsoid.
        Positive values mean geoid is above ellipsoid, negative means below.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            Geoid undulation N in meters
        """
        try:
            # Create CRS with geoid correction
            # EPSG:4326 = WGS84 geographic (lat/lon)
            # EPSG:4326+5773 = WGS84 + EGM96 geoid height
            
            # WGS84 3D with ellipsoid height
            crs_wgs84_3d = CRS.from_epsg(4979)
            
            # Compound CRS: WGS84 geographic + EGM96 geoid height
            # This represents coordinates with orthometric height
            crs_geoid = CRS.compound_crs([
                CRS.from_epsg(4326),  # WGS84 geographic 2D
                CRS.from_epsg(5773)   # EGM96 geoid height
            ])
            
            # Create transformer from geoid to ellipsoid heights
            transformer = Transformer.from_crs(
                crs_geoid,     # Source: orthometric (geoid-based)
                crs_wgs84_3d,  # Target: ellipsoid height
                always_xy=True
            )
            
            # Transform with zero orthometric height to get the geoid undulation
            # Output is (lon, lat, ellipsoid_height) where h_ellipsoid = 0 + N = N
            _, _, geoid_undulation = transformer.transform(lon, lat, 0.0)
            
            logger.debug(f"Geoid undulation at ({lat}, {lon}): {geoid_undulation:.3f}m")
            return geoid_undulation
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to compute geoid undulation via PROJ: {e}")
            logger.info("Falling back to approximate EGM96 lookup...")
            return GeoidConverter._approximate_geoid_undulation(lat, lon)
    
    @staticmethod
    def _approximate_geoid_undulation(lat: float, lon: float) -> float:
        """
        Approximate geoid undulation for common regions when PROJ grids unavailable.
        
        This is a rough approximation based on regional averages.
        For accurate results, install PROJ datum grids.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            Approximate geoid undulation in meters
        """
        # Very rough regional approximations (for fallback only)
        # Europe
        if 35 <= lat <= 72 and -10 <= lon <= 40:
            return 45.0  # Average for Europe ~30-60m
        # North America
        elif 25 <= lat <= 55 and -130 <= lon <= -60:
            return -30.0  # Average for continental US
        # Asia
        elif 10 <= lat <= 55 and 60 <= lon <= 150:
            return -20.0  # Very rough average
        # Default global average
        else:
            logger.warning("Using global average geoid undulation (0m) - results may be inaccurate")
            return 0.0
    
    @staticmethod
    def orthometric_to_ellipsoid(lat: float, lon: float, orthometric_height: float) -> float:
        """
        Convert orthometric height (geoid-based) to ellipsoid height.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            orthometric_height: Height above the geoid (from OpenTopoData, DEMs, etc.)
            
        Returns:
            Ellipsoid height (suitable for WGS84/Cesium)
        """
        N = GeoidConverter.get_geoid_undulation(lat, lon)
        ellipsoid_height = orthometric_height + N
        
        logger.info(f"🏔️ Height conversion: {orthometric_height:.2f}m (orthometric) + "
                   f"{N:.2f}m (geoid) = {ellipsoid_height:.2f}m (ellipsoid)")
        
        return ellipsoid_height
    
    @staticmethod
    def ellipsoid_to_orthometric(lat: float, lon: float, ellipsoid_height: float) -> float:
        """
        Convert ellipsoid height to orthometric height (geoid-based).
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            ellipsoid_height: Height above the WGS84 ellipsoid
            
        Returns:
            Orthometric height (height above mean sea level/geoid)
        """
        N = GeoidConverter.get_geoid_undulation(lat, lon)
        orthometric_height = ellipsoid_height - N
        
        return orthometric_height


__all__ = [
    'ElevationService',
    'GeoidConverter',
]
