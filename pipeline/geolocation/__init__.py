"""
Geolocation Module

This module provides various geolocation services for estimating GPS coordinates
from images using different AI models and approaches.

Available Models:
- GeoCLIP: Fast, vision-based geolocation
- Gemini AI: Google's Gemini model for detailed analysis
- Ollama: Local LLM-based geolocation
"""

from .geoclip import GeoClipBatchPredictor
from .gemini import GeminiGeolocator
from .ollama import ImageToCoordinates

__all__ = [
    'GeoClipBatchPredictor',
    'GeminiGeolocator',
    'ImageToCoordinates',
]
