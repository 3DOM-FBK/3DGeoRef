"""
Geolocation Module

This module provides various geolocation services for estimating GPS coordinates
from images using different AI models and approaches.

Available Models:
- GeoCLIP: Fast, vision-based geolocation (pipeline.geolocation.geoclip)
- Gemini AI: Google's Gemini model for detailed analysis (pipeline.geolocation.gemini)
- Ollama: Local LLM-based geolocation (pipeline.geolocation.ollama)

Note: These modules are NOT imported automatically to save memory.
      Import them directly when needed:
      
      from pipeline.geolocation.geoclip import GeoClipBatchPredictor
      from pipeline.geolocation.gemini import GeminiGeolocator
      from pipeline.geolocation.ollama import ImageToCoordinates
"""

# Lazy loading - modules are imported only when accessed
__all__ = [
    'GeoClipBatchPredictor',
    'GeminiGeolocator',
    'ImageToCoordinates',
]
