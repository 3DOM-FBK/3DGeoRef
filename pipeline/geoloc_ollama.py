import os
import requests
import ollama
from time import sleep
from typing import Optional, List, Dict
from collections import Counter


class NominatimGeocoder:
    BASE_URL = "https://nominatim.openstreetmap.org"
    
    def __init__(self, user_agent: str = "MyApp/1.0"):
        """
        Args:
            user_agent: App identifier for Nominatim API usage.
        """
        self.headers = {'User-Agent': user_agent}
        self.rate_limit_delay = 1
    

    def search(self, query: str, limit: int = 1) -> List[dict]:
        url = f"{self.BASE_URL}/search"
        params = {
            'q': query,
            'format': 'json',
            'limit': limit,
            'addressdetails': 1
        }
        
        sleep(self.rate_limit_delay)
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return []
    

    def reverse(self, lat: float, lon: float) -> Optional[dict]:
        url = f"{self.BASE_URL}/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'addressdetails': 1
        }
        
        sleep(self.rate_limit_delay)
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return None
    

    def get_coordinates(self, query: str) -> Optional[Dict]:
        results = self.search(query, limit=1)
        
        if results:
            result = results[0]
            return {
                'latitude': float(result['lat']),
                'longitude': float(result['lon']),
                'display_name': result['display_name'],
                'place_id': result['place_id'],
                'type': result.get('type'),
                'importance': result.get('importance', 0)
            }
        return None


class ImageToCoordinates:
    
    def __init__(self, exclude_keywords=None, ollama_model: str = "llama3.2-vision"):
        self.geocoder = NominatimGeocoder(user_agent="ImageGeoLocator/1.0")
        self.ollama_model = ollama_model
        self._check_ollama_model()
        self.exclude_keywords = exclude_keywords or ["top_view"]
    

    def _check_ollama_model(self):
        """
        Checks if the specified Ollama model is available locally.
        """
        try:
            models = ollama.list()
            model_names = [m.model for m in models.models]

        except Exception as e:
            raise RuntimeError(f"Error checking Ollama models: {e}")
    

    def identify_building_name(self, image_path: str) -> Optional[str]:
        """
        Uses Ollama to identify the main building in an image.

        Args:
            image_path: Path to the image to analyze.

        Returns:
            The building name as a string in the format "Building Name, City, Country",
            or None if unknown or on error.
        """
        prompt = """
        You are an expert in recognizing architectural landmarks from images, including renderings or synthetic views.

        Look carefully at this image and identify the **main building** it depicts.

        Respond ONLY in **English**, using this exact format:

        "Building Name, City, Country"

        Examples:

        "Eiffel Tower, Paris, France"  
        "Colosseum, Rome, Italy"  
        "Big Ben, London, United Kingdom"

        If you are not confident about the identification, reply only with "UNKNOWN".

        Do not include any explanations or descriptions.
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )
            
            building_name = response['message']['content'].strip()
            
            if "UNKNOWN" in building_name.upper():
                return None
            
            return building_name
            
        except Exception:
            return None
    

    def identify_building_street(self, image_path: str) -> Optional[str]:
        """
        Uses Ollama to identify the street address of a building in an image.

        Args:
            image_path: Path to the image to analyze.

        Returns:
            The street address as a string in the format "Street Address, City, Country",
            or None if unknown or on error.
        """
        prompt = """
        You are an expert in recognizing architectural landmarks from images, including renderings or synthetic views.

        Look carefully at this image and identify the **main building** it depicts.

        Respond ONLY in **English**, using this exact format:

        "Street Address, City, Country"

        Examples:

        "Champ de Mars 5, Paris, France"  
        "Piazza del Colosseo 1, Rome, Italy"  
        "Westminster, London, United Kingdom"

        If you do not recognize the street address or it is not possible to determine it from the image, reply only with "UNKNOWN".

        Do not include any explanations or descriptions.
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )
            
            building_name = response['message']['content'].strip()
            
            if "UNKNOWN" in building_name.upper():
                return None
            
            return building_name
            
        except Exception:
            return None

    
    def _calculate_confidence(self, coords: Dict) -> str:
        importance = coords.get('importance', 0)
        
        if importance > 0.7:
            return 'high'
        elif importance > 0.4:
            return 'medium'
        else:
            return 'low'
    

    def process(self, image_path: str) -> Dict:
        # First attempt: identify by building name
        building_name = self.identify_building_name(image_path)
        
        coords = None
        if building_name and building_name.upper() != "UNKNOWN":
            coords = self.geocoder.get_coordinates(building_name)
        
        # Second attempt: identify by street address
        if not coords:
            building_name = self.identify_building_street(image_path)
            if building_name and building_name.upper() != "UNKNOWN":
                coords = self.geocoder.get_coordinates(building_name)
        
        # If no coordinates found
        if not coords:
            return {
                'latitude': None,
                'longitude': None
            }
        
        return {
            'latitude': coords['latitude'],
            'longitude': coords['longitude']
        }
    

    def run_pipeline(self, folder_path: str):
        """
        Predicts the most common GPS location from multiple images in a folder.

        Args:
            folder_path: Path to the folder containing images.

        Returns:
            tuple: (most_common_location, all_predictions_counter)
                where most_common_location is a (lat, lon) tuple,
                and all_predictions_counter is a Counter of all predicted coordinates.
        """
        if not os.path.isdir(folder_path):
            return None, None

        # Collect valid image files
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
            and not any(keyword in f.lower() for keyword in self.exclude_keywords)
        ])

        if not image_files:
            return None, None

        all_predictions = []

        # Process each image
        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            try:
                result = self.process(image_path)
                if result and result.get('latitude') is not None and result.get('longitude') is not None:
                    all_predictions.append((result['latitude'], result['longitude']))
            except Exception:
                continue

        # Determine the most common prediction
        counter = Counter(all_predictions)
        most_common = counter.most_common(1)[0][0] if counter else None

        return most_common, counter