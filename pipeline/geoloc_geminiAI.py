import os
import json
from PIL import Image
import google.genai as genai
from google.genai.types import GenerateContentConfig
from collections import Counter

# 1. Configuration and Initialization
# Make sure the GEMINI_API_KEY environment variable is set.

class GeminiGeolocator:
    """
    A class to perform image-based geolocation using Google's Gemini model.
    It analyzes an image for unique geographic or architectural details
    and returns GPS coordinates with the highest possible precision.
    """

    def __init__(self, exclude_keywords=None, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the GeminiGeolocator client and configuration.
        
        Args:
            model_name: The Gemini model to use for geolocation.
        """
        try:
            self.client = genai.Client()
        except Exception as e:
            raise RuntimeError(
                f"Error initializing Gemini client: {e}. "
                "Make sure the GEMINI_API_KEY environment variable is set."
            )

        self.model_name = model_name
        self.exclude_keywords = exclude_keywords or ["top_view"]
        self.system_prompt = (
            "You are a highly specialized geographic AI expert. "
            "Your primary goal is to identify specific, non-generic landmarks in the provided image "
            "(e.g., street signs, unique architectural details, specific shop names, utility poles, mountain peaks, road markings). "
            "Use these unique features to cross-reference with satellite and street-view data "
            "to determine the coordinates with the highest possible precision."
        )
        self.user_prompt = (
            "ANALYZE THE IMAGE FOR UNIQUE DETAILS. Provide the GPS coordinates (latitude and longitude) "
            "where the photo was taken with the highest possible precision (ideally within 10 meters). "
            "Only if the image is entirely devoid of unique landmarks, you can lower the accuracy. "
            "Return strictly in JSON format like this: "
            '{"latitude": <value>, "longitude": <value>}.'
        )


    def geolocate_image(self, image_path: str):
        """
        Sends an image to Gemini to obtain its estimated GPS coordinates.
        
        Args:
            image_path: Path to the image to analyze.
        
        Returns:
            A dictionary with 'latitude' and 'longitude' if successful, otherwise None.
        """
        try:
            # Load the image
            img = Image.open(image_path)

            # Prepare configuration
            config = GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
            )

            # Combine the image and prompt
            contents = [img, self.user_prompt]

            # Send request to Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            # Parse JSON response
            try:
                coords = json.loads(response.text.strip())
                latitude = coords.get("latitude")
                longitude = coords.get("longitude")

                if latitude is not None and longitude is not None:
                    return {'latitude': latitude, 'longitude': longitude}
                return None

            except json.JSONDecodeError:
                return None

        except Exception:
            return None
    

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
                result = self.geolocate_image(image_path)
                if result:
                    all_predictions.append((result['latitude'], result['longitude']))
            except Exception:
                continue

        # Determine the most common prediction
        counter = Counter(all_predictions)
        most_common = counter.most_common(1)[0][0] if counter else None

        return most_common, counter