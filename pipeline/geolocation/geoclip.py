import os
import json
import torch
from geoclip import GeoCLIP
from collections import Counter


class GeoClipBatchPredictor:
    """
    A class to perform batch geolocation predictions using GeoCLIP.
    It processes all images in a given folder, filters out unwanted views,
    and returns the most common predicted GPS location among all images.
    """

    def __init__(self, exclude_keywords=None, top_k: int = 3):
        """
        Initializes the predictor.

        Args:
            model (GeoCLIP): An instance of the GeoCLIP model or compatible geolocation model.
            exclude_keywords (list[str], optional): Keywords to exclude from image filenames. Defaults to ["top_view"].
            top_k (int, optional): Number of top GPS predictions to retrieve for each image. Defaults to 3.
        """
        self.model = GeoCLIP()
        self.top_k = top_k
        self.exclude_keywords = exclude_keywords or ["top_view"]


    def predict_folder(self, folder_path: str):
        """
        Predicts the most common GPS location from multiple images in a folder.

        Args:
            folder_path (str): Path to the folder containing images.

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
                gps_preds, probs = self.model.predict(image_path, top_k=self.top_k)
                all_predictions.extend([tuple(gps) for gps in gps_preds])
            except Exception:
                continue

        # Determine the most common prediction
        counter = Counter(all_predictions)
        most_common = counter.most_common(1)[0][0] if counter else None

        return most_common, counter