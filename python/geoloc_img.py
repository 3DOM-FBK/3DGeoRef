import torch
from geoclip import GeoCLIP
import argparse
from collections import Counter
import os
import platform
import json


# ===== Function: parse_args =====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True, help="Path to input folder")
    parser.add_argument("--nr_prediction", type=int, default=3, help="Number of gps prediction (default: 3)")
    
    return parser.parse_args()


# ===== Function: predict_most_common_location_from_folder =====
def predict_most_common_location_from_folder(model, folder_path, top_k=3, exclude_keywords=["top_view"]):
    """
    Predicts the position from multiple images in a folder, excluding those containing keywords (e.g. “top”).

    Parameters:
        model: instance of GeoCLIP or similar
        folder_path (str): path to the folder with images
        top_k (int): top-k to be considered for each image
        exclude_keywords (list[str]): keywords to be excluded in the image name

    Returns:
        most_common (lat, lon): all collected predictions
    """
    all_predictions = []
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        and not any(keyword in f.lower() for keyword in exclude_keywords)
    ])

    if not image_files:
        return None, None

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        gps_preds, probs = model.predict(image_path, top_k=top_k)
        all_predictions.extend([tuple(gps) for gps in gps_preds])

    counter = Counter(all_predictions)
    most_common = counter.most_common(1)[0][0] if counter else None

    return most_common, counter


# ===== Function: main =====
if __name__ == "__main__":
    model = GeoCLIP()

    args = parse_args()
    image_path = args.input_folder
    nr_prediction = args.nr_prediction

    most_common, counter = predict_most_common_location_from_folder(model, image_path, top_k=nr_prediction)

    if most_common:
        lat = round(float(most_common[0]), 6)
        lon = round(float(most_common[1]), 6)
        print (json.dumps({"lat": lat, "lon": lon}))