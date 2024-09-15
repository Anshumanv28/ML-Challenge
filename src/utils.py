# Ensure necessary libraries are installed in Colab
import re
import os
import requests
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import urllib
from PIL import Image, ImageFile
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import sys

# Ensure correct module path
sys.path.append('/content/src')

# Check if constants module exists; use proper import
try:
    from src import constants  # Assuming constants.py is in the src directory
except ModuleNotFoundError:
    print("Ensure that 'src/constants.py' is in the correct location!")

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load the ResNet50 model for feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract image features using ResNet50
def extract_image_features_safe(image_path):
    """Extract features from an image using ResNet50, with error handling."""
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = resnet_model.predict(x)
        return features.flatten()
    except OSError:
        print(f"Image {image_path} is corrupted or truncated, skipping.")
        return np.zeros((2048,))  # Return a default feature vector on error

# Function to correct common mistakes in units
def common_mistake(unit):
    if unit in constants.allowed_units:
        return unit
    # Correct spelling differences between 'ter' and 'tre'
    if unit.replace('ter', 'tre') in constants.allowed_units:
        return unit.replace('ter', 'tre')
    # Replace 'feet' with 'foot' if applicable
    if unit.replace('feet', 'foot') in constants.allowed_units:
        return unit.replace('feet', 'foot')
    return unit

# Function to parse strings and return the number and unit
def parse_string(s):
    s_stripped = "" if s is None or str(s) == 'nan' else s.strip()
    if s_stripped == "":
        return None, None
    # Ensure the string follows the pattern of number followed by a unit
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        raise ValueError("Invalid format in {}".format(s))
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    # Check if the unit is in the allowed units
    if unit not in constants.allowed_units:
        raise ValueError(
            "Invalid unit [{}] found in {}. Allowed units: {}".format(
                unit, s, constants.allowed_units))
    return number, unit

# Function to create a black placeholder image in case of download failure
def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        print(f"Failed to create placeholder image: {str(e)}")
        return

# Function to download an image from a given URL, with retries and a delay
def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    # Skip if the image has already been downloaded
    if os.path.exists(image_save_path):
        return

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return
        except urllib.error.HTTPError as e:
            if e.code == 400:
                print(f"Skipping URL due to HTTP 400: {image_link}")
                return  # Stop retrying on HTTP 400 error
            else:
                print(f"Error downloading {image_link}: {e}. Retrying ({attempt + 1}/{retries})...")
                time.sleep(delay)

    # If retries fail, create a placeholder image
    create_placeholder_image(image_save_path)

# Function to download images with optional multiprocessing
def download_images(image_links, download_folder, allow_multiprocessing=True):
    # Create the folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    if allow_multiprocessing:
        # Use multiprocessing for faster downloads
        download_image_partial = partial(download_image,
                                         save_folder=download_folder,
                                         retries=3,
                                         delay=3)

        # Reduce the number of processes to avoid overwhelming the system in Colab
        with multiprocessing.Pool(4) as pool:  # Use fewer processes in Colab
            list(
                tqdm(pool.imap(download_image_partial, image_links),
                     total=len(image_links)))
            pool.close()
            pool.join()
    else:
        # Download images sequentially
        for image_link in tqdm(image_links, total=len(image_links)):
            download_image(image_link,
                           save_folder=download_folder,
                           retries=3,
                           delay=3)
