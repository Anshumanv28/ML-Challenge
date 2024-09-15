import os
import numpy as np
import pandas as pd
import logging
import sys
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import gc

print("hi I am alive")
# Ensure that the path to `downloadHandler.py` is included
sys.path.append('/content/drive/MyDrive/Colab Notebooks')

from downloadHandler import handle_image_downloads  # Ensure this import is correct

# Configure logging
log_dir = '/content/drive/MyDrive/Colab Notebooks/logs/'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the ResNet50 model for feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract image features with error handling
def extract_image_features_safe(image_path):
    """Extract features from an image using ResNet50, with error handling."""
    try:
        logging.debug(f"Loading image: {image_path}")
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = resnet_model.predict(x)
        return features.flatten()
    except OSError:
        logging.error(f"Image {image_path} is corrupted or truncated, skipping.")
        return np.zeros((2048,))  # Return a default feature vector on error

# Batch size for processing images (further reduced)
batch_size = 1000

# Function to process a batch of images
def process_image_batch(image_paths, folder, batch_num, is_test=False):
    logging.info(f"Processing batch {batch_num} of size {len(image_paths)}.")
    batch_features = []
    for image_path in tqdm(image_paths):
        full_image_path = os.path.join(folder, Path(image_path).name)
        features = extract_image_features_safe(full_image_path)
        batch_features.append(features)
    
    batch_features = np.array(batch_features)
    # Save batch to disk
    np.save(f'features_batch_{batch_num}.npy', batch_features)
    
    # Cleanup
    del batch_features
    gc.collect()
    logging.info(f"Batch {batch_num} processed and saved.")

# Function to load extracted features from batches
def load_batches(batch_count):
    X_batches = []
    for i in range(batch_count):
        logging.info(f"Loading batch {i}.")
        X_batches.append(np.load(f'features_batch_{i}.npy'))
        # Cleanup after loading each batch
        del X_batches[-1]
        gc.collect()
    return np.concatenate(X_batches, axis=0)

def main():
    logging.info("Starting main function.")

    # Set up image folders
    train_image_folder = 'images/train/'
    test_image_folder = 'images/test/'
    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)
    logging.info("Image folders created or verified.")

    # Load train and test datasets
    train_data = pd.read_csv('dataset/train.csv')
    test_data = pd.read_csv('dataset/test.csv')
    logging.info("Datasets loaded.")

    # Handle image downloads
    logging.info("Handling image downloads.")
    handle_image_downloads(train_data, test_data, train_image_folder, test_image_folder)

    # Extract image features for training data in batches
    logging.info("Extracting features from training images...")
    train_image_paths = [os.path.join(train_image_folder, img_file) for img_file in train_data['image_link']]
    
    for i in range(0, len(train_image_paths), batch_size):
        batch_paths = train_image_paths[i:i + batch_size]
        process_image_batch(batch_paths, train_image_folder, i // batch_size)
    
    # Load the extracted batches
    total_batches = (len(train_image_paths) + batch_size - 1) // batch_size
    X_train = load_batches(total_batches)

    # Prepare training data (Combine features with metadata)
    logging.info("Preparing training data.")
    y_train = train_data['entity_value'].values  # Target values
    group_id = train_data['group_id'].values.reshape(-1, 1)
    entity_name = train_data['entity_name'].apply(lambda x: constants.entity_name_to_numeric.get(x, -1)).values.reshape(-1, 1)

    # Ensure that train_features, group_id, and entity_name have matching lengths
    if len(X_train) == len(group_id) == len(entity_name):
        X_train_combined = np.concatenate([X_train, group_id, entity_name], axis=1)
    else:
        raise ValueError("Mismatch in the lengths of train_features, group_id, or entity_name.")

    # Train the model using RandomForestRegressor
    logging.info("Training the model...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_combined,
                                                                  y_train,
                                                                  test_size=0.2,
                                                                  random_state=42)

    # Train the RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_split, y_train_split)

    # Validate the model
    val_predictions = model.predict(X_val)
    logging.info(f"Validation predictions: {val_predictions[:5]}")

    # Evaluate the model
    train_predictions = model.predict(X_train_split)
    train_mse = mean_squared_error(y_train_split, train_predictions)
    train_r2 = r2_score(y_train_split, train_predictions)
    logging.info(f"Training Mean Squared Error: {train_mse}")
    logging.info(f"Training R² Score: {train_r2}")

    val_mse = mean_squared_error(y_val, val_predictions)
    val_r2 = r2_score(y_val, val_predictions)
    logging.info(f"Validation Mean Squared Error: {val_mse}")
    logging.info(f"Validation R² Score: {val_r2}")

    # Extract features for test data in batches
    logging.info("Extracting features from test images...")
    test_image_paths = [os.path.join(test_image_folder, img_file) for img_file in test_data['image_link']]
    
    for i in range(0, len(test_image_paths), batch_size):
        batch_paths = test_image_paths[i:i + batch_size]
        process_image_batch(batch_paths, test_image_folder, i // batch_size, is_test=True)

    # Load the extracted batches for test data
    total_test_batches = (len(test_image_paths) + batch_size - 1) // batch_size
    test_features = load_batches(total_test_batches)

    # Predict entity values for test data
    logging.info("Making predictions on test data...")
    test_predictions = model.predict(test_features)

    # Format the predictions
    def format_prediction(value, unit):
        """Format the prediction to 'x unit' where x is a float and unit is from allowed units."""        
        return f"{value:.2f} {unit}"

    # Choose appropriate unit based on entity_name
    formatted_predictions = []
    for idx, pred in enumerate(test_predictions):
        entity_name = test_data['entity_name'].iloc[idx]
        unit = constants.entity_name_to_unit.get(entity_name, 'gram')  # Default to 'gram' if not found
        formatted_predictions.append(format_prediction(pred, unit))

    # Save predictions to CSV for submission
    output_df = pd.DataFrame({
        'index': test_data['index'],
        'prediction': formatted_predictions
    })

    output_df.to_csv('test_out.csv', index=False)
    logging.info("Output saved to test_out.csv")

    logging.info("Main function completed.")

# Ensure multiprocessing works properly
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Fix for macOS multiprocessing
    main()
