from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import os
import pandas as pd
from src.utils import download_images, extract_image_features_safe
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src import constants
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# Step 1: Load the ResNet50 model for feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract image features with error handling
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

# Define the main function to run the code
def main():
    # Step 2: Set up image folders
    train_image_folder = 'images/train/'
    test_image_folder = 'images/test/'
    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)

    # Load train and test datasets
    train_data = pd.read_csv('dataset/train.csv')
    test_data = pd.read_csv('dataset/test.csv')

    # Step 3: Check for missing images before downloading
    def check_missing_images(image_links, image_folder):
        missing_images = [img for img in image_links if not os.path.exists(os.path.join(image_folder, Path(img).name))]
        return missing_images

    # For train data
    missing_train_images = check_missing_images(train_data['image_link'], train_image_folder)
    if missing_train_images:
        print(f"Downloading {len(missing_train_images)} train images...")
        download_images(missing_train_images, train_image_folder)
    else:
        print("All train images already downloaded.")

    # For test data
    missing_test_images = check_missing_images(test_data['image_link'], test_image_folder)
    if missing_test_images:
        print(f"Downloading {len(missing_test_images)} test images...")
        download_images(missing_test_images, test_image_folder)
    else:
        print("All test images already downloaded.")

    # Step 4: Extract image features for training data using multiprocessing
    print("Extracting features from training images...")
    train_image_paths = [os.path.join(train_image_folder, img_file) for img_file in train_data['image_link']]
    
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        train_features = list(tqdm(pool.imap(extract_image_features_safe, train_image_paths), total=len(train_image_paths)))
    
    train_features = np.array(train_features)

    # Step 5: Prepare training data (Combine features with metadata)
    y_train = train_data['entity_value'].values  # Target values
    group_id = train_data['group_id'].values.reshape(-1, 1)
    entity_name = train_data['entity_name'].apply(lambda x: constants.entity_name_to_numeric.get(x, -1)).values.reshape(-1, 1)

    # Ensure that train_features, group_id, and entity_name have matching lengths
    if len(train_features) == len(group_id) == len(entity_name):
        X_train = np.concatenate([train_features, group_id, entity_name], axis=1)
    else:
        raise ValueError("Mismatch in the lengths of train_features, group_id, or entity_name.")

    # Step 6: Train the model using RandomForestRegressor
    print("Training the model...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train,
                                                                  y_train,
                                                                  test_size=0.2,
                                                                  random_state=42)

    # Train the RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_split, y_train_split)

    # Validate the model
    val_predictions = model.predict(X_val)
    print(f"Validation predictions: {val_predictions[:5]}")

    # Step 7: Extract image features for test data using multiprocessing
    print("Extracting features from test images...")
    test_image_paths = [os.path.join(test_image_folder, img_file) for img_file in test_data['image_link']]
    
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        test_features = list(tqdm(pool.imap(extract_image_features_safe, test_image_paths), total=len(test_image_paths)))

    test_features = np.array(test_features)

    # Step 8: Predict entity values for test data
    print("Making predictions on test data...")
    test_predictions = model.predict(test_features)

    # Step 9: Format the predictions
    def format_prediction(value, unit):
        """Format the prediction to 'x unit' where x is a float and unit is from allowed units."""
        return f"{value:.2f} {unit}"

    # Choose appropriate unit based on entity_name
    formatted_predictions = []
    for idx, pred in enumerate(test_predictions):
        entity_name = test_data['entity_name'].iloc[idx]
        unit = constants.entity_name_to_unit.get(entity_name, 'gram')  # Default to 'gram' if not found
        formatted_predictions.append(format_prediction(pred, unit))

    # Step 10: Save predictions to CSV for submission
    output_df = pd.DataFrame({
        'index': test_data['index'],
        'prediction': formatted_predictions
    })

    output_df.to_csv('test_out.csv', index=False)
    print("Output saved to test_out.csv")

# Ensure multiprocessing works properly
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Fix for macOS multiprocessing
    main()
