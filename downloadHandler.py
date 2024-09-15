import os
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

print("Downloader here")

# Configure logging
log_dir = '/content/drive/MyDrive/Colab Notebooks/logs/'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'download.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def download_image(image_link, image_folder):
    """Download a single image and log progress."""
    try:
        logging.debug(f"Starting download of image {image_link}")
        filename = Path(image_link).name
        image_save_path = os.path.join(image_folder, filename)
        utils.download_image(image_link, image_folder)  # Adjust this to your actual download function
        logging.debug(f"Completed download of image {image_link}")
    except Exception as e:
        logging.error(f"Error downloading image {image_link}: {e}")

def download_images_in_batches(image_links, image_folder, batch_size=1000):
    """Download images in batches and log progress."""
    total_images = len(image_links)
    logging.info(f"Starting to download images in batches of {batch_size}.")
    
    # Limit the number of concurrent threads
    max_threads = min(8, os.cpu_count() or 1)  # Adjust this number based on your system resources
    logging.info(f"Using {max_threads} concurrent threads.")
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for i in range(0, total_images, batch_size):
            batch_links = image_links[i:i + batch_size]
            logging.info(f"Downloading batch {i // batch_size + 1}/{(total_images + batch_size - 1) // batch_size}")
            
            futures = [executor.submit(download_image, link, image_folder) for link in batch_links]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error in batch processing: {e}")
    
    logging.info("All batches downloaded.")
    gc.collect()  # Collect garbage if needed
    logging.info("Garbage collection completed.")

def check_missing_images(image_links, image_folder):
    """Check for missing images and return the list of missing images."""
    existing_images = set(Path(image_folder).glob('*'))
    missing_images = [img for img in image_links if Path(img).name not in existing_images]
    logging.info(f"Found {len(missing_images)} missing images.")
    return missing_images

def handle_image_downloads(train_data, test_data, train_image_folder, test_image_folder):
    """Handle downloading of images for both train and test datasets."""
    def download_images_for_data(data, folder, data_type):
        logging.info(f"Starting download for {data_type} data.")
        missing_images = check_missing_images(data['image_link'], folder)
        if missing_images:
            start_time = time.time()
            logging.info(f"Downloading {len(missing_images)} images for {data_type}...")
            download_images_in_batches(missing_images, folder)
            logging.info(f"Images download completed for {data_type} in {time.time() - start_time:.2f} seconds.")
        else:
            logging.info(f"All images already downloaded for {data_type}.")

    # For train data
    download_images_for_data(train_data, train_image_folder, 'train')
    
    # For test data
    download_images_for_data(test_data, test_image_folder, 'test')
