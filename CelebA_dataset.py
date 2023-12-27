'''
This script downloads and extracts the CelebA dataset. Feel free to modify this script to suit your needs.
'''

import os
import requests
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Function to download and extract CelebA dataset

def download_and_extract_CelebA():
    celeba_url = 'https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1'
    download_path = 'celeba.zip'
    extraction_path = 'celeba_dataset'

    # Check if CelebA is already downloaded
    if os.path.exists('celeba_dataset'):
        print("CelebA dataset is already downloaded.")
        return

    # Download CelebA zip file
    response = requests.get(celeba_url, stream=True)
    with open(download_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    # Extract CelebA dataset
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

    os.remove(download_path)

# Function to load attributes from CelebA dataset
def load_attributes():
    attr_path = 'celeba_dataset/celeba/list_attr_celeba.txt'
    df = pd.read_csv(attr_path, skiprows=1, delim_whitespace=True)
    return df

# Function to display sample images from CelebA dataset
def display_sample_images(image_dir, num_images=5):
    image_files = os.listdir(image_dir)[:num_images]

    plt.figure(figsize=(15, 5))
    for i, image_file in enumerate(image_files):
        plt.subplot(1, num_images, i + 1)
        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i + 1}")

    plt.show()


if __name__ == "__main__":
    download_and_extract_CelebA()
    attributes_df = load_attributes()

    print("CelebA Attributes:")
    print(attributes_df.head())

    image_dir = 'celeba_dataset/celeba/images'
    display_sample_images(image_dir)

    # TODO: display images with specific attributes
    # TODO: greyscale and sale images
