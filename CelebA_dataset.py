'''
This script downloads and extracts the CelebA dataset. Feel free to modify this script to suit your needs.
'''

import os
import requests
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split


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
def load_attributes(attr_path='celeba_dataset/list_attr_celeba.txt'):
    df = pd.read_csv(attr_path, skiprows=1, delim_whitespace=True)
    return df


def batch_greyscale_conversion(input_folder, output_folder):
    """
    Convert all images in the input folder to greyscale and save them in the output folder.

    Parameters:
    - input_folder (str): Path to the folder containing input images.
    - output_folder (str): Path to the folder to save greyscaled images.

    """

    if os.path.exists(output_folder):
        print("grey images already exist.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if the file is an image (you can add more image file extensions if needed)
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Generate output path for greyscaled image
            output_path = os.path.join(output_folder, f"{filename}")

            # Perform greyscale conversion
            greyscale_conversion(input_path, output_path)

def greyscale_conversion(input_image_path, output_image_path):
    """
    Convert an image to greyscale.

    Parameters:
    - input_image_path (str): Path to the input image.
    - output_image_path (str): Path to save the greyscale image.

   """
    img = Image.open(input_image_path).convert('L')
    img.save(output_image_path)

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

def display_images_with_attributes(image_dir, attributes_df, attribute_name, attribute_value, num_images=5):
    # Filter images based on the specified attribute
    filtered_df = attributes_df[attribute_name][attributes_df[attribute_name] == attribute_value]
    image_files = filtered_df.index[:num_images]

    plt.figure(figsize=(15, 5))
    for i, image_file in enumerate(image_files):
        plt.subplot(1, num_images, i + 1)
        image_path = os.path.join(image_dir, f'{image_file}')
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i + 1}")

    plt.show()

def select_attributes_celeba_dataset(input_folder, output_folder, attributes_df, selected_attributes):
    """
    Select a subset of attributes in the CelebA dataset.

    Parameters:
    - input_folder (str): Path to the folder containing CelebA images.
    - output_folder (str): Path to the folder to save the subset of images.
    - attributes_df (pd.DataFrame): DataFrame containing CelebA attributes.
    - selected_attributes (list): List of attribute names to use.
    """
    # Create output folder for the selected subset of images
    subset_folder = os.path.join(output_folder, 'selected')
    # if os.path.exists(subset_folder):
    #     print("grey images already exist.")
    #     return

    # Create the output folder if it doesn't exist
    #os.makedirs(subset_folder, exist_ok=True)

    # Reset the index to ensure 'index' becomes a column
    attributes_df_reset = attributes_df.reset_index()


    # Create a new DataFrame containing only the selected attributes
    subset_attributes_df_excluding_index = attributes_df_reset[selected_attributes][
        attributes_df_reset[selected_attributes] == 1].fillna(0).applymap(int)

    # Concatenate 'index' column with the new DataFrame
    subset_attributes_df = pd.concat([attributes_df_reset['index'], subset_attributes_df_excluding_index], axis=1)
    subset_attributes_df = subset_attributes_df.rename(columns={'index': 'image_id'})

    # Remove rows with all zeros
    valid_indices = subset_attributes_df.loc[(subset_attributes_df.iloc[:, 1:] != 0).any(axis=1)]

    # Save attributes as a CSV file
    attributes_csv_path = os.path.join(output_folder, 'subset_attributes.csv')
    valid_indices.to_csv(attributes_csv_path, index=False, sep=' ')

    # Split the subset into train and test sets
    # Copy images to the subset folder
    for image_id in valid_indices['image_id']:
        image_path = os.path.join(input_folder, f'{image_id}')
        shutil.copy(image_path, os.path.join(subset_folder, f'{image_id}'))

        print(f"Copying {image_id}...")




if __name__ == "__main__":
    download_and_extract_CelebA()
    attr_path = 'C:\\Users\\tosic\\tensorflow_datasets\\celeba_dataset\\celeba\\list_attr_celeba.txt'
    image_dir = 'C:\\Users\\tosic\\tensorflow_datasets\\celeba_dataset\\celeba\\images'
    output_path_greyscale = ('C:\\Users\\tosic\\tensorflow_datasets\\celeba_dataset\\greyscale_images')

    selected_attributes = ["Bushy_Eyebrows", "Smiling", "Young", "Attractive", "Eyeglasses", "Wearing_Earrings",
                           "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Blond_Hair"]

    attributes_df = load_attributes(attr_path)

    #print("CelebA Attributes:")
    #print(attributes_df.head())

    display_sample_images(image_dir)

    # display images with specific attributes, possible attributes -1 and 1
    display_images_with_attributes(image_dir, attributes_df, 'Attractive', 1)

    # convert images to greyscale
    batch_greyscale_conversion(image_dir, output_path_greyscale)

    output_folder_selected = 'C:\\Users\\tosic\\tensorflow_datasets\\celeba_dataset'
    # only use selected attributes
    select_attributes_celeba_dataset(output_path_greyscale, output_folder_selected, attributes_df, selected_attributes)


