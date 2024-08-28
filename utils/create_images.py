import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
from math import ceil

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filtra solo i file immagine
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            images[filename] = img
    return images

def create_comparison_image(folders, output_path="comparison.png", border_width=10):
    fig, axes = plt.subplots(len(folders), 4, figsize=(16, 12))
    
    for i, folder in enumerate(folders):
        images = load_images_from_folder(folder)
        
        # Identifica la query e le predizioni
        query_img = images.get(next((key for key in images if "query" in key), None))  # Supponendo che l'estensione sia .jpg
        pred_imgs = [
            images.get(next((key for key in images if "pred_0" in key), None)),
            images.get(next((key for key in images if "pred_1" in key), None)),
            images.get(next((key for key in images if "pred_2" in key), None))
        ]
        
        # Visualizza la query
        axes[i, 0].imshow(query_img)
        axes[i, 0].axis('off')
        
        # Visualizza le predizioni con i bordi
        for j, pred_img in enumerate(pred_imgs):
            pred_img_name = next((key for key in images if f"pred_{j}" in key), None)
            
            if "incorrect" in pred_img_name:
                color = 'red'
            elif "correct" in pred_img_name:
                color = 'green'
            else:
                color = 'black'  # Nessun colore se il nome non contiene "correct" o "incorrect"
            
            pred_img_with_border = ImageOps.expand(pred_img, border=border_width, fill=color)
            axes[i, j + 1].imshow(pred_img_with_border)
            axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()




def visualize_images(folder_path, num_images, output_path):
    """
    Randomly samples a specified number of images from a given folder and its subfolders,
    and visualizes them in a grid. The resulting image is saved to the specified output path.

    Args:
        folder_path (str): The path to the folder containing the images.
        num_images (int): The number of images to randomly sample and display.
        output_path (str): The path to save the output image.
    """

    # Get a list of all image paths in the folder and its subfolders
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    # Check if num_images is larger than the available images
    if num_images > len(image_paths):
        num_images = len(image_paths)
        print(f"Warning: Requested {num_images} images, but only {len(image_paths)} are available.")

    # Randomly sample the specified number of images
    selected_image_paths = random.sample(image_paths, num_images)

    # Load the sampled images
    images = [plt.imread(image_path) for image_path in selected_image_paths]

    # Determine grid size based on the number of images
    grid_size = ceil(np.sqrt(num_images))
    
    # Create the figure and axes for the grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    # Display images in the grid
    for i in range(grid_size * grid_size):
        row_index = i // grid_size
        col_index = i % grid_size

        if i < num_images:
            axes[row_index, col_index].imshow(images[i])
            axes[row_index, col_index].axis('off')
        else:
            axes[row_index, col_index].axis('off')  # Hide any extra subplots

    plt.tight_layout()

    # Save the image grid to the specified output path
    plt.savefig(output_path)
    plt.show()
