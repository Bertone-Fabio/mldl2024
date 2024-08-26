import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable
import torch
import os
from PIL import Image
import torchvision.transforms.functional as TF


def get_validation_recalls(eval_dataset, db_desc, q_desc, k_values, save_dir, print_results=False, faiss_gpu=False, dataset_name="Name", num_queries_to_save=5):

    """
    This function calculates and optionally prints recall_at_k metrics for a given dataset using FAISS for retrieval.

    Args:
        eval_dataset (object): The evaluation dataset object.
        db_desc (numpy.ndarray): A numpy array containing descriptors for the database images.
        q_desc (numpy.ndarray): A numpy array containing descriptors for the query images.
        k_values (list): A list of integer k values for which recall will be calculated.
        save_dir (str): The directory path to save visualizations (optional).
        print_results (bool, optional): Flag to print the recall_at_k results (default: False).
        faiss_gpu (bool, optional): Flag to use FAISS on GPU (default: False, not currently implemented).
        dataset_name (str, optional): Name of the dataset (default: "Name").
        num_queries_to_save (int, optional): Number of queries to save visualizations for (default: 5).

    Returns:
        dict: A dictionary containing recall_at_k values for each k in k_values.
        numpy.ndarray: A numpy array containing the raw predictions from FAISS search.
    """

    # Convert descriptors to numpy arrays for FAISS compatibility
    db_desc = db_desc.numpy()
    q_desc = q_desc.numpy()

    # Get the embedding size from the database descriptors
    embed_size = db_desc.shape[1]

    # Create a flat L2 FAISS index
    faiss_index = faiss.IndexFlatL2(embed_size)

    # Add database descriptors to the index
    faiss_index.add(db_desc)

    # Search for nearest neighbors for query descriptors in the index
    _, predictions = faiss_index.search(q_desc, max(k_values))

    # Initialize correct_at_k array to store recall_at_k values for each k
    correct_at_k = np.zeros(len(k_values))

    # Loop through each query and its predictions
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # Check if any element in the top N predictions is present in the ground truth
            if np.any(np.in1d(pred[:n], eval_dataset.get_positives()[q_idx])):
                # If found, all subsequent recall_at_k values for this query are also 1
                correct_at_k[i:] += 1
                # Break the inner loop for efficiency
                break

    # Calculate recall_at_k by dividing correct_at_k by the total number of queries
    correct_at_k = correct_at_k / len(predictions)

    # Create a dictionary to store recall_at_k for each k value
    recall_at_k = {k: v for (k, v) in zip(k_values, correct_at_k)}

    # Print results if requested
    if print_results:
        print('\n')  # Print a new line
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in k_values]
        table.add_row(['Recall@K'] + [f'{100*v:.2f}' for v in correct_at_k])
        print(table.get_string(title=f"Performance on {dataset_name}"))

    # Save visualizations if requested and num_queries_to_save is positive
    if num_queries_to_save > 0:
        save_predictions(predictions, eval_dataset, eval_dataset.get_positives(), save_dir, num_queries_to_save)

    return recall_at_k, predictions





def save_predictions(predictions, inference_dataset, ground_truth, save_dir, num_queries_to_save=5):
    """
    This function saves visualizations for a specified number of queries and their top retrieved images.

    Args:
        predictions (numpy.ndarray): A numpy array containing FAISS search results for each query.
        inference_dataset (object): The inference dataset object containing image paths.
        ground_truth (list): A list of ground truth positive labels for each query in the dataset.
        save_dir (str): The directory path to save visualizations.
        num_queries_to_save (int, optional): The number of queries to save visualizations for (default: 5).

    Returns:
        None
    """

    # Create the save directory if it doesn't exist, ignoring errors if it already exists
    os.makedirs(save_dir, exist_ok=True)

    # Limit the number of saved queries to the minimum between requested and available
    num_queries_to_save = min(num_queries_to_save, len(predictions))

    # Randomly select a subset of queries to save visualizations for (without replacement)
    q_idxs = np.random.choice(len(predictions), size=num_queries_to_save, replace=False)

    for q_idx in q_idxs:
        # Get the predicted nearest neighbors for the current query
        pred = predictions[q_idx]

        # Calculate the query image index based on the dataset's number of database images
        query_index = inference_dataset.num_db_images + q_idx

        # Currently commented out: Assuming query labels are not used for visualization
        # _, query_label = inference_dataset[query_index]

        # Get the query image path from the dataset
        query_image_path = inference_dataset.all_image_paths[query_index]

        # Open the query image using Pillow
        query_image = Image.open(query_image_path)

        # Create a directory for the current query's visualizations
        query_dir = os.path.join(save_dir, f"query_{q_idx}")
        os.makedirs(query_dir, exist_ok=True)

        # Save the query image with a filename indicating the query index
        query_image_filename = f"query_{q_idx}.jpg"
        query_image_path = os.path.join(query_dir, query_image_filename)
        query_image.save(query_image_path)

        # Loop through each predicted neighbor index
        for i, pred_index in enumerate(pred):
            # Currently commented out: Assuming predicted labels are not used for visualization
            # _, pred_label = inference_dataset[pred_index]

            # Get the predicted image path from the dataset
            pred_image_path = inference_dataset.all_image_paths[pred_index]

            # Open the predicted image using Pillow
            pred_image = Image.open(pred_image_path)

            # Determine the prediction type ("correct" if ground truth match, "incorrect" otherwise)
            pred_type = "correct" if pred_index in ground_truth[q_idx] else "incorrect"

            # Create a filename for the predicted image indicating its position and type
            pred_image_filename = f"pred_{i}_{pred_type}.jpg"
            pred_image_path = os.path.join(query_dir, pred_image_filename)

            # Save the predicted image
            pred_image.save(pred_image_path)

    return  # No explicit return value, function modifies file system

