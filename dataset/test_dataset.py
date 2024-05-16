import os
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors


class TestDataset(data.Dataset):
    def __init__(self, root_dir, db_dir="database", query_dir="queries", positive_threshold=25):
        """
        Dataset comprising database and query images for validation/testing.
        Parameters:
        -----------
        root_dir : str, path to the validation or test set containing {db_dir} and {query_dir} folders.
        db_dir : str, name of the database folder.
        query_dir : str, name of the queries folder.
        positive_threshold : int, distance in meters to consider a match as positive.
        """
        super().__init__()
        self.root_dir = root_dir
        self.db_dir = os.path.join(root_dir, db_dir)
        self.query_dir = os.path.join(root_dir, query_dir)
        self.dataset_name = os.path.basename(root_dir)

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Directory {self.root_dir} does not exist")
        if not os.path.exists(self.db_dir):
            raise FileNotFoundError(f"Directory {self.db_dir} does not exist")
        if not os.path.exists(self.query_dir):
            raise FileNotFoundError(f"Directory {self.query_dir} does not exist")

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Collect paths and UTM coordinates for images
        self.db_image_paths = sorted(glob(os.path.join(self.db_dir, "**", "*.jpg"), recursive=True))
        self.query_image_paths = sorted(glob(os.path.join(self.query_dir, "**", "*.jpg"), recursive=True))
        
        if len(self.db_image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.db_dir}, please check the path")
        if len(self.query_image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.query_dir}, please check the path")

        # UTM coordinates extraction
        self.db_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.db_image_paths]).astype(float)
        self.query_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.query_image_paths]).astype(float)

        # Identify positive matches within the threshold distance
        knn_model = NearestNeighbors(n_jobs=-1)
        knn_model.fit(self.db_utms)
        self.query_positives = knn_model.radius_neighbors(self.query_utms, radius=positive_threshold, return_distance=False)

        self.all_image_paths = self.db_image_paths + self.query_image_paths

        self.num_db_images = len(self.db_image_paths)
        self.num_query_images = len(self.query_image_paths)

    def __getitem__(self, idx):
        image_path = self.all_image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        transformed_image = self.image_transform(image)
        return transformed_image, idx

    def __len__(self):
        return len(self.all_image_paths)

    def __repr__(self):
        return f"<Dataset {self.dataset_name} - Queries: {self.num_query_images}; Database: {self.num_db_images}>"

    def get_positives(self):
        return self.query_positives
