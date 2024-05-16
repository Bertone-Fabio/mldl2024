import torch
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tfm
from collections import defaultdict

# Define default transformation

default_transform = tfm.Compose([
    tfm.ToTensor(),
    tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TrainDataset(Dataset):
    def __init__(self, root_dir, images_per_place=4, minimum_images_per_place=4, transform=default_transform):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.images_per_place = images_per_place
        self.minimum_images_per_place = minimum_images_per_place

        # Gather all image paths
        self.image_paths = sorted(glob(f"{root_dir}/**/*.jpg", recursive=True))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in directory {root_dir}. Please check the path.")

        # Organize images by place id
        self.place_images = defaultdict(list)
        for path in self.image_paths:
            place_id = path.split("@")[-2]
            self.place_images[place_id].append(path)

        # Ensure each place has enough images
        self.place_images = {k: v for k, v in self.place_images.items() if len(v) >= self.minimum_images_per_place}
        self.place_ids = sorted(list(self.place_images.keys()))

        # Assert to check images per place
        assert self.images_per_place <= self.minimum_images_per_place, \
            f"images_per_place should be less than or equal to minimum_images_per_place"

    def __len__(self):
        return len(self.place_ids)

    def __getitem__(self, index):
        place_id = self.place_ids[index]
        selected_paths = np.random.choice(self.place_images[place_id], self.images_per_place, replace=False)
        images = [self.transform(Image.open(path).convert('RGB')) for path in selected_paths]
        images_stack = torch.stack(images)
        labels = torch.tensor(index).repeat(self.images_per_place)
        return images_stack, labels
