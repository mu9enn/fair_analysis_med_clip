import os
import pandas as pd
from PIL import Image
import torch

def get_categories_list(csv_file_path: str) -> list:
    """Get list of column names from CSV header."""
    df = pd.read_csv(csv_file_path, nrows=0)
    return df.columns.tolist()[7:]

def pil_loader(path: str) -> Image.Image:
    """Loads an image from a file path using PIL."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, csv_path, mode, cxr_filepath):
        """Initialize the dataset with transforms, CSV path, mode, and image directory."""
        print(f"==>path: {csv_path}")
        print(f"==>mode: {mode}")
        annotations = pd.read_csv(csv_path)
        self.samples = [(annotations.loc[i, 'Image Index'], annotations.loc[i, 'Finding Labels'])
                        for i in range(len(annotations))]
        self.mode = mode
        self.transform = transform
        self.cxr_filepath = cxr_filepath

        categories = get_categories_list(csv_path)
        self.diagnosis_map = {str(categories[i]): i for i in range(len(categories))}

    def __getitem__(self, i):
        """Get an image and its label by index."""
        image_id, target = self.samples[i]
        path = os.path.join(self.cxr_filepath, image_id)
        try:
            img = pil_loader(path)
            image = self.transform(img)
            target = self.diagnosis_map[target]
            return image, target
        except IOError as e:
            print(f"Failed to load image {path}: {e}")
            # Return a dummy tensor or handle gracefully in production code
            return torch.zeros((3, 224, 224)), 0

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)