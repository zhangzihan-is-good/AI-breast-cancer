import os
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import yaml
import numpy as np
from PIL import Image
import random
import cv2

# Set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# If CUDA is used
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)

class DicomDataset(Dataset):
    def __init__(self, root_dir):
        """
        Initialization function
        :param root_dir: The root directory of the data, e.g., 'path/to/Data'
        """
        self.root_dir = root_dir
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Function to load samples, generating eight sample entries for each DICOM file
        """
        samples = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                files = [f for f in os.listdir(subdir_path) if f.endswith('.dcm')]
                for file in files:
                    file_path = os.path.join(subdir_path, file)
                    # Add eight variant entries for each file
                    for variant in range(8):
                        samples.append((file_path, variant))
        return samples

    def __len__(self):
        """
        Return the number of samples in the dataset
        """
        return len(self.samples)


    def __getitem__(self, idx):
        """
        Get the sample corresponding to the index
        """
        dicom_path, variant = self.samples[idx]
        dicom_image = pydicom.dcmread(dicom_path).pixel_array

        # Use downsampling function
        resized_image = self.downsample(dicom_image)

        # Convert the image to a numpy array
        jpg_image = np.array(resized_image)
        # Remove the first line and the last two lines
        jpg_image = jpg_image[1:-2, :]

        # Determine which side the black pixels are mainly concentrated on
        black_side = self.find_black_pixels_side(jpg_image)

        # Add black pixels on the corresponding side to make the image square for data augmentation
        if black_side == 'left':
            padding = np.zeros((jpg_image.shape[0], 187), dtype=jpg_image.dtype)
            jpg_image = np.hstack((padding, jpg_image))
        else:  # black_side == 'right'
            padding = np.zeros((jpg_image.shape[0], 187), dtype=jpg_image.dtype)
            jpg_image = np.hstack((jpg_image, padding))

        # Data augmentation: select the corresponding variant
        pil_image = Image.fromarray(jpg_image)
        augmented_image = self.select_variant(pil_image, variant)
        
        # Contrast enhancement - using CLAHE
        np_image = np.array(augmented_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(np_image)

        # Convert to PyTorch tensor
        image_tensor = torch.from_numpy(np.array(enhanced_image)).unsqueeze(0).float()

        return image_tensor
    
    @staticmethod
    def select_variant(image, variant):
        """
        Select the corresponding augmented image based on the variant index
        """
        angle = (variant % 4) * 90
        is_mirrored = variant >= 4

        rotated_image = image.rotate(angle)
        if is_mirrored:
            return rotated_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            return rotated_image


    @staticmethod
    def find_black_pixels_side(image):
        """
        Determine whether black pixels are mainly concentrated on the left or right side of the image
        :param image: PIL Image object or a numpy array
        :return: 'left' or 'right'
        """
        # Divide the image into left and right halves
        left_half = image[:, :image.shape[1] // 2]
        right_half = image[:, image.shape[1] // 2:]

        # Calculate the number of black pixels in each half
        black_pixels_left = np.sum(left_half == 0)
        black_pixels_right = np.sum(right_half == 0)

        return 'left' if black_pixels_left > black_pixels_right else 'right'

    def _convert_to_8bit(self, image):
        """
        Convert DICOM image to 8-bit grayscale image
        """
        image_8bit = image.astype(np.uint8)
        return image_8bit

    def downsample(self, image):
        """
        Downsample the given DICOM image
        """
        # Convert DICOM image to 8-bit grayscale image
        image_8bit = self._convert_to_8bit(image)

        # Create PIL image and downsample
        pil_image = Image.fromarray(image_8bit)
        original_size = pil_image.size  # (width, height)
        new_size = (original_size[0] // 2, original_size[1] // 2)
        resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

        return resized_image    
    
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)
    print("OK")
    mean = 0.
    std = 0.
    nb_samples = 0.

    for i,data in enumerate(loader):
        print(i)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

# Compute the mean and standard deviation using the dataset
mean, std = compute_mean_std(DicomDataset(root_dir='./Data'))

print(f"Mean: {mean}, Std: {std}")
