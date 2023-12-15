import os
import torch
from torch.utils.data import Dataset
import pydicom  # Used to read .dcm files
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import pandas as pd

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

def _convert_to_8bit(image):
    """
    Convert DICOM image to 8-bit grayscale image
    """
    image = image - np.min(image)
    image = image / np.max(image)
    image = (image * 255).astype(np.uint8)
    return image

def downsample(image):
    """
    Downsample the given DICOM image
    """
    # Convert DICOM image to 8-bit grayscale image
    image_8bit = _convert_to_8bit(image)

    # Create PIL image and downsample
    pil_image = Image.fromarray(image_8bit)
    original_size = pil_image.size  # (width, height)
    new_size = (original_size[0] // 2, original_size[1] // 2)
    resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

    return resized_image  

class PretrainDataset(Dataset):
    def __init__(self, root_dir):
        """
        Initialization function
        :param root_dir: The root directory of the data, e.g., 'path/to/Data'
        """
        self.root_dir = root_dir
        self.samples = self._load_samples()
        # Declare functions as attributes to improve training efficiency
        self.select_variant = select_variant
        self.find_black_pixels_side = find_black_pixels_side
        self.downsample = downsample
        
        
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
                    # print(file_path)
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
        image_tensor = torch.from_numpy(enhanced_image).unsqueeze(0).float()

        # Normalize
        normalize = transforms.Normalize(mean=[22.05], std=[42.99])
        norm_image = normalize(image_tensor)
        
        # return augmented_image, image_tensor, norm_image
        return norm_image


class class_Dataset(Dataset):
    def __init__(self, root_dir, label_path):
        self.root_dir = root_dir
        self.labels = pd.read_excel(label_path)
        self.samples = self._load_samples()
        # Declare functions as attributes to improve training efficiency
        self.select_variant = select_variant
        self.find_black_pixels_side = find_black_pixels_side
        self.downsample = downsample

    def _load_samples(self):
        samples = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                label_rows = self.labels[self.labels['ID1'] == subdir]
                for index, label_row in label_rows.iterrows():
                    left_files = ['1-1.dcm', '1-2.dcm']
                    right_files = ['1-3.dcm', '1-4.dcm']
                    left_paths = [os.path.join(subdir_path, file) for file in left_files]
                    right_paths = [os.path.join(subdir_path, file) for file in right_files]

                    label = 0 if label_row['classification'] == 'Benign' else 1
                    side = label_row['LeftRight']

                    # Check file existence and add
                    if side == 'L' and all(os.path.exists(path) for path in left_paths):
                        for variant in range(8):
                            samples.append((left_paths, variant, label))
                    elif side == 'R' and all(os.path.exists(path) for path in right_paths):
                        for variant in range(8):
                            samples.append((right_paths, variant, label))

        return samples

    def __len__(self):
        """
        Return the number of samples in the dataset
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_paths, variant, label = self.samples[idx]
        # Load images
        images = [pydicom.dcmread(path).pixel_array for path in file_paths]       
        new_images = []
        for dicom_image in images:
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
            image_tensor = torch.from_numpy(enhanced_image).unsqueeze(0).float()

            # Normalize
            normalize = transforms.Normalize(mean=[22.05], std=[42.99])
            norm_image = normalize(image_tensor)

                
            new_images.append(norm_image)
        
        # Return images and label
        return new_images[0], new_images[1], label