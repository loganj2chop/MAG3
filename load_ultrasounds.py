#### This process loads the dicom's into np arrays. I also measures the avg pixel values of the mask
### I often run this multiple times after I run the manifest through select_dicom.py



import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#from model_arch import UNET  # Ensure this import works correctly
import torch.nn as nn
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import pandas as pd
import pydicom
from matplotlib.animation import FuncAnimation
import numpy as np
from skimage.transform import resize
import SimpleITK as sitk


# Function to calculate the average pixel value of the mask
def avg_pixel_value(mask):
    return np.mean(mask)

# Function to calculate the median pixel value of the mask
def median_pixel_value(mask):
    return np.median(mask)

# Function to calculate the contour area of the non-black pixels
def contour_area(mask):
    return np.sum(mask > .00001)  # Count the number of non-zero pixels (non-black pixels)

def resize_image(image, target_size):
    return resize(image, target_size, preserve_range=True, anti_aliasing=True)

def crop_image(image, top_percent, bottom_percent, left_percent, right_percent, target_size):
    height, width = image.shape[-2], image.shape[-1]
    top_crop = int(height * top_percent)
    bottom_crop = int(height * bottom_percent)
    left_crop = int(width * left_percent)
    right_crop = int(width * right_percent)
    cropped_image = image[..., top_crop:height-bottom_crop, left_crop:width-right_crop]
    resized_image = resize_image(cropped_image, target_size)
    return resized_image

def load(df):
    image_files = []
    mask_files = []
    ids = []
    target_size = (256, 256)
    top_percent, bottom_percent, left_percent, right_percent = 0.1, 0.1, 0.05, 0.05
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
    
    # Load the model parameters
    model.load_state_dict(torch.load('model_pretrained.pth'))

    # Check if CUDA (GPU support) is available and move the model to GPU if it is
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    model = model.to(device)  # Move the model to the appropriate device
    model.eval()  # Set the model to evaluation mode

    for i, row in df.iterrows():
        try:
            # Load the DICOM file
            dcmfile_path = row['dcmfile']
            print('dcmfile_path', dcmfile_path)
            dicom_data = pydicom.dcmread(dcmfile_path)
            print('read dicom')        
            
            # Extract the pixel array
            pixel_array = dicom_data.pixel_array      
            print("Original pixel array shape:", pixel_array.shape)

            # Skip if the image is not in grayscale
            if pixel_array.shape[-1] == 3:
                print('working but out of shape', pixel_array.shape)
                continue

            # Process the image
            processed_image = resize_image(pixel_array, (target_size))
            processed_images_np = np.array([processed_image])

            # Crop and resize the image
            cropped_resized_image = crop_image(processed_images_np[0], top_percent, bottom_percent, left_percent, right_percent, target_size)

            # Convert the normalized images to a PyTorch tensor
            image_tensor = torch.tensor(cropped_resized_image / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.to(device)

            # Process the single image
            with torch.no_grad():  # No gradients needed
                output = model(image_tensor)
                mask = torch.sigmoid(output).cpu().numpy()  # Apply sigmoid and convert to numpy
                mask_squeezed = np.squeeze(mask[0, 0])  # Extract and squeeze the mask
                
                # Add the processed image and mask to lists
                image_files.append(image_tensor.cpu().numpy())
                mask_files.append(mask_squeezed)

                # Compute metrics
                avg_value = avg_pixel_value(mask_squeezed)
                median_value = median_pixel_value(mask_squeezed)
                area_value = contour_area(mask_squeezed)

                # Append the id and metrics
                ids_list = [row['study_id'], row['dcmfile'], row['textafter'], row['left'], row['right'],
                            avg_value, median_value, area_value]
                ids.append(ids_list)

        except Exception as e:
            print(e)

    # Convert to NumPy arrays
    image_files_np = np.array(image_files)
    mask_files_np = np.array(mask_files)
    return image_files_np, mask_files_np, ids

# Load the data and process
df = pd.read_csv('Mag3OGalltoselect.csv')
image_files_np, mask_files_np, ids = load(df)

# Save the arrays
np.save('imagenp', image_files_np)
np.save('masknp', mask_files_np)

# Save the ids list to a CSV file
ids_df = pd.DataFrame(ids, columns=['study_id', 'dcmfile', 'textafter', 'left', 'right', 'avg_pixel_value', 'median_pixel_value', 'contour_area'])
ids_df.to_csv('OGids_with_metrics.csv', index=False)