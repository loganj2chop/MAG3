import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
from skimage.transform import resize
import segmentation_models_pytorch as smp
import pydicom
import SimpleITK as sitk
import os

# Define your existing helper functions like resize_image, crop_image, etc.
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

# Initialize the UNet model
def load_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
    model.load_state_dict(torch.load('model_pretrained.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return model, device

# Load and process images and masks
def load_data(df, model, device):
    underlay_images = []
    ids = []
    orginal_image = []
    target_size = (256, 256)

    for _, row in df.iterrows():
        try:
            # Load DICOM file and preprocess
            dcmfile_path = row['dcmfile']
            dicom_data = pydicom.dcmread(dcmfile_path)
            pixel_array = dicom_data.pixel_array
            if len(pixel_array.shape) > 2 and pixel_array.shape[-1] == 3:
                continue  # Skip non-grayscale images
            
            # Resize and crop image
            processed_image = resize_image(pixel_array, target_size)
            cropped_resized_image = crop_image(processed_image, 0.1, 0.1, 0.05, 0.05, target_size)
            image_tensor = torch.tensor(cropped_resized_image / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # Apply model to get mask
            with torch.no_grad():
                output = model(image_tensor)
                mask = torch.sigmoid(output).cpu().numpy().squeeze(0)
                ###mask_cleaned = (mask > 0.001).astype(np.uint8)  ## OLD
                mask_cleaned = (mask > 0.5).astype(np.uint8)


                # Apply mask to the image
                underlay_image = cropped_resized_image * mask_cleaned
                underlay_images.append(underlay_image)
                orginal_image.append(cropped_resized_image)

                # Store metadata
                ids.append([
                    row['study_id'], row['dcmfile'], row['textafter'], row['left'], row['right'], row['Bad'],row['new-old'], row['function']
                ])
        except Exception as e:
            print(f"Error processing file {row['dcmfile']}: {e}")

    # Convert to NumPy arrays
    underlay_images_np = np.array(underlay_images)
    original_image_np = np.array(orginal_image)
    return underlay_images_np, ids, original_image_np

# Main function to perform 5-fold split, stratified by "Bad" and grouped by "study_id"
def create_folds(df):
    model, device = load_model()
    gkf = GroupKFold(n_splits=5)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df['study_id'])):
        # Split the DataFrame and process images for each fold
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        # Load and process data for train and test splits
        underlay_train, ids_train, original_train = load_data(df_train, model, device)
        underlay_test, ids_test, original_test = load_data(df_test, model, device)

        # Save the train/test splits for each fold
        np.save(f'underlay_new_{fold+1}_train.npy', underlay_train)
        np.save(f'underlay_new_{fold+1}_test.npy', underlay_test)

        np.save(f'original_new_{fold+1}_train.npy', original_train)
        np.save(f'original_new_{fold+1}_test.npy', original_test)

        
        

        # Convert ids lists to DataFrames and save as CSV
        ids_df_train = pd.DataFrame(ids_train, columns=['study_id', 'dcmfile', 'textafter', 'left', 'right','Bad','new-old','function'])
        ids_df_test = pd.DataFrame(ids_test, columns=['study_id', 'dcmfile', 'textafter', 'left', 'right', 'Bad','new-old','function'])

        ids_df_train.to_csv(f'ids_with_metrics_fin_mix{fold+1}_train.csv', index=False)  ### adding mix for both new and old
        ids_df_test.to_csv(f'ids_with_metrics_fin_mix{fold+1}_test.csv', index=False)

        print(f"Saved fold {fold+1} train and test splits.")

# Load the main CSV and create folds
#df = pd.read_csv('mag3redux.csv', encoding='ISO-8859-1')
#df = pd.read_csv('loadmanifestusemag3.csv', encoding='ISO-8859-1')#### new only
df = pd.read_csv('mag3withfunction.csv', encoding='ISO-8859-1')


create_folds(df)