import os
import pandas as pd
import pydicom
from pathlib import Path
import argparse

def get_study_tags(tags_path: Path, get_shape: bool):
    tags = []
    dcm_files = tags_path.rglob("*.dcm")
    for dcm_file in dcm_files:
        try:
            dicom_tags = {}
            subject = dcm_file.relative_to(tags_path).parts[0]
            ds = pydicom.read_file(dcm_file)
            for e in ds.iterall():
                tag = e.tag
                try:
                    description = pydicom.datadict.dictionary_description(tag)
                    des_value = e.repval.replace('"', "")
                except KeyError:
                    # Tag is not present in the data dictionary
                    description = "Privatetag"  # Replace with the appropriate description
                    des_value = ''
                if description != "Pixel Data":
                    # Handling the tag if it's not "Pixel Data"
                    parent_directory = os.path.dirname(dcm_file)
                    dicom_tags["subject"] = subject
                    dicom_tags["file_name"] = dcm_file.parts[-1]
                    if get_shape:  # Only need shape for nuclear med DICOMS
                        img = ds.pixel_array
                        shape = img.shape
                        dicom_tags["shape"] = shape
                    dicom_tags["directory"] = parent_directory
                    dicom_tags[description] = des_value
            tags.append(dicom_tags)
        except ValueError as tags_error:
            print(f"Error processing {dcm_file}: {tags_error}")
    tags_df = pd.DataFrame(tags)
    tags_df = tags_df.drop_duplicates()
    return tags_df

def main():
    parser = argparse.ArgumentParser(description="Extract DICOM tags and save as a CSV.")
    parser.add_argument("tags_path", type=str, help="Path to the directory containing DICOM files.")
    parser.add_argument("output_path", type=str, help="Path to save the output CSV file.")
    parser.add_argument("csv_name", type=str, help="Name of the output CSV file.")
    parser.add_argument("--get_shape", action="store_true", help="Whether to include the shape of pixel data.")
    
    args = parser.parse_args()

    tags_path = Path(args.tags_path)
    output_path = Path(args.output_path)
    csv_name = args.csv_name

    if not tags_path.exists() or not tags_path.is_dir():
        print(f"Error: The provided tags path '{tags_path}' does not exist or is not a directory.")
        return

    output_csv = output_path / csv_name
    tags_df = get_study_tags(tags_path, args.get_shape)
    tags_df.to_csv(output_csv, index=False)
    print(f"Tags extracted and saved to {output_csv}")

if __name__ == "__main__":
    main()
