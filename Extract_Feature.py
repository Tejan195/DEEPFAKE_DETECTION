import os
import zipfile
import shutil
import cv2

def copy_images(image_folder, output_folder):
    """Copies .png images to an output folder."""
    os.makedirs(output_folder, exist_ok=True)
    
    images_found = False
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith('.png'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(output_folder, file)
                shutil.copy(src_path, dst_path)  # Copy image
                images_found = True
                print(f"Copied {file} to {output_folder}")
    
    if not images_found:
        print(f"No .png files found in {image_folder} or its subfolders.")
    else:
        print(f"Finished copying images to {output_folder}")

downloads_folder = os.path.expanduser("~/Downloads")
zip_filename = "faceapp.zip"
zip_path = os.path.join(downloads_folder, zip_filename)

if not os.path.exists(zip_path):
    raise FileNotFoundError(f"Could not find {zip_path}. Please check the file name and location.")

extracted_folder = "C:/Users/tejan/OneDrive/Desktop/New folder (2)/dataset"
os.makedirs(extracted_folder, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)
print(f"Extracted {zip_filename} to {extracted_folder}")

image_folder = extracted_folder
output_base_folder = "C:/Users/tejan/OneDrive/Desktop/New folder (2)/output_images"
copy_images(image_folder, output_base_folder)

output_zip = "C:/Users/tejan/Downloads/Extracted_Facedata"
if os.listdir(output_base_folder):
    shutil.make_archive(output_zip, 'zip', output_base_folder)
    print(f"Created {output_zip}.zip")
else:
    print(f"Skipping zip creation: {output_base_folder} is empty.")
