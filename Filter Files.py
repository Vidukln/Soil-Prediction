import os
import shutil

train_source_dir = 'D:/Cropped Images/Train_Patches'
val_source_dir = 'D:/Cropped Images/Val_Patches'
train_dest_dir = 'D:/Cropped Images/Filtered_Train_Patches'
val_dest_dir = 'D:/Cropped Images/Filtered_Val_Patches'

filename_common_part = '_patch_'

image_names_to_keep = [
    '2.png', '6.png', '7.png', '8.png', '10.png', '11.png', '12.png', 
    '13.png', '14.png', '16.png', '17.png', '18.png', '22.png'
]

#Function to extract sample ID from filename
def extract_sample_id(filename):
    return filename[:4]

def filter_and_copy_images(source_dir, dest_dir, image_names):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    #Iterate through the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(tuple(image_names)):
                # Construct source and destination paths
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, file)
                # Copy the file to the destination directory
                shutil.copy(src_path, dest_path)

#Filter and copy images for train data
filter_and_copy_images(train_source_dir, train_dest_dir, image_names_to_keep)

#Filter and copy images for validation data
filter_and_copy_images(val_source_dir, val_dest_dir, image_names_to_keep)
