import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

#Function to generate patches from an image
def generate_patches(image, patch_size):
    patches = []
    image_height, image_width = image.shape[:2]
    patch_height, patch_width = patch_size

    #Calculate number of patches in rows and columns
    num_rows = image_height // patch_height
    num_cols = image_width // patch_width

    #Generate patches
    for i in range(num_rows):
        for j in range(num_cols):
            y = i * patch_height
            x = j * patch_width
            patch = image[y:y+patch_height, x:x+patch_width]
            patches.append(patch)

    return patches

try:
    #Path to the folder containing images
    image_folder = "/content/drive/My Drive/Cropped Images/Set 01"

    #Path to save the patches
    patch_folder = "/content/drive/My Drive/Cropped Images/Patches"

    #Patch size
    patch_size = (500, 500)  

    #Create the patch folder if it doesn't exist
    if not os.path.exists(patch_folder):
        os.makedirs(patch_folder)

    #Load the data from a CSV file
    data = pd.read_csv("/content/drive/My Drive/Cropped Images/Set 01/DataWSA.csv")

    #Extract the field names from the IDs
    data['Field'] = data['ID'].str[:3]

    #Perform a train-test split while maintaining field-wise separation
    train_fields, test_fields = train_test_split(data['Field'].unique(), test_size=0.2, random_state=42)

    #Split the data based on train-test fields
    train_data = data[data['Field'].isin(train_fields)]
    test_data = data[data['Field'].isin(test_fields)]

    #Loop through all images in the image folder
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):  # Adjust file extensions as needed
            #Load the image
            image_path = os.path.join(image_folder, filename)
            image = np.array(Image.open(image_path))

            #Generate patches from the image
            patches = generate_patches(image, patch_size)

            #Save the patches
            for i, patch in enumerate(patches):
                patch_filename = os.path.splitext(filename)[0] + f"_patch_{i}.png"
                patch_path = os.path.join(patch_folder, patch_filename)
                Image.fromarray(patch).save(patch_path)

    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)

    #Print the number of samples in each split
    print("Number of samples in the training set:", len(train_data))
    print("Number of samples in the testing set:", len(test_data))

except Exception as e:
    print("Error occured during patching:", e)
