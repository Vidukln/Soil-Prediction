#Read the DataWSA file
data_wsa = pd.read_csv("/content/drive/My Drive/Cropped Images/Set 01/DataWSA.csv", delimiter=',')

#Read the Patch_Info.csv file
patch_info = pd.read_csv("/content/drive/My Drive/Cropped Images/Patch_Info.csv")

#Remove '.png' from 'Image' column in Patch_Info
patch_info['Image'] = patch_info['Image'].str.replace('.png', '')

#Merge the two dataframes 
merged_data = pd.merge(patch_info, data_wsa, left_on='Image', right_on='ID', how='left')

merged_data.drop(columns=['ID'], inplace=True)

#Save the merged data to a new CSV file
merged_data.to_csv("/content/drive/My Drive/Cropped Images/Merged_Patch_Info.csv", index=False)

print(merged_data.head())

import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

#Read the merged data file
merged_data = pd.read_csv("/content/drive/My Drive/Cropped Images/Merged_Patch_Info.csv")

#Get unique image names
unique_images = merged_data['Image'].unique()

#Split unique image names into training and validation sets(fieldwise seperation)
train_images, val_images = train_test_split(unique_images, test_size=0.2, random_state=42)

#Filter the dataset based on the training and validation image sets
train_data = merged_data[merged_data['Image'].isin(train_images)]
val_data = merged_data[merged_data['Image'].isin(val_images)]

#Define paths
patch_folder = "/content/drive/My Drive/Cropped Images/Patches"
train_patch_folder = "/content/drive/My Drive/Cropped Images/Train_Patches"
val_patch_folder = "/content/drive/My Drive/Cropped Images/Val_Patches"

#Create folders if they don't exist
os.makedirs(train_patch_folder, exist_ok=True)
os.makedirs(val_patch_folder, exist_ok=True)

#Copy patches to train or validation folders
for index, row in train_data.iterrows():
    patch_path = os.path.join(patch_folder, row['Patch'])
    train_patch_path = os.path.join(train_patch_folder, row['Patch'])
    try:
        shutil.copy(patch_path, train_patch_path)
    except FileNotFoundError:
        print(f"File {patch_path} not found. Skipping...")

for index, row in val_data.iterrows():
    patch_path = os.path.join(patch_folder, row['Patch'])
    val_patch_path = os.path.join(val_patch_folder, row['Patch'])
    try:
        shutil.copy(patch_path, val_patch_path)
    except FileNotFoundError:
        print(f"File {patch_path} not found. Skipping...")
