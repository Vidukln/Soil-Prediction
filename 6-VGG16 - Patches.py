import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

#Path to CSV files
train_patch_csv = 'D:/Cropped Images/trainlist_modified.csv'
val_patch_csv = 'D:/Cropped Images/vallist_modified.csv'

#Load training and validation data from CSV files
train_data = pd.read_csv(train_patch_csv)
val_data = pd.read_csv(val_patch_csv)

#Path to patches folders
train_patches_folder = 'D:/Cropped Images/Filtered_Train_Patches'
val_patches_folder = 'D:/Cropped Images/Filtered_Val_Patches'

#Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    directory=train_patches_folder,
    x_col='Filename',
    y_col='WSA',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

val_generator = val_datagen.flow_from_dataframe(
    val_data,
    directory=val_patches_folder,
    x_col='Filename',
    y_col='WSA',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

#Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Fine-tune the pre-trained model
for layer in base_model.layers[:-8]:
    layer.trainable = True

#Create new model on top of the pre-trained base model
model = Sequential([
    base_model,
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

#Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

#Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

#Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Plot training & validation MAE
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Get actual and predicted values
actual_values = np.concatenate([val_generator[i][1] for i in range(len(val_generator))])
predicted_values = model.predict(val_generator).flatten()

#Fit a regression line
regression_line = np.polyfit(actual_values, predicted_values, 1)

#Plot actual vs. predicted values
plt.scatter(actual_values, predicted_values, label='Predictions')
plt.plot(actual_values, np.polyval(regression_line, actual_values), color='green', label='Regression Line')
plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], color='red', label='Perfect Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

#Extracting the Image ID from the Filename column
val_data['Image ID'] = val_data['Filename'].str[:4]

#Grouping by Image ID and calculating the average predicted WSA value
val_data['Predicted_WSA'] = model.predict(val_generator).flatten()
average_predicted_values = val_data.groupby('Image ID')['Predicted_WSA'].mean().values

#Getting the unique Image IDs for plotting
unique_image_ids = val_data['Image ID'].unique()

#Plot actual vs. average predicted values
plt.scatter(actual_values, average_predicted_values, label='Predictions')
plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], color='red', label='Perfect Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Average Predicted Values')
plt.title('Actual vs. Average Predicted Values')
plt.legend()
plt.show()
