import pandas as pd
import numpy as np
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

#Function to load and preprocess images
def load_images(data, patches_folder):
    images = []
    for patch_name in data['Patch_Name']:
        img_path = os.path.join(patches_folder, patch_name)
        img = load_img(img_path, target_size=(224, 224))  # Adjust target_size as needed
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

#Load patch CSV files
train_patch_csv = 'D:/Cropped Images/train_patches.csv'
val_patch_csv = 'D:/Cropped Images/val_patches.csv'

train_data = pd.read_csv(train_patch_csv)
val_data = pd.read_csv(val_patch_csv)

#Path to patches folders
train_patches_folder = 'D:/Cropped Images/Train_Patches'
val_patches_folder = 'D:/Cropped Images/Val_Patches'

#Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,  # Increased rotation range
    width_shift_range=0.3,  # Increased width shift range
    height_shift_range=0.3,  # Increased height shift range
    shear_range=0.3,  # Increased shear range
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    directory=train_patches_folder,
    x_col='Patch_Name',
    y_col='WSA',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

val_generator = val_datagen.flow_from_dataframe(
    val_data,
    directory=val_patches_folder,
    x_col='Patch_Name',
    y_col='WSA',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

#Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

#Create new model on top of the pre-trained base model
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

#Compile the model with a smaller learning rate
model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error', metrics=['mae'])

#Train the model with more epochs
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

#Evaluate the model
loss, mae = model.evaluate(val_generator)
print("Validation Loss:", loss)
print("Validation MAE:", mae)

#Generate predictions
X_val = load_images(val_data, val_patches_folder)
predictions = model.predict(X_val)

#Obtain the actual WSA values from the validation data 
actual_values = val_data['WSA'].values 

#Calculate evaluation metrics
mse = np.mean((predictions - actual_values) ** 2)
mae = np.mean(np.abs(predictions - actual_values))

#Print the evaluation metrics 
print("Mean Squared Error (MSE):", mse) 
print("Mean Absolute Error (MAE):", mae) 

#Plot actual vs predicted values 
plt.scatter(actual_values, predictions) 
plt.xlabel('Actual WSA') 
plt.ylabel('Predicted WSA') 
plt.title('Actual vs Predicted WSA') 
plt.show()

#Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#Plot training and validation MAE
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Training and Validation MAE')
plt.legend()
plt.show()

plt.scatter(actual_values, predictions, label='Actual vs Predicted')
plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], color='red', label='Diagonal Line')
plt.xlabel('Actual WSA')
plt.ylabel('Predicted WSA')
plt.title('Actual vs Predicted WSA')
plt.legend()
plt.show()
