import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#Function to load and preprocess images
def load_images(data, patches_folder):
    images = []
    for patch_name in data['Patch_Name']:
        img_path = os.path.join(patches_folder, patch_name)
        img = load_img(img_path, target_size=(224, 224))
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

#Load and preprocess images for training and validation
X_train = load_images(train_data, train_patches_folder)
X_val = load_images(val_data, val_patches_folder)

#Extract WSA values
y_train = train_data['WSA'].values
y_val = val_data['WSA'].values

#Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

#Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

#Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

#Evaluate the model
loss, mae = model.evaluate(X_val, y_val)
print("Validation Loss:", loss)
print("Validation MAE:", mae)

import matplotlib.pyplot as plt

X_val = load_images(val_data,val_patches_folder)
predictions = model.predict(X_val) 

actual_values = val_data['WSA'].values 

mse = np.mean((predictions - actual_values) ** 2) 
mae = np.mean(np.abs(predictions - actual_values)) 

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

#Plot actual vs predicted values
plt.scatter(actual_values, predictions)
plt.xlabel('Actual WSA')
plt.ylabel('Predicted WSA')
plt.title('Actual vs Predicted WSA')
plt.show()

plt.scatter(actual_values, predictions, c=['blue', 'red'], label=['Actual', 'Predicted'])
plt.xlabel('Actual WSA')
plt.ylabel('Predicted WSA')
plt.title('Actual vs Predicted WSA')
plt.legend()
plt.show()


# Calculate and print evaluation metrics
mse = np.mean((predictions - actual_values) ** 2)
mae = np.mean(np.abs(predictions - actual_values))
