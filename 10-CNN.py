#Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

#Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#Load the CSV file containing soil data
data = pd.read_csv('/content/drive/My Drive/Cropped Images/filenames.csv')

#Extract field ID from filename
data['Field_ID'] = data['Filename'].str[:3]

#Group data by Field_ID
grouped_data = data.groupby('Field_ID')

#Randomly shuffle the groups
grouped_data = [group.sample(frac=1, random_state=42) for _, group in grouped_data]

#Split the groups into training and validation sets (80-20 split)
train_size = int(0.8 * len(grouped_data))
train_data = pd.concat(grouped_data[:train_size])
validation_data = pd.concat(grouped_data[train_size:])

#Split the data into training and testing sets
X_train_filenames = train_data['Image_Filenames']
y_train = train_data['WSA']
X_validation_filenames = validation_data['Image_Filenames']
y_validation = validation_data['WSA']

#Preprocess labels
scaler = MinMaxScaler()
y_train_normalized = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
y_validation_normalized = scaler.transform(np.array(y_validation).reshape(-1, 1))

img_height = 224
img_width = 224
batch_size = 32

#Augmentation parameters
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#Load and preprocess images for training with augmentation
train_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory='/content/drive/My Drive/Cropped Images/Set 01/',
    x_col='Image_Filenames',
    y_col='WSA',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='other'
)

#Load and preprocess images for validation
validation_generator = test_datagen.flow_from_dataframe(
    dataframe=validation_data,
    directory='/content/drive/My Drive/Cropped Images/Set 01/',
    x_col='Image_Filenames',
    y_col='WSA',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='other'
)

#Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping], 
    verbose=1 
)

#Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Predictions on validation data
predictions = model.predict(validation_generator)

#Denormalize predictions
predictions_denormalized = scaler.inverse_transform(predictions).flatten()

#Denormalize actual values
actual_values_denormalized = scaler.inverse_transform(np.array(y_validation_normalized)[:len(predictions_denormalized)].reshape(-1, 1)).flatten()

#Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(actual_values_denormalized, predictions_denormalized, color='blue', label='Predicted')
plt.plot([min(actual_values_denormalized), max(actual_values_denormalized)],
         [min(actual_values_denormalized), max(actual_values_denormalized)],
         color='red', label='Actual')
plt.title('Actual vs Predicted WSA')
plt.xlabel('Actual WSA')
plt.ylabel('Predicted WSA')
plt.legend()
plt.show()

#Plot actual values against differences
differences = predictions_denormalized - actual_values_denormalized
plt.figure(figsize=(10, 6))
plt.scatter(actual_values_denormalized, differences, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Relationship between Actual Values and Differences')
plt.xlabel('Actual WSA')
plt.ylabel('Difference (Predicted - Actual)')
plt.show()

#Calculate mean squared error
mse = np.square(predictions_denormalized - actual_values_denormalized).mean()

#Plot MSE distribution
plt.figure(figsize=(8, 6))
plt.hist(np.square(predictions_denormalized - actual_values_denormalized), bins=20, color='skyblue', edgecolor='black')
plt.title('Mean Squared Error Distribution')
plt.xlabel('Mean Squared Error')
plt.ylabel('Frequency')
plt.show()

print("Mean Squared Error:", mse)
