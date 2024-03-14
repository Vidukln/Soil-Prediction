from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt

#Read the CSV file 
file_path = '/content/drive/My Drive/Cropped Images/Set 01/DataWSA.csv'
df = pd.read_csv(file_path, sep=',')

print(df.head())

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#Histogram for WSA
axs[0].hist(df['WSA'], bins=10, color='skyblue', edgecolor='black')
axs[0].set_title('Water Stable Aggregator (WSA)')
axs[0].set_xlabel('WSA')
axs[0].set_ylabel('Frequency')

#Histogram for UA
axs[1].hist(df['UA'], bins=10, color='salmon', edgecolor='black')
axs[1].set_title('Unstable Aggregator (UA)')
axs[1].set_xlabel('UA')
axs[1].set_ylabel('Frequency')

#Histogram for SA
axs[2].hist(df['SA'], bins=10, color='lightgreen', edgecolor='black')
axs[2].set_title('Stable Aggregator (SA)')
axs[2].set_xlabel('SA')
axs[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

statistics = df.describe()
print(statistics)
