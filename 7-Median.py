#Grouping by Image ID and calculating the median predicted WSA value
val_data['Predicted_WSA'] = model.predict(val_generator).flatten()
median_predicted_values = val_data.groupby('Image ID')['Predicted_WSA'].median().values

#Plot actual vs. median predicted values
plt.scatter(actual_values, median_predicted_values, label='Predictions')
plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], color='red', label='Perfect Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Median Predicted Values')
plt.title('Actual vs. Median Predicted Values')
plt.legend()
plt.show()
