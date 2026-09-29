import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load diabetes dataset
df = datasets.load_diabetes()

# Display feature names
print("Feature names:", df['feature_names'])

# Load data
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Display original shapes
print("Original X shape:", diabetes_X.shape)
print("Original y shape:", diabetes_y.shape)

# Use only the 3rd feature (BMI)
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Display new shape
print("New X shape:", diabetes_X.shape)

# Split data into training and testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression model
regr = linear_model.LinearRegression()

# Train the model
regr.fit(diabetes_X_train, diabetes_y_train)

# Predict the test data
diabetes_y_pred = regr.predict(diabetes_X_test)

# Display coefficients
print("Coefficients:", regr.coef_)

# Display Mean Squared Error
print("Mean squared error: %.2f" %
      mean_squared_error(diabetes_y_test, diabetes_y_pred))

# Display coefficient of determination
print("Coefficient of determination: %.2f" %
      r2_score(diabetes_y_test, diabetes_y_pred))

# Plot the results
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Linear Regression - Diabetes Dataset")

plt.show()