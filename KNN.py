from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load Iris dataset
irisData = load_iris()

print(irisData.data)

# Input features
x = irisData.data

# Target/output
y = irisData.target

print(y)

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42
)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=7)

# Train the model
knn.fit(x_train, y_train)

# Predict test data
print(knn.predict(x_test))