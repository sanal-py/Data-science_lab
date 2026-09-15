from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

weather = [
    'sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy',
    'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast',
    'overcast', 'rainy'
]

temp = [
    'hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool',
    'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'
]

play = [
    'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes',
    'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no'
]

# Label Encoding
le = preprocessing.LabelEncoder()

weather_encoded = le.fit_transform(weather)
print("Weather:", weather_encoded)

temp_encoded = le.fit_transform(temp)
label = le.fit_transform(play)

print("Temp:", temp_encoded)
print("Play:", label)

# Creating feature list
features = list(zip(weather_encoded, temp_encoded))

print("Features:")
print(features)

# Create Gaussian Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(features, label)

# Prediction
predicted = model.predict([[0, 2]])

# Convert predicted number back to label
predicted_label = le.inverse_transform(predicted)[0]

print("Predicted Value:", predicted)

print(f"Play: {'Yes' if predicted_label == 'yes' else 'No'}")