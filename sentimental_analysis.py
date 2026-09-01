import nltk
from nltk.corpus import movie_reviews
import random

# Download movie_reviews dataset if not available
try:
    movie_reviews.categories()
except LookupError:
    nltk.download('movie_reviews')

# Create documents with words and their category
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((movie_reviews.words(fileid), category))

# Shuffle the documents
random.shuffle(documents)

# Get all words from the movie reviews
all_words = nltk.FreqDist(
    word.lower() for word in movie_reviews.words()
)

# Select the 2000 most common words
word_features = list(all_words)[:2000]

# Function to create features for each document
def document_features(document):
    document_words = set(document)
    features = {}

    for word in word_features:
        features['contains({})'.format(word)] = (
            word in document_words
        )

    return features

# Create feature sets
featuresets = [
    (document_features(document), category)
    for (document, category) in documents
]

# Split data into training and testing sets
train_set = featuresets[100:]
test_set = featuresets[:100]

# Train Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Calculate accuracy
accuracy = nltk.classify.accuracy(classifier, test_set)
print("Accuracy:", accuracy)

# Display most informative features
print("\nMost Informative Features:")
classifier.show_most_informative_features(5)
