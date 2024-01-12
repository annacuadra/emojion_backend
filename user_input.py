import re
import joblib
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np

# Load the trained model and vectorizer from the file
model_components = joblib.load('OVA_Linear_model.pkl')
SVM_model, vectorizer, tfidf_transformer = model_components

# Define the same preprocessing functions and data structures used during training
stopword = pd.read_csv('datasets\stopwords_tl.csv')
stopwords_set = set(stopword['stopword'])

stemmer = pd.read_csv('datasets\stem_tl.csv')
word_to_stem = dict(zip(stemmer['word'], stemmer['stem']))

replace_patterns = {
    re.compile(r"\bngayo\'y\b"): 'ngayon ay',
    re.compile(r"\bhangga\'t\b"): 'hanggang',
    re.compile(r"\b\'?y\b"): ' ay',
    re.compile(r"\b\'?t\b"): ' at',
    re.compile(r"\b\'?yan\b"): 'iyan',
    re.compile(r"\b\'?yo\b"): 'iyo',
    re.compile(r"\b\'?yon\b"): 'iyon',
    re.compile(r"\b\'?yun\b"): 'iyun',
    re.compile(r"\b\'?pagkat\b"): 'sapagkat',
    re.compile(r"\b\'?di\b"): 'hindi',
    re.compile(r"\b\'?kaw\b"): "ikaw",
    re.compile(r"\b\'?to\b"): 'ito',
    re.compile(r"\b\'?wag\b"): 'huwag',
    re.compile(r"\bgano\'n\b"): 'ganoon'
}

foul_words = {
    'gago','gaga', 'puta', 'pakyu','pakshet','buang','walanghiya ','piste','lintik',
    'putangina','tarantado','punyeta','bwisit','kupal','hinyupak', 'tanga', 'tangina','bobo','boba','putragis', 'syet'
}

class_names = {
    1: 'fear',
    2: 'anger',
    3: 'joy',
    4: 'sadness',
    5: 'disgust',
    6: 'surprise'
}
def data_preprocess(text, replace_patterns, word_to_stem, stopwords_set):
    text = text.lower()

    for pattern, replacement in replace_patterns.items():
        text = pattern.sub(replacement, text)

    text = re.sub("[^a-zA-Z0-9\s?!]", '', text)
    tokens = word_tokenize(text)
    text = ' '.join([word_to_stem.get(word, word) for word in tokens if word.lower() not in stopwords_set])

    return text
# Allow user input
user_input = input("Enter a text: ")
user_input_processed = data_preprocess(user_input, replace_patterns, word_to_stem, stopwords_set)

# Transform the preprocessed input using the loaded vectorizer and tfidf_transformer
user_input_vectorized = vectorizer.transform([user_input_processed])
user_input_tfidf = tfidf_transformer.transform(user_input_vectorized)

# Use decision_function to obtain decision values for each class
decision_values = SVM_model.decision_function(user_input_tfidf)[0]

# Apply the softmax function to convert decision values into probabilities
exp_values = np.exp(decision_values - np.max(decision_values))  # Subtracting the max for numerical stability
probabilities = exp_values / exp_values.sum(axis=0, keepdims=True)

# Create a dictionary to store emotion probabilities with class names
emotion_probabilities_dict = {class_names[i+1]: probability * 100 for i, probability in enumerate(probabilities)}

# Set the probability to 0 for classes not present in the user input
for emotion in class_names.values():
    if emotion not in emotion_probabilities_dict:
        emotion_probabilities_dict[emotion] = 0.0

# Display the result
print()
for emotion, percentage in emotion_probabilities_dict.items():
    print(f"{emotion}: {percentage:.2f}%")
# Find the emotion with the highest probability
max_emotion = max(emotion_probabilities_dict, key=emotion_probabilities_dict.get)

# Display the result
print(f"\nThe predicted emotion for the input text is: {max_emotion}")
