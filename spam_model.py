from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample Dataset
messages = [
    "Congratulations! You won a free iPhone",
    "Claim your prize now",
    "Get free recharge today",
    "Hello, how are you?",
    "Let's meet tomorrow",
    "Can you send the notes?"
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = Spam, 0 = Ham

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Train Model
model = MultinomialNB()
model.fit(X, labels)

# Save Model
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Spam Classifier Model Trained Successfully")