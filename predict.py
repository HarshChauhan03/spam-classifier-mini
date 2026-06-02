import pickle

# Load Model
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# User Input
message = input("Enter a message: ")

# Transform
message_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vector)

if prediction[0] == 1:
    print("🚨 Spam Message")
else:
    print("✅ Ham (Normal Message)")