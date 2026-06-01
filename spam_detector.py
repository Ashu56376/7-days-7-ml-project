import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

print("Loading dataset...")

df = pd.read_csv(
    url,
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("\nDataset Loaded Successfully!")
print("Total Messages:", len(df))

print("\nSpam vs Ham Distribution:")
print(df["label"].value_counts())

# Features and Labels
X = df["message"]
y = df["label"]

# Convert text into numerical features
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Top Spam Words
spam_messages = df[df["label"] == "spam"]["message"]

spam_vectorizer = CountVectorizer(stop_words="english")
spam_matrix = spam_vectorizer.fit_transform(spam_messages)

word_counts = np.array(spam_matrix.sum(axis=0)).flatten()
words = spam_vectorizer.get_feature_names_out()

top_words = sorted(
    zip(words, word_counts),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\nTop 10 Spam Words:")
for word, count in top_words:
    print(f"{word} : {count}")

# Custom Message Testing
while True:

    user_message = input("\nEnter SMS Message: ")

    message_vector = vectorizer.transform([user_message])

    prediction = model.predict(message_vector)[0]

    print("\nPrediction:", prediction.upper())

    if prediction == "spam":
        print("⚠ This message is SPAM")
    else:
        print("✓ This message is HAM")

    choice = input("\nCheck another message? (y/n): ")

    if choice.lower() != "y":
        break

print("\nProgram Finished Successfully!")