import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import messagebox

# =====================
# 1. DATASET
# =====================
data = {
    'review': [
        "This movie was absolutely amazing and wonderful!",
        "I loved every moment of this film, brilliant acting!",
        "Outstanding performance, best movie of the year!",
        "Fantastic plot and great cinematography!",
        "Incredible story, highly recommend to everyone!",
        "This was a terrible movie, complete waste of time",
        "Worst film I have ever seen, very disappointing",
        "Boring and dull, could not finish watching it",
        "Awful acting and poor storyline, do not watch",
        "Horrible experience, totally ruined my evening",
        "It was okay, nothing special about it",
        "Average movie, some good parts some bad",
        "Decent film but could have been much better",
        "Not great not terrible, just okay overall",
        "Mediocre at best, expected much more from it",
        "Loved the action scenes but story was weak",
        "Great visuals but disappointing ending overall",
        "Amazing soundtrack but average plot overall",
        "Good performances but slow pacing throughout",
        "Excellent direction but weak character development",
    ],
    'sentiment': [
        'positive', 'positive', 'positive', 'positive', 'positive',
        'negative', 'negative', 'negative', 'negative', 'negative',
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        'mixed', 'mixed', 'mixed', 'mixed', 'mixed'
    ]
}

df = pd.DataFrame(data)
print("Dataset loaded!")
print(df['sentiment'].value_counts())

# TextBlob Analysis
df['polarity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

print("\nTextBlob Analysis Done!")
print(df[['review', 'polarity', 'subjectivity']].head())

# =====================
# 2. ML MODEL
# =====================
# Binary classification - positive vs negative
df_binary = df[df['sentiment'].isin(['positive', 'negative'])].copy()

X = df_binary['review']
y = df_binary['sentiment']

vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# =====================
# 3. GRAPHS
# =====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sentiment Analysis — Dashboard', fontsize=16, fontweight='bold')

# Graph 1 - Sentiment Distribution
df['sentiment'].value_counts().plot(
    kind='bar', ax=axes[0,0],
    color=['green','red','blue','orange'], edgecolor='black'
)
axes[0,0].set_title('Sentiment Distribution')
axes[0,0].tick_params(rotation=0)

# Graph 2 - Polarity Distribution
axes[0,1].hist(df['polarity'], bins=10, color='steelblue', edgecolor='black')
axes[0,1].set_title('Polarity Distribution')
axes[0,1].set_xlabel('Polarity (-1 to 1)')
axes[0,1].axvline(x=0, color='red', linestyle='--', label='Neutral')
axes[0,1].legend()

# Graph 3 - Polarity vs Subjectivity
colors = {'positive':'green','negative':'red','neutral':'blue','mixed':'orange'}
for sentiment in df['sentiment'].unique():
    mask = df['sentiment'] == sentiment
    axes[1,0].scatter(
        df[mask]['polarity'],
        df[mask]['subjectivity'],
        label=sentiment,
        color=colors[sentiment],
        s=100, alpha=0.7
    )
axes[1,0].set_title('Polarity vs Subjectivity')
axes[1,0].set_xlabel('Polarity')
axes[1,0].set_ylabel('Subjectivity')
axes[1,0].legend()
axes[1,0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Graph 4 - Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1],
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])
axes[1,1].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('sentiment_analysis.png')
plt.show()
print("Graph saved!")

# =====================
# 4. GUI APP
# =====================
def analyze():
    text = entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Kuch likho!")
        return

    # TextBlob Analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        sentiment = "POSITIVE 😊"
        color = "green"
    elif polarity < -0.1:
        sentiment = "NEGATIVE 😞"
        color = "red"
    else:
        sentiment = "NEUTRAL 😐"
        color = "blue"

    result_label.config(
        text=f"Sentiment: {sentiment}\nPolarity: {polarity:.2f} | Subjectivity: {subjectivity:.2f}",
        fg=color
    )

root = tk.Tk()
root.title("😊 Sentiment Analyzer")
root.geometry("550x420")
root.configure(bg='#1e1e2e')

tk.Label(root, text="😊 Sentiment Analyzer",
         font=('Arial', 20, 'bold'), bg='#1e1e2e', fg='white').pack(pady=20)

tk.Label(root, text="Review ya text yahan likho:",
         font=('Arial', 12), bg='#1e1e2e', fg='#cdd6f4').pack()

entry = tk.Text(root, height=6, width=55, font=('Arial', 11),
                bg='#313244', fg='white', insertbackground='white')
entry.pack(pady=10, padx=20)

tk.Button(root, text="Analyze Sentiment",
          font=('Arial', 13, 'bold'), bg='#89b4fa', fg='#1e1e2e',
          command=analyze, padx=20, pady=8).pack(pady=10)

result_label = tk.Label(root, text="", font=('Arial', 13, 'bold'),
                        bg='#1e1e2e', fg='white', justify='center')
result_label.pack(pady=10)

tk.Label(root, text=f"Model Accuracy: {accuracy*100:.2f}%",
         font=('Arial', 10), bg='#1e1e2e', fg='#6c7086').pack(side='bottom', pady=10)

print("\nGUI starting...")
root.mainloop()