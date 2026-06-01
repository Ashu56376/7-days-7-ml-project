import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================
# 1. LOAD & TRAIN MODEL
# =====================
print("Loading dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

X = df['message']
y = df['label']

# TF-IDF (better than CountVectorizer)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Logistic Regression (better accuracy)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# =====================
# 2. GRAPHS
# =====================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Spam Detector - Analysis', fontsize=16, fontweight='bold')

# Graph 1 - Spam vs Ham
df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['green','red'], edgecolor='black')
axes[0].set_title('Spam vs Ham Distribution')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')
axes[0].tick_params(rotation=0)

# Graph 2 - Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
axes[1].set_title('Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

# Graph 3 - Top Spam Words
spam_messages = df[df['label'] == 'spam']['message']
from sklearn.feature_extraction.text import CountVectorizer as CV
spam_vec = CV(stop_words='english')
spam_matrix = spam_vec.fit_transform(spam_messages)
word_counts = np.array(spam_matrix.sum(axis=0)).flatten()
words = spam_vec.get_feature_names_out()
top_words = sorted(zip(words, word_counts), key=lambda x: x[1], reverse=True)[:10]
w, c = zip(*top_words)
axes[2].barh(w, c, color='tomato', edgecolor='black')
axes[2].set_title('Top 10 Spam Words')
axes[2].set_xlabel('Count')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('spam_analysis.png')
plt.show()
print("Graph saved as spam_analysis.png!")

# =====================
# 3. GUI APP
# =====================
def predict_spam():
    msg = entry.get("1.0", tk.END).strip()
    if not msg:
        messagebox.showwarning("Warning", "Koi message likho!")
        return
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    spam_prob = round(max(prob) * 100, 2)
    if pred == 'spam':
        result_label.config(text=f"⚠ SPAM! ({spam_prob}% confident)", fg='red')
    else:
        result_label.config(text=f"✓ HAM - Safe! ({spam_prob}% confident)", fg='green')

# Build GUI
root = tk.Tk()
root.title("🚫 Spam Detector")
root.geometry("500x400")
root.configure(bg='#1e1e2e')

tk.Label(root, text="🚫 Spam Detector", font=('Arial', 20, 'bold'),
         bg='#1e1e2e', fg='white').pack(pady=20)

tk.Label(root, text="Message yahan likho:", font=('Arial', 12),
         bg='#1e1e2e', fg='#cdd6f4').pack()

entry = tk.Text(root, height=6, width=50, font=('Arial', 11),
                bg='#313244', fg='white', insertbackground='white')
entry.pack(pady=10, padx=20)

tk.Button(root, text="Check Spam", font=('Arial', 13, 'bold'),
          bg='#89b4fa', fg='#1e1e2e', command=predict_spam,
          padx=20, pady=8).pack(pady=10)

result_label = tk.Label(root, text="", font=('Arial', 14, 'bold'),
                        bg='#1e1e2e', fg='white')
result_label.pack(pady=10)

tk.Label(root, text=f"Model Accuracy: {accuracy*100:.2f}%", font=('Arial', 10),
         bg='#1e1e2e', fg='#6c7086').pack(side='bottom', pady=10)

print("\nGUI App starting...")
root.mainloop()