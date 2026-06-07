import nltk
import numpy as np
import random
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3
import speech_recognition as sr
import threading
from datetime import datetime

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# =====================
# 1. VOICE ENGINE
# =====================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

def listen():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5)
            return r.recognize_google(audio)
    except sr.WaitTimeoutError:
        return "timeout"
    except sr.UnknownValueError:
        return "unclear"
    except Exception:
        return "error"

# =====================
# 2. KNOWLEDGE BASE
# =====================
knowledge_base = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "good morning", "namaste", "sup"],
        "responses": ["Hello! How can I help? 😊", "Hi there!", "Namaste! 🙏"]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you", "take care", "exit"],
        "responses": ["Goodbye! 👋", "See you later!", "Take care! 😊"]
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "great", "awesome", "perfect"],
        "responses": ["You're welcome! 😊", "Happy to help!", "Anytime!"]
    },
    "ml_questions": {
        "patterns": ["machine learning", "explain ml", "what is ai", "deep learning", "neural network"],
        "responses": [
            "Machine Learning is AI where computers learn from data! 🤖",
            "Deep Learning uses neural networks with many layers! 🧠",
            "Neural Networks are inspired by the human brain!"
        ]
    },
    "python_questions": {
        "patterns": ["what is python", "python language", "why python", "python for ml"],
        "responses": [
            "Python is the most popular ML language! 🐍",
            "Python has NumPy, Pandas, Scikit-learn, TensorFlow — amazing libraries!"
        ]
    },
    "projects": {
        "patterns": ["projects", "7 days", "ml projects", "what have you built"],
        "responses": [
            "7 Days 7 ML Projects:\n1. House Price Prediction\n2. Spam Detector\n3. Flower Classifier\n4. Movie Recommender\n5. Sentiment Analysis\n6. Image Classifier\n7. AI Chatbot (me!) 🤖"
        ]
    },
    "name": {
        "patterns": ["your name", "who are you", "introduce yourself"],
        "responses": ["I am ML Bot — Day 7 of 7 Days 7 ML Projects! 🤖"]
    },
    "jokes": {
        "patterns": ["joke", "funny", "make me laugh"],
        "responses": [
            "Why do programmers prefer dark mode? Light attracts bugs! 😄",
            "Why did the ML model go to therapy? Loss function issues! 😂",
            "What do you call a singing neural network? Deep learning with pitch! 🎵"
        ]
    },
    "time": {
        "patterns": ["what time", "current time", "time please"],
        "responses": ["__TIME__"]
    },
    "help": {
        "patterns": ["help", "what can you do", "features"],
        "responses": ["I can:\n✅ Answer ML questions\n✅ Voice input/output 🎤\n✅ Remember conversation 🧠\n✅ Tell jokes 😄\nJust ask!"]
    },
    "default": {
        "patterns": [],
        "responses": [
            "Interesting! Can you rephrase that? 🤔",
            "Ask me about ML or Python!",
            "I need more training for that 😄"
        ]
    }
}

# =====================
# 3. NLP ENGINE
# =====================
all_patterns = []
all_labels = []
conversation_history = []

for intent, data in knowledge_base.items():
    for pattern in data['patterns']:
        all_patterns.append(pattern)
        all_labels.append(intent)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(all_patterns)

def get_response(user_input):
    user_lower = user_input.lower().strip()

    if any(p in user_lower for p in knowledge_base['time']['patterns']):
        return f"Current time is {datetime.now().strftime('%I:%M %p')} ⏰"

    if any(p in user_lower for p in knowledge_base['goodbye']['patterns']):
        return random.choice(knowledge_base['goodbye']['responses'])

    try:
        user_vec = vectorizer.transform([user_lower])
        similarities = cosine_similarity(user_vec, X)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score > 0.15:
            intent = all_labels[best_idx]
            return random.choice(knowledge_base[intent]['responses'])
        else:
            return random.choice(knowledge_base['default']['responses'])
    except:
        return random.choice(knowledge_base['default']['responses'])

# =====================
# 4. GUI — root PEHLE
# =====================
root = tk.Tk()
root.title("🤖 ML Chatbot — Day 7")
root.geometry("650x650")
root.configure(bg='#1e1e2e')

# BooleanVar root ke BAAD
voice_enabled = tk.BooleanVar(root)
voice_enabled.set(False)

def update_status(msg):
    status_label.config(text=msg)

def add_message(sender, message, tag):
    chat_area.config(state='normal')
    time_now = datetime.now().strftime("%H:%M")
    chat_area.insert(tk.END, f"[{time_now}] ", 'time')
    if sender == "You":
        chat_area.insert(tk.END, f"You: {message}\n", tag)
    else:
        chat_area.insert(tk.END, f"ML Bot: {message}\n\n", tag)
    chat_area.config(state='disabled')
    chat_area.see(tk.END)

def send_message(event=None):
    user_msg = entry.get().strip()
    if not user_msg:
        return
    entry.delete(0, tk.END)
    add_message("You", user_msg, 'user')
    conversation_history.append({'user': user_msg})
    response = get_response(user_msg)
    add_message("ML Bot", response, 'bot')
    if voice_enabled.get():
        speak(response)
    update_status("Ready! 💬")

def voice_input():
    def _listen():
        update_status("Listening... 🎤")
        voice_btn.config(state='disabled')
        text = listen()
        voice_btn.config(state='normal')
        if text in ["timeout", "unclear", "error"]:
            update_status("Try again! 🎤")
        else:
            entry.delete(0, tk.END)
            entry.insert(0, text)
            update_status(f"Heard: {text} ✅")
            send_message()
    threading.Thread(target=_listen, daemon=True).start()

def clear_chat():
    chat_area.config(state='normal')
    chat_area.delete(1.0, tk.END)
    chat_area.config(state='disabled')
    conversation_history.clear()
    add_message("ML Bot", "Chat cleared! How can I help? 😊", 'bot')

# Header
header = tk.Frame(root, bg='#313244', pady=10)
header.pack(fill='x')
tk.Label(header, text="🤖 ML Chatbot",
         font=('Arial', 22, 'bold'), bg='#313244', fg='white').pack()
tk.Label(header, text="Day 7 — Voice + Memory + NLP",
         font=('Arial', 10), bg='#313244', fg='#6c7086').pack()

# Status
stats_frame = tk.Frame(root, bg='#181825', pady=5)
stats_frame.pack(fill='x')
status_label = tk.Label(stats_frame, text="Ready to chat! 💬",
                        font=('Arial', 10), bg='#181825', fg='#a6e3a1')
status_label.pack(side='left', padx=15)

# Chat Area
chat_frame = tk.Frame(root, bg='#1e1e2e')
chat_frame.pack(fill='both', expand=True, padx=15, pady=10)
chat_area = scrolledtext.ScrolledText(
    chat_frame, wrap=tk.WORD, state='disabled',
    font=('Arial', 11), bg='#313244', fg='white',
    insertbackground='white', height=22, relief='flat', padx=10, pady=10
)
chat_area.pack(fill='both', expand=True)
chat_area.tag_config('user', foreground='#89b4fa', font=('Arial', 11, 'bold'))
chat_area.tag_config('bot', foreground='#a6e3a1', font=('Arial', 11))
chat_area.tag_config('time', foreground='#6c7086', font=('Arial', 9))

# Welcome
chat_area.config(state='normal')
chat_area.insert(tk.END, "ML Bot: Hello! I am ML Bot 🤖\n", 'bot')
chat_area.insert(tk.END, "ML Bot: Voice + Memory + NLP powered chatbot!\n", 'bot')
chat_area.insert(tk.END, "ML Bot: Ask me anything about ML or Python!\n\n", 'bot')
chat_area.config(state='disabled')

# Controls
ctrl_frame = tk.Frame(root, bg='#1e1e2e')
ctrl_frame.pack(fill='x', padx=15, pady=5)
tk.Checkbutton(ctrl_frame, text="🔊 Voice Reply",
               variable=voice_enabled, bg='#1e1e2e', fg='white',
               selectcolor='#313244', activebackground='#1e1e2e',
               activeforeground='white', font=('Arial', 10)).pack(side='left')
tk.Button(ctrl_frame, text="🗑️ Clear",
          font=('Arial', 10), bg='#f38ba8', fg='white',
          command=clear_chat, padx=10).pack(side='right')

# Input
input_frame = tk.Frame(root, bg='#1e1e2e', pady=10)
input_frame.pack(fill='x', padx=15)
entry = tk.Entry(input_frame, font=('Arial', 12),
                 bg='#313244', fg='white', insertbackground='white', relief='flat')
entry.pack(side='left', fill='x', expand=True, ipady=10, padx=(0, 8))
entry.bind('<Return>', send_message)
voice_btn = tk.Button(input_frame, text="🎤",
                      font=('Arial', 14), bg='#a6e3a1', fg='#1e1e2e',
                      command=voice_input, padx=10, pady=5)
voice_btn.pack(side='left', padx=(0, 8))
tk.Button(input_frame, text="Send 📤",
          font=('Arial', 12, 'bold'), bg='#89b4fa', fg='#1e1e2e',
          command=send_message, padx=15, pady=5).pack(side='left')

tk.Label(root, text="Press Enter to send | 🎤 for voice",
         font=('Arial', 9), bg='#1e1e2e', fg='#6c7086').pack(pady=5)

print("ML Chatbot starting...")
root.mainloop()