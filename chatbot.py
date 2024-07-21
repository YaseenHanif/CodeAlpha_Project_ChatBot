# -*- coding: utf-8 -*-

import nltk
import random
import string
import warnings
from tkinter import *
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# Load and preprocess the text data
f = open(r'C:\Users\Yaseen\Downloads\Act1Scene1.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw)  # Converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # Converts to list of words

# Preprocessing
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey","what is your name?")
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! you are talking to me","my name is Aneka"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Response generation
def response(user_response):
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        chatbot_response = chatbot_response + "I am sorry! I don't understand you"
    else:
        chatbot_response = chatbot_response + sent_tokens[idx]
    sent_tokens.remove(user_response)
    return chatbot_response

# GUI
def send():
    user_input = user_entry.get("1.0",'end-1c').strip()
    user_entry.delete("0.0", END)
    if user_input:
        chat_window.config(state=NORMAL)
        chat_window.insert(END, "You: " + user_input + '\n\n')
        chat_window.config(foreground="#442265", font=("Verdana", 12 ))

        if user_input.lower() in ['bye', 'thanks', 'thank you']:
            if user_input.lower() == 'bye':
                chat_window.insert(END, "Aneka: Bye! Have a great time!\n\n")
                chat_window.config(state=DISABLED)
            else:
                chat_window.insert(END, "Aneka: You're welcome!\n\n")
                chat_window.config(state=DISABLED)
        else:
            if greeting(user_input.lower()) is not None:
                chat_window.insert(END, "Aneka: " + greeting(user_input.lower()) + '\n\n')
            else:
                chat_window.insert(END, "Aneka: " + response(user_input.lower()) + '\n\n')

        chat_window.config(state=DISABLED)
        chat_window.yview(END)

# Creating the main window
root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

# Creating the chat window
chat_window = scrolledtext.ScrolledText(root, bd=1, bg="white", width=50, height=8, font=("Arial", 12), wrap=WORD)
chat_window.config(state=DISABLED)

# Placing the chat window
chat_window.place(x=6, y=6, height=385, width=370)

# Creating the entry box
user_entry = Text(root, bd=0, bg="white", width=29, height=5, font=("Arial", 12), wrap=WORD)

# Placing the entry box
user_entry.place(x=6, y=400, height=90, width=265)

# Creating the send button
send_button = Button(root, text="Send", command=send, bd=0, bg="#4CAF50", fg="white", width=12, height=5, font=("Arial", 12, "bold"))

# Placing the send button
send_button.place(x=280, y=400, height=90, width=100)

# Running the main loop
root.mainloop()
