from sklearn.feature_extraction.text import TfidfVectorizer
sentences = [
    "hello",
    "hi",
    "hey",
    "good morning",

    "bye",
    "goodbye",
    "see you",

    "thanks",
    "thank you",
    "thank you so much",

    "what is your name",
    "who are you"
]


labels = [
    "greeting",
    "greeting",
    "greeting",
    "greeting",

    "goodbye",
    "goodbye",
    "goodbye",

    "thanks",
    "thanks",
    "thanks",

    "name",
    "name"
]

vectorizer = TfidfVectorizer() 
# tf-idf tool? vectorizer create karti hia
X=vectorizer.fit_transform(sentences)
# ye hi sentences ko numerical representation mein convert karega

# ye line sentence ko number me convert karega
"""
Because ML model directly:

"hello"
"goodbye"

ko mathematical input ke roop mein use nahi kar sakta.

TF-IDF karega:

Text
 ↓
TF-IDF
 ↓
Numerical representation
"""

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X,labels)
"""
Yahan:

X      = TF-IDF numbers
labels = correct intent

Model seekhega:

hello → greeting
hi → greeting
bye → goodbye
thanks → thanks
"""
while True:
    message = input("You: ")
    if(message.lower()=="exit"):
        break
    user_vector=vectorizer.transform([message])
    intent=model.predict(user_vector)[0]
    print("Intent:", intent)    
