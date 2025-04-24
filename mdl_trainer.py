from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import joblib

# After collecting dataset😴
emotion_texts = {
    "happy": [
        "I feel fantastic today", "What a wonderful morning", "Everything is going right",
        "I'm so proud of myself", "This day is amazing", "Feeling blessed and joyful",
        "I love my life", "Nothing can ruin my mood today", "Smiling all day", "Sunshine and happiness",
        "I'm so excited for the weekend", "Feeling like a winner", "Just got great news",
        "Life is beautiful", "Feeling so great today", "So thankful today", "What a joyful day",
        "Dancing with happiness", "Heart full of joy", "I'm feeling amazing"
    ],
    "sad": [
        "I'm feeling really down today", "Tears won't stop", "Everything feels hopeless",
        "I miss the old days", "Nobody understands me", "I feel like crying",
        "Today is just too hard", "My heart is heavy", "I'm mentally exhausted", "This is painful",
        "All I want is peace", "I feel unwanted", "This dataset is too hard to create", "I can't pretend anymore",
        "Emptiness surrounds me", "I'm not okay", "Sadness won't go away", "Broken from inside",
        "I feel lost", "Why do I always feel this way?"
    ],
    "angry": [
        "I'm furious with everything", "Stop testing my patience", "This makes me so mad",
        "I hate how things turned out", "I can't tolerate this anymore", "My blood is boiling",
        "Don't talk to me right now", "I want to scream", "Why is this happening again",
        "I'm extremely irritated", "Everything is annoying", "This is pure injustice",
        "I'm done being nice", "So unfair!", "You've crossed the line", "I'm really pissed off",
        "I want to punch something", "Raging with anger", "This is unbearable", "He made me so angry"
    ],
    "fear": [
        "I'm terrified of what's next", "This place gives me chills", "I heard a strange noise",
        "That person is suspicious", "I feel uneasy", "Something doesn't feel right",
        "I'm scared to move", "I think someone is following me", "That story was terrifying",
        "I don't want to be alone", "My heart races in fear", "Fear has taken over me",
        "That's too scary to handle", "I'm hiding in my room", "I heard footsteps at night",
        "Everything feels haunted", "It's dark and quiet", "I don't feel safe", "That startled me", "I froze in fear"
    ],
    "surprise": [
        "You won't believe what happened!", "That shocked me", "I wasn't expecting this",
        "Unbelievable turn of events", "It came out of nowhere", "Wow, amazing!",
        "This is a huge surprise", "I didn't see that coming", "You're kidding me, right?",
        "No way that just happened", "My jaw dropped", "I'm in complete shock",
        "What a twist!", "I'm mind blown", "Really? That happened?",
        "Unexpected news today", "Totally caught off guard", "That was amazing to hear",
        "Such a plot twist!", "Whoa, that happened fast"
    ],
    "neutral": [
        "I went to the store", "Just doing daily chores", "It's a regular Thursday",
        "Having lunch now", "Nothing new to report", "I'm working from home",
        "It's fine, just normal", "Same schedule as always", "I'm typing a message",
        "No strong feelings today", "I brushed my teeth", "I had coffee this morning",
        "Reading a book", "Eating chips", "Answering emails",
        "The weather is average", "I checked the calendar", "I replied to a text",
        "Doing my tasks", "Walking to the room"
    ]
}

# enough of dataset - lets duplicate to create more from this 😉
emotion_texts_expanded = {emotion: (sentences * 2)[:40] for emotion, sentences in emotion_texts.items()}

#seperate labels and text inputs
texts_expanded, labels_expanded = [], []
for emotion, examples in emotion_texts_expanded.items():
    texts_expanded.extend(examples)
    labels_expanded.extend([emotion] * len(examples))

df_expanded = pd.DataFrame({'text': texts_expanded, 'emotion': labels_expanded})

# split the train and test dataset for further validation to predict models accuracy
X_train, X_test, y_train, y_test = train_test_split(
    df_expanded['text'], df_expanded['emotion'], test_size=0.2, random_state=42, stratify=df_expanded['emotion']
)

# Now lets combine the NLP+ML to create a powerful model🏋️‍♀️ 
model = make_pipeline(TfidfVectorizer(), KNeighborsClassifier(n_neighbors=1))
model.fit(X_train, y_train)

joblib.dump(model,"emotion_classification.pkl")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

print(accuracy, pd.DataFrame(report).transpose(), conf_matrix)
