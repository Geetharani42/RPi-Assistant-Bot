import joblib
import pyttsx3
import speech_recognition as sr
import RPi.GPIO as GPIO

M1a, M1b, M2a, M2b = 25, 24, 23, 18

# Setup GPIO for controlling vehicle movement
GPIO.setmode(GPIO.BCM)
GPIO.setup([M1a, M1b, M2a, M2b], GPIO.OUT, initial=GPIO.LOW)

# Load trained model
model = joblib.load("emotion_classification.pkl")

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 125)
engine.setProperty('volume', 1.0)

# Responses by emotion
emotion_responses = {
    "happy": "You sound really happy today! Keep smiling!",
    "sad": "I can sense you're feeling sad. I'm here to cheer you up!",
    "angry": "You seem angry. Take a deep breath, it's going to be okay.",
    "fear": "You're feeling scared. Don't worry, you're not alone.",
    "surprise": "Wow! That sounds surprising!",
    "neutral": "Just a normal day, I see. Let's make it a good one!"
}

# Speech recognizer
recognizer = sr.Recognizer()

def forward():
    GPIO.output(M1a, GPIO.HIGH)
    GPIO.output(M1b, GPIO.LOW)
    GPIO.output(M2a, GPIO.HIGH)
    GPIO.output(M2b, GPIO.LOW)

def backward():
    GPIO.output(M1a, GPIO.LOW)
    GPIO.output(M1b, GPIO.HIGH)
    GPIO.output(M2a, GPIO.LOW)
    GPIO.output(M2b, GPIO.HIGH)

def left():
    GPIO.output(M1a, GPIO.HIGH)
    GPIO.output(M1b, GPIO.LOW)
    GPIO.output(M2a, GPIO.LOW)
    GPIO.output(M2b, GPIO.HIGH)

def right():
    GPIO.output(M1a, GPIO.LOW)
    GPIO.output(M1b, GPIO.HIGH)
    GPIO.output(M2a, GPIO.HIGH)
    GPIO.output(M2b, GPIO.LOW)

def stop():
    GPIO.output([M1a, M1b, M2a, M2b], GPIO.LOW)

def control_vehicle(command):
    stop()
    actions = {
        "forward": forward,
        "backward": backward,
        "left": left,
        "right": right,
        "stop": stop,
    }
    for keyword, action in actions.items():
        if keyword in command:
            print(f"Executing: {keyword}")
            action()
            return

def listen_and_detect_emotion():
    with sr.Microphone() as source:
        print("Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        if any(keyword in text for keyword in ["forward", "left", "right", "backward", "stop"]):
            control_vehicle(text)
        emotion = model.predict([text])[0]
        print("Detected emotion:", emotion)
        response = emotion_responses.get(emotion, "I'm not sure what you're feeling.")
        print("Response:", response)
        engine.say(response)
        engine.runAndWait()

    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

while True:
  listen_and_detect_emotion()
