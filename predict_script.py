import joblib

translate_mood = ['joy','love','anger','fear','surprise']
model = joblib.load('mood_model.pkl')## outputed from the predict script

while True:
    msg = input("Enter a message (or 'quit'): ")
    if msg.lower() == "quit":
        break
    mood = model.predict([msg])[0]


    print(f"Predicted mood: {translate_mood[mood]}\n")
