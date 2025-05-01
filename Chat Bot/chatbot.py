# chatbot.py

from datetime import datetime


def chatbot():
    print("Chatbot: Hello! I am a simple chatbot. Type 'bye' to exit.")

    while True:
        user_input = input("You: ").lower().strip()

        if user_input in ['hi', 'hello', 'hey']:
            print("Chatbot: Hi there! How can I assist you today?")
        elif 'how are you' in user_input:
            print(
                "Chatbot: I'm just a bunch of code, but I'm functioning as expected!"
            )
        elif 'your name' in user_input:
            print("Chatbot: I'm CODSOFT Chatbot, your virtual assistant.")
        elif 'help' in user_input:
            print(
                "Chatbot: You can ask me about time, greetings, or just have a chat!"
            )
        elif 'time' in user_input:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"Chatbot: The current time is {now}.")
        elif user_input in ['bye', 'exit', 'goodbye']:
            print("Chatbot: Goodbye! Have a great day!")
            break
        else:
            print("Chatbot: I'm not sure how to respond to that.")


if __name__ == "__main__":
    chatbot()
