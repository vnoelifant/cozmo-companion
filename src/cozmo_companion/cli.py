import json
import os
import sys
import traceback

import typer
from decouple import config

from .assistant import VoiceAssistant

app = typer.Typer()


@app.command()
def converse(audio_filename: str):
    EXIT_CONDITION: str = "exit"

    assistant = VoiceAssistant()
    assistant.speak("Hello! Chat with GPT and I will speak it's responses!")

    while True:
        try:
            speech_text = assistant.listen(audio_filename)
            print(f"User Speech Text: {speech_text} \n")

            if EXIT_CONDITION in speech_text.strip().lower():
                print("User initiating end of Conversation...")
                assistant.speak("Goodbye!")
                print("Exiting program...")
                break

            gpt_response_msg = assistant.get_gpt_completion(speech_text)
            print(f"GPT Response Message: {gpt_response_msg} \n")

            assistant.speak(gpt_response_msg)

        except KeyboardInterrupt:
            print("Closing via keyboard interrupt")
            sys.exit(0)
