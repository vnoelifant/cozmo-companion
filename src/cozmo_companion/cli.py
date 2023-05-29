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
    assistant.speak(
        "Hello! Ask GPT anything and I will echo back what it says in response!"
    )

    while True:
        try:
            speech_text = assistant.listen(audio_filename)

            gpt_response_msg = assistant.get_gpt_completion(speech_text)

            assistant.speak(gpt_response_msg)

            if speech_text in EXIT_CONDITION:
                print("Exiting program...")
                break

        except KeyboardInterrupt:
            print("closing via keyboard interrupt")
            sys.exit(0)
