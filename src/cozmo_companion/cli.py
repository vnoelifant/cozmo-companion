import json
import os
import sys
import traceback

import openai
import typer

from decouple import config

from .dialogue_cozmo import Dialogue
from .recorder_cozmo import Recorder

app = typer.Typer()

openai.api_key = config("OPENAI_API_KEY")

EXIT_CONDITION: str = "exit"
GPT_MODEL: str = "gpt-3.5-turbo"
max_tokens: int = 1000
temperature: int = 1.2


def get_completion(prompt, model=GPT_MODEL):
    messages = [{"role": "user", "content": prompt}]

    gpt_response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )

    gpt_response_msg = gpt_response["choices"][0]["message"]["content"]

    return gpt_response_msg


@app.command()
def converse(filename: str):
    curr_dir = os.getcwd()
    speech_out = os.path.join(curr_dir, "wav_output", filename + ".wav")

    my_convo = Dialogue(speech_out)
    my_convo.get_cozmo_response(
        "Hello! Ask GPT anything and I will echo back what it says in response!"
    )

    while True:
        try:
            print("starting recording process")
            recorder = Recorder(speech_out)
            print("Please say something nice into the microphone\n")
            recorder.save_to_file()
            print("Transcribing audio....\n")

            speech_text = my_convo.transcribe_audio()

            gpt_response_msg = get_completion(speech_text)

            my_convo.get_cozmo_response(gpt_response_msg)

            if speech_text in EXIT_CONDITION:
                print("Exiting program...")
                break

        except KeyboardInterrupt:
            print("closing via keyboard interrupt")
            sys.exit(0)
