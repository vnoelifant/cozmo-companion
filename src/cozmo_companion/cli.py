import json
import os
import sys
import traceback
from pathlib import Path
from typing import Final

import typer

from .dialogue_cozmo import Dialogue
from .recorder_cozmo import Recorder

app = typer.Typer()

EXIT_CONDITION: Final[str] = "exit"

@app.command()
def converse(filename: str):
    Path.cwd() / "wav_output" / f"{filename}.wav"
    curr_dir = Path.cwd()
    speech_out = curr_dir / "wav_output" / f"{filename}.wav"

    my_convo = Dialogue(speech_out)
    my_convo.get_cozmo_response("Hello! I will echo back what you say!")

    while True:
        try:
            print("starting recording process")
            recorder = Recorder(speech_out)
            print("Please say something nice into the microphone\n")
            recorder.save_to_file()
            print("Transcribing audio....\n")

            try:
                speech_text = my_convo.transcribe_audio()
                my_convo.get_cozmo_response(speech_text)
                if speech_text in EXIT_CONDITION:
                    print("Exiting program...")
                    break

            except:
                traceback.print_exc()
                print("error in speech detection")
                my_convo.get_cozmo_response(
                    "Sorry I couldn't understand you. Please repeat"
                )

        except KeyboardInterrupt:
            print("closing via keyboard interrupt")
            sys.exit(0)

