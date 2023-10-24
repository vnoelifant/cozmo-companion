import typer

from .assistant import VoiceAssistant

app = typer.Typer()


@app.command()
def converse(user_speech_filename: str):
    assistant = VoiceAssistant()
    try:
        assistant.converse(user_speech_filename)
    except KeyboardInterrupt:
        print("\nClosing via keyboard interrupt.")
