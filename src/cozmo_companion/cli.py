import typer

from .assistant import VoiceAssistant

app = typer.Typer()


@app.command()
def converse():
    assistant = VoiceAssistant()
    try:
        assistant.start_session()
    except KeyboardInterrupt:
        print("\nClosing via keyboard interrupt.")
