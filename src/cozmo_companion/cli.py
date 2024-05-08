import typer
import asyncio

# Importing the VoiceAssistant class from the assistant module
from .assistant import VoiceAssistant

# Initializing the Typer application for command-line interface
app = typer.Typer()


# Defining a command for the Typer application
@app.command()
def converse():
    """
    Initialize the voice assistant and start a conversation session.
    This function creates an instance of the VoiceAssistant and
    initiates its session. If interrupted with a keyboard command
    (like Ctrl+C), it provides a graceful exit message.
    """
    # Creating an instance of the VoiceAssistant
    assistant = VoiceAssistant()
    try:
        # Starting the session with the user
        asyncio.run(assistant.start_session())
    except KeyboardInterrupt:
        # Handling keyboard interrupt to gracefully exit the application
        print("\nClosing via keyboard interrupt.")
