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

    This function sets up and runs an instance of the VoiceAssistant. It handles user interactions
    using an asynchronous event loop. If the user interrupts the program (e.g., using Ctrl+C),
    the function ensures a graceful shutdown.
    """
    # Create an instance of the VoiceAssistant to manage user interactions
    assistant = VoiceAssistant()

    # Create a new asyncio event loop to manage the assistant's asynchronous operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)  # Set the newly created loop as the current event loop

    try:
        try:
            # Run the assistant's session until it completes or is interrupted
            loop.run_until_complete(assistant.start_session())
        except KeyboardInterrupt:
            # Handle a user interrupt (Ctrl+C) by displaying a clean exit message
            print("\nClosing via keyboard interrupt.")
    finally:
        # Ensure the event loop is properly closed to prevent resource leaks
        loop.close()
