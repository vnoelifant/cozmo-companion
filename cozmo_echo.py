import json
import sys
import traceback
from dialogue_cozmo import Dialogue
from recorder_cozmo import Recorder


def main():

    
    my_convo = Dialogue('speech.wav')
    my_convo.get_cozmo_response("Hello! I will echo back what you say!")

    try:
        print("starting recording process")
        recorder = Recorder("speech.wav")
        print("Please say something nice into the microphone\n")
        recorder.save_to_file()
        print("Transcribing audio....\n")

        try:
            speech_text = my_convo.transcribe_audio()
            my_convo.get_cozmo_response(speech_text)

        except:
            traceback.print_exc()
            print("error in speech detection")
            my_convo.get_cozmo_response("Sorry I couldn't understand you")
            sys.exit(0)

    except KeyboardInterrupt:
        print("closing via keyboard interrupt")
        sys.exit(0)
       
if __name__ == '__main__':
    main()
