import wave
from array import array
from struct import pack

import pyaudio


class Recorder:
    def __init__(self):
        self.THRESHOLD = 500
        self.CHUNK_SIZE = 1024
        self.FORMAT = pyaudio.paInt16
        self.RATE = 44100
        self.RECORD_SECONDS = 5
        self.silent_count = 0
        self.is_recording = True

    def record(self, audio_file):
        # create an instance of pyAudio
        p = pyaudio.PyAudio()
        # create a stream
        stream = p.open(
            format=self.FORMAT,
            channels=1,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK_SIZE,
        )

        frames = []

        while self.is_recording:
            """
            This operation is in charge of controlling how many chunks of the buffer the system reads before
            saving self.RECORD_SECONDS of data.

            For example, to record 5 seconds, we have to save 44,100 samples/second * 5 seconds = 220,500 samples.

            Because most audio systems work with chunks (also called frames or blocks), the system will need to read chunks
            of data. One chunk in this case is 1024 samples. Thus if each iteration (chunk) takes 1024 samples, the for
            loop will have to loop 220,500/1024 = 215 times.

            So basically the system reads (220,500/1024) = 215.33 chunks out of the buffer until it has saved the
            whole 5 seconds of audio data.
            """
            for i in range(0, int(self.RATE / self.CHUNK_SIZE * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK_SIZE)
                data_chunk = array("h", data)
                vol = max(data_chunk)
                if vol >= self.THRESHOLD:
                    print("Something Said")
                    frames.append(data)
                else:
                    print("Nothing Said")
            self.is_recording = False

        sample_width = p.get_sample_size(self.FORMAT)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Recording Complete")

        print("Saving speech audio to file")
        self.save_audio_to_file(sample_width, frames, audio_file)

    def save_audio_to_file(self, sample_width, frames, audio_file):
        with wave.open(audio_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(self.RATE)
            wf.writeframes(b"".join(frames))  # append frames recorded to file
