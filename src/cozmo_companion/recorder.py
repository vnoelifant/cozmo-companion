import wave
from array import array

import pyaudio


class Recorder:
    """Recorder class to capture audio with silence detection and a maximum recording duration."""

    # Constants defining the audio properties and thresholds
    FORMAT = pyaudio.paInt16
    RATE = 44100
    CHUNK_SIZE = 1024
    THRESHOLD = 500
    CHANNELS = 1
    SILENCE_THRESHOLD = 100  # Stop after consecutive silent chunks

    def __init__(self, audio_file, record_seconds=5):
        """
        Initialize the recorder with target audio file and recording duration.

        Parameters:
        - audio_file (str): Path to save the recorded audio.
        - record_seconds (int): Maximum duration to record audio in seconds.
        - is_recording (bool): Flag if recording is in progress
        """
        self.audio_file = audio_file
        self.record_seconds = record_seconds
        self.is_recording = True

    def _is_audio_loud(self, data_chunk):
        """Check if the audio data is above the threshold.

        Parameters:
        - data_chunk (array): Array of audio data samples.

        Returns:
        - bool: True if audio is above the threshold, False otherwise.
        """
        return max(data_chunk) >= self.THRESHOLD

    def _is_prolonged_silence(self, silent_chunks):
        """Check if the number of silent chunks exceeds the silence threshold.

        Parameters:
        - silent_chunks (int): Number of consecutive silent chunks.

        Returns:
        - bool: True if prolonged silence is detected, False otherwise.
        """
        return silent_chunks >= self.SILENCE_THRESHOLD

    def record(self):
        """Record audio for a specified duration and save to a file."""
        p = pyaudio.PyAudio()
        frames = []

        try:
            # Open a stream for audio recording
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                output=True,
                frames_per_buffer=self.CHUNK_SIZE,
            )
            # Record audio in chunks
            self._record_audio_chunks(stream, frames)
            print("Recording Complete")
            # Save the recorded audio to a file
            self._save_audio_to_file(p.get_sample_size(self.FORMAT), frames)

        except Exception as e:
            print(f"Error while recording: {e}")
        finally:
            # Ensure stream is properly closed after recording
            stream.stop_stream()
            stream.close()
            p.terminate()

    def _record_audio_chunks(self, stream, frames):
        """Record audio in chunks for the specified duration or until prolonged silence.

        Parameters:
        - stream (PyAudio Stream): Active audio stream for recording.
        - frames (list): List to store audio frames.
        """
        silent_chunks = 0
        for _ in range(0, int(self.RATE / self.CHUNK_SIZE * self.record_seconds)):
            if not self.is_recording:
                break
            data = stream.read(self.CHUNK_SIZE)
            data_chunk = array("h", data)
            # Check if the audio chunk is loud enough
            if self._is_audio_loud(data_chunk):
                print("Something Said")
                frames.append(data)
                silent_chunks = 0
            else:
                print("Nothing Said")
                # If silent, keep track of consecutive silent chunks
                silent_chunks += 1
                if self._is_prolonged_silence(silent_chunks):
                    print("Prolonged silence detected. Stopping recording.")
                    self.is_recording = False

    def _save_audio_to_file(self, sample_width, frames):
        """Save recorded audio data to a file.

        Parameters:
        - sample_width (int): Sample width (number of bytes) of the recorded audio.
        - frames (list): List of recorded audio frames.
        """
        with wave.open(self.audio_file, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(sample_width)
            wf.setframerate(self.RATE)
            wf.writeframes(b"".join(frames))
        print("Saving speech audio to file complete")
