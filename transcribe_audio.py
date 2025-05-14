import os
import openai
import streamlit as st

class AudioTranscriber:
    def __init__(self):
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        self.client = client

    def transcribe_file(self, audio_path):
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return None
        """
        Transcribe a single audio file using OpenAI Whisper API.

        Args:
            audio_path (str): Path to the audio file

        Returns:
            str: The transcribed text
        """
        try:
            print(f"Transcribing: {audio_path}")
            with open(audio_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript
        except Exception as e:
            print(f"Error transcribing {audio_path}: {str(e)}")
            return None

    def transcribe_directory(self, directory_path):
        """
        Transcribe all audio files in a directory using OpenAI Whisper API.

        Args:
            directory_path (str): Path to the directory containing audio files

        Returns:
            dict: Dictionary containing transcriptions for each audio file
        """
        transcriptions = {}

        for audio_file in os.listdir(directory_path):
            if audio_file.endswith(('.mp3', '.wav', '.m4a')):
                audio_path = os.path.join(directory_path, audio_file)
                transcription = self.transcribe_file(audio_path)

                if transcription:
                    transcriptions[audio_file] = transcription
                    print(f"Transcription for {audio_file}:")
                    print(transcription)
                    print("-" * 50)

        return transcriptions


def main():
    transcriber = AudioTranscriber()
    directory_path = "Audio files"

    if os.path.exists(directory_path):
        transcriptions = transcriber.transcribe_directory(directory_path)
        print("\nAll transcriptions:")
        for audio_file, text in transcriptions.items():
            print(f"\nFile: {audio_file}")
            print(f"Transcription: {text}")
            print("-" * 50)
    else:
        print(f"Directory {directory_path} not found")


if __name__ == "__main__":
    main()
