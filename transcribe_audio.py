import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torchaudio
import os
import numpy as np
import librosa

class AudioTranscriber:
    def __init__(self, model_id="openai/whisper-small"):
        """
        Initialize the AudioTranscriber with a specific model.
        
        Args:
            model_id (str): The ID of the Whisper model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = model_id
        
        # Initialize processor and model
        print("Initializing model and processor...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(self.device)
        
        # Create pipeline with specific configuration
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            generate_kwargs={
                "language": "en",
                "task": "transcribe"
            }
        )
        print(f"Model initialized on {self.device}")

    def transcribe_file(self, audio_path):
        """
        Transcribe a single audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: The transcribed text
        """
        try:
            print(f"Processing: {audio_path}")
            
            # Load audio using librosa
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Use the pipeline for transcription
            result = self.pipe(
                audio_array,
                chunk_length_s=30,  # Process in 30-second chunks
                stride_length_s=5   # 5-second overlap between chunks
            )
            
            return result["text"]
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

    def transcribe_directory(self, directory_path):
        """
        Transcribe all audio files in a directory.
        
        Args:
            directory_path (str): Path to the directory containing audio files
            
        Returns:
            dict: Dictionary containing transcriptions for each audio file
        """
        transcriptions = {}
        
        # Process each audio file in the directory
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
    # Example usage
    transcriber = AudioTranscriber()
    
    # Transcribe all files in a directory
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