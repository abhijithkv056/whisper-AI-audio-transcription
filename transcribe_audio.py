import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torchaudio
import os
import numpy as np

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
                "task": "transcribe",
                "return_timestamps": True
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
            
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Ensure single channel and correct shape
            waveform = waveform.squeeze()
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            
            # Convert to numpy and ensure correct shape
            audio_array = waveform.numpy()
            
            # Process the audio with proper input features
            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            
            # Convert input features to numpy array
            input_features = inputs["input_features"].cpu().numpy()
            
            # Use the pipeline for transcription with chunking
            result = self.pipe(
                input_features,
                chunk_length_s=30,  # Process in 30-second chunks
                stride_length_s=5   # 5-second overlap between chunks
            )
            
            # Combine all chunks into a single text
            full_text = " ".join([chunk["text"] for chunk in result["chunks"]])
            return full_text
            
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
    directory_path = "call_101"
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