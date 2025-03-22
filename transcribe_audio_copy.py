import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
import os
from tqdm import tqdm

def transcribe_audio(audio_path, output_file=None):
    try:
        # Set up device and model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Using device: {device}")
        
        model_id = "openai/whisper-large-v3"
        
        # Initialize processor and model
        print("Loading model and processor...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
        
        # Load and process the audio file
        print(f"Loading audio file: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Process the audio with attention mask
        print("Processing audio...")
        inputs = processor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt",
            return_attention_mask=True  # Explicitly request attention mask
        ).to(device)
        
        # Generate transcription with language specification
        print("Generating transcription...")
        generated_ids = model.generate(
            **inputs,
            language="en",  # Specify English language
            task="transcribe"  # Specify transcription task
        )
        
        # Decode the transcription
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Save to file if output path is provided
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcription)
            print(f"Transcription saved to: {output_file}")
        
        return transcription
        
    except FileNotFoundError:
        print(f"Error: Audio file '{audio_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    audio_path = "call_101.mp3"
    output_file = "transcription.txt"
    
    transcription = transcribe_audio(audio_path, output_file)
    if transcription:
        print("\nTranscription:")
        print(transcription) 