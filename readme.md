# Audio Transcription App

A Streamlit-based web application that transcribes audio files using OpenAI's Whisper model. This application allows users to upload audio files and get their transcriptions instantly.

## Features

- 🎵 Support for multiple audio formats (MP3, WAV, M4A)
- 🚀 Powered by OpenAI's Whisper model
- 💻 Web interface built with Streamlit
- ⚡ Real-time transcription processing
- 🔄 Automatic audio resampling and format conversion
- 📝 Clean and user-friendly interface

## Prerequisites

Before running the application, make sure you have the following installed:

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- FFmpeg (for audio processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-transcription-app.git
cd audio-transcription-app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload an audio file using the file uploader

4. Click the "Transcribe Audio" button to start the transcription process

5. Wait for the processing to complete and view the transcription

## Project Structure

```
audio-transcription-app/
├── main.py                 # Streamlit web application
├── transcribe_audio.py     # Audio transcription class
├── requirements.txt        # Project dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Dependencies

- torch
- transformers
- torchaudio
- streamlit
- numpy
- librosa (optional, for additional audio processing)

## Model Information

The application uses OpenAI's Whisper model for transcription. By default, it uses the "whisper-small" model, but you can modify the model size in the `AudioTranscriber` class initialization.

Available model options:
- whisper-tiny
- whisper-base
- whisper-small
- whisper-medium
- whisper-large
- whisper-large-v3

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for the transformers library
- Streamlit for the web framework

## Support

If you encounter any issues or have questions, please open an issue in the GitHub repository.
