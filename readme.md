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

- Python 3.10 (required for Streamlit Cloud compatibility)
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

3. Upgrade pip and setuptools:
```bash
pip install --upgrade pip setuptools wheel
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

## Local Development

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload an audio file using the file uploader

4. Click the "Transcribe Audio" button to start the transcription process

5. Wait for the processing to complete and view the transcription

## Streamlit Cloud Deployment

This application is configured for deployment on Streamlit Cloud:

1. Push your code to a GitHub repository
2. Connect your repository to Streamlit Cloud
3. The application will automatically deploy using the specified Python version (3.10)

### Deployment Requirements

- `runtime.txt` specifies Python 3.10
- `requirements.txt` contains pinned package versions for compatibility
- No problematic dependencies that require Python 3.13+

## Project Structure

```
audio-transcription-app/
├── main.py                 # Streamlit web application
├── transcribe_audio.py     # Audio transcription class
├── requirements.txt        # Project dependencies
├── runtime.txt            # Python version specification
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Dependencies

- torch==2.0.1
- transformers==4.30.2
- torchaudio==2.0.1
- streamlit==1.24.0
- numpy==1.24.3
- librosa==0.10.0
- tqdm==4.65.0
- soundfile==0.12.1
- python-dotenv==1.0.0
- requests==2.31.0

## Model Information

The application uses OpenAI's Whisper model for transcription. By default, it uses the "whisper-small" model, but you can modify the model size in the `AudioTranscriber` class initialization.

Available model options:
- whisper-tiny
- whisper-base
- whisper-small
- whisper-medium
- whisper-large
- whisper-large-v3

## Troubleshooting

If you encounter installation issues:

1. Make sure you're using Python 3.10 (required for Streamlit Cloud)
2. Try upgrading pip and setuptools:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```
3. If specific packages fail to install, try installing them individually
4. For CUDA-related issues, make sure you have the correct CUDA version installed for your PyTorch version

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
