# Audio Transcription and Analysis System 🎙️

A powerful web application that transcribes audio files and provides structured conversation analysis using AI. Built with Streamlit, OpenAI's Whisper for transcription, and LangChain for intelligent conversation parsing. Perfect for interviews, meetings, podcasts, and any spoken content that needs to be transcribed and analyzed.

## 🌟 Features

- **Audio Transcription**: Supports multiple audio formats (MP3, WAV, M4A)
- **AI-Powered Analysis**: Processes transcripts to extract structured conversations
- **Real-time Processing**: Instant transcription and analysis
- **User-Friendly Interface**: Clean Streamlit web interface
- **Smart Conversation Structuring**: Automatically organizes multi-speaker dialogues
- **Error Handling**: Robust error management for file processing
- **Flexible Output**: Both raw transcription and structured conversation format

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Speech-to-Text**: OpenAI Whisper
- **Language Processing**: LangChain + OpenAI GPT
- **Audio Processing**: librosa, soundfile
- **Python Version**: 3.10

## 📋 Prerequisites

1. Python 3.10
2. OpenAI API key
3. FFmpeg installed on your system
4. CUDA-capable GPU (recommended for faster processing)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/audio-transcription-system.git
   cd audio-transcription-system
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## 💻 Usage

1. **Start the application**
   ```bash
   streamlit run main.py
   ```

2. **Using the Web Interface**
   - Navigate to http://localhost:8501
   - Upload an audio file (MP3, WAV, or M4A)
   - Click "Transcribe Audio"
   - View both raw transcription and structured conversation analysis

## 📁 Project Structure

```
audio-transcription-system/
├── main.py                 # Streamlit web application
├── transcribe_audio.py     # Audio transcription class
├── requirements.txt        # Project dependencies
├── runtime.txt            # Python version specification
├── .env                   # Environment variables (create this)
└── README.md              # Project documentation
```

## 🔧 Core Components

### 1. Audio Transcription
- Uses OpenAI's Whisper model for accurate speech recognition
- Handles multiple audio formats
- Provides high-quality text conversion
- Supports multiple languages

### 2. Conversation Analysis
- Processes raw transcripts using LangChain
- Structures conversations into clear speaker segments
- Identifies and labels different speakers
- Maintains conversation flow and context

### 3. Web Interface
- Clean, intuitive Streamlit interface
- Real-time processing feedback
- Progress indicators and status updates
- Error handling and user notifications
- Easy file upload and management

## 🔍 API Reference

### Main Components

1. **AudioTranscriber Class**
   ```python
   transcriber = AudioTranscriber()
   result = transcriber.transcribe_file(file_path)
   ```

2. **Chat Analysis Chain**
   ```python
   chat_chain = get_chat_chain()
   analysis = chat_chain.invoke({"input": transcription})
   ```

## ⚙️ Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key

### Model Options
- Whisper model size can be configured in `transcribe_audio.py`
- Available options: tiny, base, small, medium, large
- Choose based on your accuracy needs and computational resources

## 🚨 Troubleshooting

Common issues and solutions:

1. **FFmpeg Error**
   - Install FFmpeg using your system's package manager
   - Add FFmpeg to system PATH
   - Verify installation: `ffmpeg -version`

2. **CUDA Issues**
   - Ensure CUDA toolkit is installed
   - Check PyTorch CUDA compatibility
   - Verify GPU recognition: `nvidia-smi`

3. **API Key Error**
   - Verify `.env` file exists
   - Check API key is valid and properly formatted
   - Test API key authentication

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for Whisper model
- LangChain for conversation processing
- Streamlit for the web framework
- All contributors and users of this project

## 📞 Support

For support, please:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Contact the maintainers

---
Made with ❤️ for making audio content accessible and analyzable
