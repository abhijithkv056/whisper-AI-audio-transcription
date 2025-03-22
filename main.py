import streamlit as st
from transcribe_audio import AudioTranscriber
import tempfile
import os

def main():
    st.title("Audio Transcription App")
    st.write("Upload an audio file to get its transcription using Whisper AI")

    # Initialize the transcriber with caching
    @st.cache_resource
    def get_transcriber():
        return AudioTranscriber()

    transcriber = get_transcriber()

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a'])

    if uploaded_file is not None:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Add a button to trigger transcription
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing audio..."):
                try:
                    # Get transcription using the AudioTranscriber class
                    transcription = transcriber.transcribe_file(tmp_file_path)
                    
                    if transcription:
                        st.success("Transcription completed!")
                        st.write("### Transcription:")
                        st.write(transcription)
                    else:
                        st.error("Failed to transcribe the audio file.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    # Clean up the temporary file
                    os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
