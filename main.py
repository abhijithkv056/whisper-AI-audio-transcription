import streamlit as st
from transcribe_audio import AudioTranscriber
import tempfile
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def main():
    # Load environment variables (for OpenAI API key)
    load_dotenv()
    
    st.title("Audio Transcription App")
    st.write("Upload an audio file to get its transcription using Whisper AI and chat analysis")

    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in Streamlit Cloud secrets.")
        st.info("To set the API key in Streamlit Cloud:\n1. Go to your app's dashboard\n2. Click on 'Secrets'\n3. Add your OpenAI API key as: OPENAI_API_KEY=your-api-key-here")
        return

    # Initialize the transcriber with caching
    @st.cache_resource
    def get_transcriber():
        return AudioTranscriber()

    # Initialize LangChain components with caching
    @st.cache_resource
    def get_chat_chain():
        try:
            llm = ChatOpenAI()
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Call transcription agent. 
                You are given call transcript and you need to extract the information from the transcript.
                You need to table the information in a structured format in the form of a chatbot.
                The information should be presented in following format:
                Person 1 chat: 
                Person 2 chat:
                Person 1 chat:
                Person 2 chat:
                ..."""),
                ("user", "{input}")
            ])
            output_parser = StrOutputParser()
            return prompt | llm | output_parser
        except Exception as e:
            st.error(f"Error initializing chat chain: {str(e)}")
            return None

    transcriber = get_transcriber()
    chat_chain = get_chat_chain()

    if chat_chain is None:
        return

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
                        
                        # Display raw transcription
                        st.write("### Raw Transcription:")
                        st.write(transcription)
                        
                        # Process transcription through chat chain
                        with st.spinner("Analyzing conversation..."):
                            chat_result = chat_chain.invoke({"input": transcription})
                            st.write("### Conversation Analysis:")
                            st.write(chat_result)
                    else:
                        st.error("Failed to transcribe the audio file.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    # Clean up the temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass

if __name__ == "__main__":
    main()
