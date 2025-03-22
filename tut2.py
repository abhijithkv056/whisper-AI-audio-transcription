from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import torch
import os

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if ChromaDB exists, if not create it
if not os.path.exists("chroma_db"):
    print("Creating ChromaDB...")
    # Load your documents
    documents = []
    for file in os.listdir("call_101"):
        if file.endswith(".txt"):
            file_path = os.path.join("call_101", file)
            try:
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading file {file}: {str(e)}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # Create and save ChromaDB
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vectorstore.persist()
    print("ChromaDB created and saved!")
else:
    print("Loading existing ChromaDB...")

# Load the vectorstore
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever()

def generate_text(prompt, chat_history, model_name="mistralai/Mistral-7B-Instruct-v0.2", max_length=200):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    # Retrieve relevant context
    docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Construct augmented prompt with chat history
    history = "\n".join([f"User: {h[0]}\nChatbot: {h[1]}" for h in chat_history])
    full_prompt = f"{history}\nContext: {context}\n\nUser: {prompt}\nChatbot:" 
    
    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    # Generate text
    output = model.generate(**inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    
    # Decode and return result
    return tokenizer.decode(output[0], skip_special_tokens=True)

def chatbot():
    print("Chatbot with RAG and chat history is ready! Type 'exit' to end the conversation.")
    chat_history = []
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        
        response = generate_text(user_input, chat_history)
        chat_history.append((user_input, response))
        
        print("Chatbot:", response)

if __name__ == "__main__":
    chatbot()