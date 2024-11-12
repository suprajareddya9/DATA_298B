import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain import PromptTemplate
import pandas as pd
import shelve
import uuid
from dotenv import load_dotenv

load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="Hospital Readmission Prediction System", layout="wide")

# Fixed header
st.markdown("""
    <style>
    .header {
        position: fixed;
        top: 5;
        left: 10;
        width: 100%;
        background-color: white;
        padding: 10px 0;
        box-shadow: 0 4px 2px -2px gray;
        z-index: 1000;
    }
    .main-container {
        margin-top: 60px;  /* Adjust this based on the height of the header */
    }
    </style>
    <div class="header">
        <h1 style="text-align:center;">Hospital Readmission Prediction System</h1>
        <p style="text-align:center;">This app uses a RAG system to predict readmission based on selected models.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar model selection
with st.sidebar:
    model_choice = st.radio("Choose a model:", ["Llama3_RAG", "BioMistral_RAG", "Mistral_RAG"])
   
# Functions for Mistral model setup
# Load the CSV file
transformed_train_df = pd.read_csv("drive/transformed_data.csv")
text_content = " ".join(transformed_train_df["question_prompt_answer"].astype(str))

@st.cache_resource
def prepare_vector_store(text_content):
    # Split text into chunks
    chunk_size = 1000
    chunk_overlap = 20
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_text(text_content)
    
    # Class to hold text document chunks with unique ids
    class TextDocument:
        def __init__(self, page_content):
            self.page_content = page_content
            self.metadata = {}
            self.id = str(uuid.uuid4())  # Assign a unique ID to each document
    
    # Create document objects from text chunks
    documents = [TextDocument(page_content=chunk) for chunk in text_chunks]
    
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS index in memory
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    
    print("Created the vector store in memory.")
    
    return vector_store

vector_store = prepare_vector_store(text_content)

@st.cache_resource
def load_llm():
    llm = LlamaCpp(
        streaming=True,
        model_path="drive/mistral-7b-instruct-v0.1.Q5_K_S.gguf",
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4096
    )
    return llm
    
@st.cache_resource
def create_qa_chain(_vector_store):
    llm = load_llm()
    retriever = _vector_store.as_retriever(search_kwargs={"k": 2})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa

mistral_qa_chain = create_qa_chain(vector_store)

# Chat history management
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar for chat history management
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Main container for chat
with st.container():
    # Display chat messages
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Main chat interface
    if query := st.chat_input("Enter your query"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        # Process user query and get response from the selected model
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            
            # Log start time to measure response delay
            import time
            start_time = time.time()
            st.write(f"started processing")
            # Model selection based on the toggle
            if model_choice == "Llama3_RAG":
                st.write(f"started processing llama")
                response = llama_qa_chain.run(query)
            elif model_choice == "BioMistral_RAG":
                st.write(f"started processing bio")
                response = biomistral_rag_chain.invoke(query)
            elif model_choice == "Mistral_RAG":
                st.write(f"started processing Mistral")
                response = mistral_qa_chain.run(query)
            else:
                response = "Please select a model to proceed"

            # Log end time and print to help debug
            end_time = time.time()
            st.write(f"Model response time: {end_time - start_time:.2f} seconds")
            
            # Display response in the chat
            response_placeholder.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Save chat history after each interaction
    save_chat_history(st.session_state.messages)
