import io
import json
import logging
import os
import base64
import gc
import tempfile
import uuid
import chromadb
from gtts import gTTS
from IPython.display import display, HTML
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate, VectorStoreIndex, ServiceContext, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pandas as pd
import streamlit as st
from PIL import Image
from ollama import generate
import csv
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# Initialize CSV file for logging
csv_file_path = "chat_logs.csv"
csv_exists = os.path.exists(csv_file_path)

if not csv_exists:
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["User Query", "Assistant Response"])

# Function to log user query and assistant response to CSV
def log_to_csv(user_query, assistant_response):
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([user_query, assistant_response])

# Initialize Streamlit session
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_file(file):
    file_extension = os.path.splitext(file.name)[1].lower()

    if file_extension == ".pdf":
        display_pdf(file)
    elif file_extension == ".txt":
        display_text(file)
    elif file_extension == ".json":
        display_json(file)
    elif file_extension == ".csv":
        display_csv(file)
    elif file_extension in [".jpg", ".jpeg", ".png", ".gif"]:
        display_image(file)
    elif file_extension in [".py", ".js", ".java", ".cpp", ".c", ".html", ".css", ".asm"]:
        display_code(file)  # Updated to include .asm
    else:
        st.error(f"Unsupported file format: {file_extension}")

def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%">
                     </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_text(file):
    st.markdown("### Text File Preview")
    text = file.read().decode("utf-8")
    st.text(text)

def display_json(file):
    st.markdown("### JSON File Preview")
    json_data = file.read().decode("utf-8")
    st.json(json.loads(json_data))

def display_csv(file):
    st.markdown("### CSV File Preview")
    csv_data = file.read().decode("utf-8")
    st.write(pd.read_csv(io.StringIO(csv_data)))

def display_image(file):
    st.markdown("### Image Preview")
    image_data = file.read()
    st.image(image_data, use_column_width=True)
    process_image(image_data)

def display_code(file):
    st.markdown("### Code File Preview")
    code = file.read().decode("utf-8")
    language = file.name.split('.')[-1] 
    st.code(code, language=language)

def process_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    with io.BytesIO() as buffer:
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

    full_response = ''
    for response in generate(model='llava:13b-v1.6',
                             prompt='describe this image and make sure to include anything notable about it (include text you see in the image):',
                             images=[image_bytes],
                             stream=True):
        full_response += response['response']

    st.markdown("### Image Description")
    st.markdown(full_response)

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    _, temp_path = tempfile.mkstemp(suffix=".mp3")
    tts.save(temp_path)
    return temp_path

# Function to initialize the index using ChromaDB
# Function to initialize the index without ChromaDB
def init_index(embed_model):
    reader = SimpleDirectoryReader(input_dir="./docs/", recursive=True)
    documents = reader.load_data()

    logging.info("Index creating with `%d` documents", len(documents))

    # No ChromaDB client and collection setup

    # Create a default storage context without using ChromaDB
    storage_context = StorageContext.from_defaults()

    # Initialize the index using the provided embed model and storage context
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

    return index


# Unique key for the clear button based on session ID
clear_button_key = f"clear_button_{session_id}"

with st.sidebar:
    selected_model = st.selectbox(
        "Select your LLM:",
        ("Phi", "Llama3", "mistral"),
        index=0,
        key='selected_model'
    )

    st.header(f"Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your file", type=["pdf", "txt", "json", "csv", "jpg", "jpeg", "png", "gif", "py", "js", "java", "cpp", "c", "html", "css", "asm"])

    if uploaded_file:
        try:
            file_key = f"{session_id}-{uploaded_file.name}"

            # Check if the model has changed or the cache needs refreshing
            if 'current_model' not in st.session_state or st.session_state.current_model != selected_model:
                st.session_state.current_model = selected_model
                # Clear cached data relevant to the previous model
                st.session_state.file_cache.pop(file_key, None)  # Remove cached data for the old model if exists
                st.experimental_rerun()  # Optionally rerun to refresh the setup with the new model

            if st.session_state.current_model == "Llama3":
                llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
            elif st.session_state.current_model == "Phi":
                llm = Ollama(model="phi", request_timeout=120.0)
            elif st.session_state.current_model == "mistral":
                llm = Ollama(model="mistral", request_timeout=120.0)    

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    # Update to handle different file types
                    if os.path.exists(temp_dir):
                        loader = SimpleDirectoryReader(
                            input_dir=temp_dir,
                            required_exts=[".pdf", ".txt", ".json", ".csv", ".jpg", ".jpeg", ".png", ".gif", ".py", ".js", ".java", ".cpp", ".c", ".html", ".css", ".asm"],
                            recursive=True
                        )
                    else:
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    docs = loader.load_data()

                    # Setup embedding model (if needed)
                    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                    Settings.embed_model = embed_model

                    # Create index based on file content
                    index = init_index(embed_model)  # Use ChromaDB-based initialization

                    # Create query engine
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)

                    # Customize prompt template (if needed)
                    qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above, I want you to think step by step to answer the query in a crisp manner. In case you don't know the answer, say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
                    
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                # Inform the user that the file is processed and Display the file uploaded
                st.success("Ready to Chat!")
                display_file(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with your Docs! 📄")

with col2:
    st.button("Clear ↺", on_click=reset_chat, key=clear_button_key)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Log user query and assistant response to CSV
    log_to_csv(prompt, full_response)

    # Generate speech from the assistant's response
    speech_path = text_to_speech(full_response)
    
    # Read the speech file and encode to base64 for HTML audio tag
    with open(speech_path, "rb") as speech_file:
        speech_data = speech_file.read()
        speech_base64 = base64.b64encode(speech_data).decode('utf-8')

    # Display audio player in the middle of the response
    audio_html = f'<audio controls style="display: block; margin: 10px auto;"><source src="data:audio/mp3;base64,{speech_base64}" type="audio/mpeg"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)
