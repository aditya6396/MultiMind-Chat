import os
import base64
import gc
import tempfile
import uuid
import streamlit as st
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import csv

# Initialize session state if not already done
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.feedback = {}
    st.session_state.messages = []

session_id = st.session_state.id
client = None

# Function to reset the chat
def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    st.session_state.query_engine = None
    st.session_state.feedback = {}
    gc.collect()

# Function to display the PDF
def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%">
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to display code
def display_code(file_content, file_type):
    st.markdown(f"### {file_type.upper()} Code Preview")
    st.code(file_content, language=file_type)

# Function to get the conversation chain
def get_conversation_chain(vectorstore):
    llm = Ollama(
        model="mistral",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain

# Function to regenerate a response
def regenerate_response(idx, query):
    st.session_state.messages = st.session_state.messages[:idx+1]
    message_placeholder = st.empty()
    full_response = ""

    if "query_engine" in st.session_state and st.session_state.query_engine:
        query_engine = st.session_state.query_engine
        streaming_response = query_engine.query(query)

        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    else:
        full_response = "No document processed yet."
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.messages[idx]["regenerated_response"] = full_response

# Initialize CSV file for logging
csv_file_path = "/home/cpatwadityasharma/chat-with-multiple-PDFs-LLAMA2/chat_logs.csv"
csv_exists = os.path.exists(csv_file_path)

if not csv_exists:
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["User Query", "Assistant Response", "User Response", "Regenerated Response", "Feedback"])

# Function to log user query, assistant response, user response, regenerated response, and feedback to CSV
def log_to_csv(user_query, assistant_response, user_response, regenerated_response, feedback):
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([user_query, assistant_response, user_response, regenerated_response, feedback])

# Sidebar for model selection and file upload
with st.sidebar:
    selected_model = st.selectbox(
        "Select your LLM:",
        ("Phi", "Llama3", "mistral"),
        index=0,
        key='selected_model'
    )

    st.header("Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf`, `.py`, or `.asm` file", type=["pdf", "py", "asm"])

    if uploaded_file:
        try:
            file_key = f"{session_id}-{uploaded_file.name}"

            if 'current_model' not in st.session_state or st.session_state.current_model != selected_model:
                st.session_state.current_model = selected_model
                st.session_state.file_cache.pop(file_key, None)
                st.experimental_rerun()

            if st.session_state.current_model == "Llama3":
                llm = Ollama(model="llama3", request_timeout=120.0)
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

                # Initialize index to None
                index = None

                # Check file type and process accordingly
                if uploaded_file.type == "application/pdf":
                    loader = SimpleDirectoryReader(
                        input_dir=temp_dir,
                        required_exts=[".pdf"],
                        recursive=True
                    )
                    docs = loader.load_data()
                    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)
                elif uploaded_file.type == "text/x-python":
                    # Handle Python files
                    file_content = uploaded_file.read().decode("utf-8")
                    display_code(file_content, "python")
                    st.session_state.messages.append({"role": "assistant", "content": "Python code uploaded."})
                elif uploaded_file.type == "text/x-asm":
                    # Handle Assembly files
                    file_content = uploaded_file.read().decode("utf-8")
                    display_code(file_content, "asm")
                    st.session_state.messages.append({"role": "assistant", "content": "Assembly code uploaded."})
                else:
                    st.error('Unsupported file type.')
                    st.stop()

                # Only create the query engine if index is defined
                if index is not None:
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)

                    qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above I want you to think step by step to answer the query in a crisp manner, in case you don't know the answer say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
                    
                    st.session_state.file_cache[file_key] = query_engine
                    st.session_state.query_engine = query_engine  # Store in session state

                    st.success("Ready to Chat!")
                    display_pdf(uploaded_file)
                else:
                    st.error("No valid index was created.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

# Layout for chat and reset button
col1, col2 = st.columns([6, 1])

with col1:
    st.header("Chat with your Docs! 📄")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

# Display chat messages
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
        # Horizontally aligned feedback buttons
        col_feedback = st.columns([3])[0]
        with col_feedback:
            st.write("Feedback:")
            if st.button("👍", key=f"like_{idx}"):
                st.session_state.feedback[f"{idx}_feedback"] = "like"
            if st.button("👎", key=f"dislike_{idx}"):
                st.session_state.feedback[f"{idx}_feedback"] = "dislike"
            if st.button("♻️", key=f"regenerate_{idx}"):
                regenerate_response(idx, st.session_state.messages[idx-1]["content"])

        st.write(f"Current Feedback: {st.session_state.feedback.get(f'{idx}_feedback', '')}")

        # Log user query, assistant response, user response, regenerated response, and feedback to CSV
        log_to_csv(st.session_state.messages[idx-1]["content"], message["content"], st.session_state.messages[idx-1]["content"], message.get("regenerated_response", ""), st.session_state.feedback.get(f"{idx}_feedback", ""))

# Handle user input
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    message_placeholder = st.empty()
    full_response = ""

    if "query_engine" in st.session_state and st.session_state.query_engine:
        query_engine = st.session_state.query_engine
         
        streaming_response = query_engine.query(prompt)

        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    else:
        full_response = "No document processed yet."
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
