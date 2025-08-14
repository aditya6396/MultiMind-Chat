import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create a new collection or connect to an existing one in ChromaDB
adhi = "document_embeddings"
if adhi not in chroma_client.list_collections():
    collection = chroma_client.create_collection(name=adhi)
else:
    collection = chroma_client.get_collection(name=adhi)

# Load a sentence transformer model to generate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up the text generation pipeline with the specified model
model_id = "unsloth/Llama-3.2-1B-Instruct"
generator = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
    max_length=1000,  # Limit the response length
    do_sample=True,
    top_k=50,
    top_p=0.9,
    temperature=1.0,
    repetition_penalty=1.0,
    num_return_sequences=1,
)

# Streamlit UI setup
st.title("AI Chatbot with Document Retrieval")
st.write("Chat with the AI model, and it will retrieve information from the document database.")

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to embed and store documents
def embed_and_store_document(doc_text, doc_id):
    embedding = embedding_model.encode([doc_text])
    # Store embedding in ChromaDB with metadata
    collection.add(embeddings=embedding.tolist(), metadatas=[{"doc_id": doc_id}], ids=[doc_id])
    st.write(f"Document '{doc_id}' stored in ChromaDB.")

# Function to retrieve the most relevant document based on the user query
def retrieve_relevant_document(query):
    query_embedding = embedding_model.encode([query])
    # Search ChromaDB for the most similar document
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=1)
    if results["ids"]:
        # Retrieve document metadata and content
        doc_id = results["ids"][0][0]
        metadata = results["metadatas"][0][0]
        return metadata["doc_id"]
    else:
        return None

# Function to generate response from the model with retrieved document
def get_model_response(prompt, context=""):
    full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
    responses = generator(full_prompt)
    return responses[0]["generated_text"]

# Document Upload Section
st.sidebar.header("Document Management")
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt"])

# Store uploaded document in ChromaDB
if uploaded_file is not None:
    doc_text = uploaded_file.read().decode("utf-8")
    doc_id = uploaded_file.name
    embed_and_store_document(doc_text, doc_id)

# Create an input box for the user to type in their message
user_input = st.text_input("You: ", placeholder="Type a message here...")

# Display the conversation history
for chat in st.session_state.chat_history:
    st.write(chat)

# Check if the user has entered a message
if user_input:
    # Retrieve the most relevant document for the given input
    relevant_doc = retrieve_relevant_document(user_input)
    document_context = f"Document ID: {relevant_doc}" if relevant_doc else "No relevant document found."

    # Generate response using the document context
    chatbot_response = get_model_response(user_input, context=document_context)

    # Display user input and model response in the chat history
    st.session_state.chat_history.append(f"You: {user_input}")
    st.session_state.chat_history.append(f"Chatbot: {chatbot_response}")

    # Update the app with new chat history
    st.write(f"You: {user_input}")
    st.write(f"Chatbot: {chatbot_response}")
