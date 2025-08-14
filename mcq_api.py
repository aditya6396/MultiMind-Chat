import os
import tempfile
import csv
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate, VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


app = FastAPI()


csv_file_path = "chat_history.csv"
csv_exists = os.path.exists(csv_file_path)

if not csv_exists:
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["User Query", "Assistant Response"])


def log_to_csv(user_query, assistant_response):
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([user_query, assistant_response])

ollama = Ollama(model="llama3", request_timeout=120.0)  


def initialize_query_engine(temp_dir):
    global query_engine
    
    loader = SimpleDirectoryReader(input_dir=temp_dir, required_exts=[".pdf", ".txt", ".json", ".csv", ".jpg", ".jpeg", ".png", ".gif", ".py", ".js", ".java", ".cpp", ".c", ".html", ".css"], recursive=True)
    docs = loader.load_data()
    
   
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
    Settings.embed_model = embed_model
    
    # Create index based on file content
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    
    # Create query engine
    Settings.llm = ollama
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
    
    
    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "You are AI assisant,I want you to think like a teacher to make the mcqs from the document on the given topic the mcqs should be unique and simple not repeated .All options should be in a one word only and give the output in correct json format\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

class FileUploadResponse(BaseModel):
    message: str

@app.post("/uploadfile/", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        # Log the received file
        print(f"File received: {file.filename}, content type: {file.content_type}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.filename)
            
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            # Log the file path where it is written
            print(f"File written to: {file_path}")
            
            initialize_query_engine(temp_dir)
        
        return {"message": "File uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    bot_reply: str

@app.post("/chatbot/", response_model=ChatResponse)
async def chatbot(chat_request: ChatRequest):
    if query_engine is None:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a document first.")
    
    user_query = chat_request.message
    try:
        streaming_response = query_engine.query(user_query)
        
        full_response = ""
        for chunk in streaming_response.response_gen:
            full_response += chunk
        
        # Log the conversation
        log_to_csv(user_query, full_response)
        
        return {"bot_reply": full_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)