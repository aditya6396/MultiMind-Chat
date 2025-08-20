# MultiMind-Chat
## Overview
This Streamlit-based web application leverages the **Ollama** framework for end-to-end local development, enabling users from any domain (e.g., education, research, legal, finance, engineering) to upload and interact with various file types (PDFs, text, images, CSVs, JSON, code files) and query them using large language models (LLMs). It supports multimodal capabilities, including image analysis via the **LLaVA 13B model**, document indexing with **ChromaDB**, and text-to-speech with **gTTS**. User interactions are logged in a CSV file (`chat_logs.csv`) for record-keeping and potential improvement. Designed for local, privacy-focused use, it streamlines file analysis and querying for diverse applications.

## Features
- **Multimodal Support**: Processes text, images (JPG, JPEG, PNG, GIF), PDFs, CSVs, JSON, and code files (Python, JavaScript, Java, C, C++, HTML, CSS).
- **LLM Options**: Llama3 (8B), Phi, and Mistral, selectable via a dropdown menu.
- **Document Indexing**: Uses ChromaDB and HuggingFace embeddings (`BAAI/bge-large-en-v1.5`) for efficient document retrieval.
- **Chat Interface**: Streamlit-based conversational UI with streaming responses and chat history.
- **Text-to-Speech**: Converts assistant responses to audio for accessibility.
- **Logging**: Stores user queries and responses in `chat_logs.csv`.
- **Privacy**: Runs locally, ensuring sensitive data (e.g., proprietary documents, personal records) remains secure.

## Project Structure
```plaintext
├── final.py              # Main application code for Streamlit and Ollama integration
├── docs/                 # Directory for storing documents to be indexed
├── chat_logs.csv         # CSV file for logging user queries and responses
└── README.md             # This file
```

## Installation

### Prerequisites
- Python 3.8+
- Streamlit
- Ollama (with models: Llama3, Phi, Mistral, LLaVA)
- ChromaDB
- LlamaIndex
- HuggingFace Transformers
- gTTS
- Pillow
- pandas

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/multimodal-file-analysis.git
   cd multimodal-file-analysis
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```plaintext
   streamlit
   ollama
   chromadb
   llama-index
   llama-index-embeddings-huggingface
   gtts
   pillow
   pandas
   ```

4. **Install and Configure Ollama**:
   - Install Ollama: Follow instructions at [Ollama's official site](https://ollama.com/).
   - Pull required models:
     ```bash
     ollama pull llama3.1:8b
     ollama pull phi
     ollama pull mistral
     ollama pull llava:13b-v1.6
     ```

5. **Set Up Document Directory**:
   - Create a `docs/` directory in the project root to store files for indexing (e.g., PDFs, text, images).

6. **Run the Application**:
   ```bash
   streamlit run final.py
   ```
   - Access the app at `http://localhost:8501`.

## Usage
1. **Select LLM**: Choose Llama3, Phi, or Mistral from the sidebar dropdown.
2. **Upload Files**: Upload a file (PDF, text, image, CSV, JSON, or code) to be indexed and displayed.
3. **Query Files**: Enter queries in the chat input (e.g., “Summarize this PDF” or “Describe this image”) to receive streaming responses.
4. **Listen to Responses**: Play audio versions of responses via the text-to-speech feature.
5. **Review Logs**: Queries and responses are saved in `chat_logs.csv` for record-keeping.

## Ollama Performance and Scalability
The application uses **Ollama** for local inference of LLMs, optimized for single-user or low-concurrency scenarios. Below is an analysis of its performance and scalability for this application:

### Performance
- **Concurrency**: Ollama supports up to **4 parallel requests** by default (`OLLAMA_NUM_PARALLEL=4`). Each user query triggers a single request to the selected LLM or LLaVA for image processing, suitable for individual use.
- **Latency**: Text-based queries (e.g., Llama3 8B) on modest hardware (16GB RAM, NVIDIA RTX 3080) take ~1-5 seconds, depending on query complexity. Image processing with LLaVA (13B) takes ~5-15 seconds due to its vision-language architecture.
- **Throughput**: Ollama’s throughput plateaus at ~0.5 requests/second under high load, leading to increased latency for queued requests. This is sufficient for single-user scenarios but limits real-time performance for multiple users.
- **Resource Usage**:
  - **Memory**: Llama3 (8B) requires ~8-10GB RAM, Phi (~3B) ~4-6GB, Mistral (~7B) ~8GB, and LLaVA (13B) ~16-20GB, especially for image tasks.
  - **CPU/GPU**: Without GPU, inference is CPU-bound and slow. With a GPU (e.g., RTX 3080 with 10GB VRAM), LLaVA can offload ~35 layers, improving performance by 2-3x.
  - **Disk I/O**: Indexing with ChromaDB and temporary file storage (e.g., for PDFs, images) increase disk usage, particularly for large files.

### Scalability
- **Single-User Focus**: Ollama is optimized for local development, supporting **1-4 concurrent users** on modest hardware (16GB RAM, NVIDIA RTX 3080). Ideal for individual researchers, educators, or professionals.
- **Small Team**: On high-end hardware (e.g., NVIDIA A100 40GB, 64GB RAM) with tuned settings (e.g., `OLLAMA_NUM_PARALLEL=32`, `OLLAMA_MAX_LOADED_MODELS=2`), Ollama can handle **10-20 concurrent users**. Beyond this, latency increases due to resource contention.
- **Large Team/Enterprise**: For 50+ users (e.g., a research lab or corporate team), multiple Ollama instances behind a load balancer (e.g., Nginx) are needed, with each instance handling one model or a subset of users. Models should be kept in memory using `OLLAMA_KEEP_ALIVE=-1`. For larger scale, cloud-based solutions (e.g., BentoCloud) are recommended.
- **Optimizations**:
  - **Model Quantization**: Use 4-bit or 8-bit quantized models to reduce memory usage (e.g., Llama3 8B to ~6GB RAM).
  - **GPU Utilization**: Maximize GPU layers with `--gpu-layers 35` for faster inference. Monitor with `nvidia-smi`.
  - **Batch Processing**: Configure `OLLAMA_MAX_QUEUE` to manage request backlogs, though this may increase latency.
  - **Caching**: The application’s `file_cache` reduces re-indexing overhead; preloading models can further optimize performance.
  - **Docker Deployment**: Run Ollama in Docker containers with isolated resources for horizontal scaling.

## How It Helps Users
The application is designed for users across domains (e.g., education, research, legal, finance, engineering) who need to analyze files:
- **File Analysis**: Query documents (e.g., research papers, legal contracts, financial reports) for summaries or specific insights (e.g., “Extract key points from this CSV”).
- **Image Analysis**: LLaVA describes images (e.g., charts, diagrams, scanned documents), identifying text or notable features, useful for education (analyzing slides) or engineering (reviewing schematics).
- **Privacy**: Local execution ensures sensitive data remains secure, critical for domains like legal or finance.
- **Accessibility**: Text-to-speech supports multitasking or accessibility needs (e.g., listening to summaries while working).
- **Feedback Loop**: Logging to `chat_logs.csv` enables users to review interactions and improve the system.

## Extensions for Improvement
1. **Domain-Specific Models**:
   - Integrate fine-tuned LLMs (e.g., LegalBERT for legal documents, FinBERT for finance) for domain-specific accuracy.
2. **Real-Time Data Integration**:
   - Support live data feeds (e.g., real-time CSVs or APIs) for dynamic querying in analytics or research.
3. **Multi-User Support**:
   - Deploy on Kubernetes with auto-scaling for team-scale use (50+ users).
   - Add user authentication for secure access.
4. **Advanced Querying**:
   - Enable multi-document queries (e.g., “Compare data across CSVs”) or structured data extraction.
5. **Feedback Analysis**:
   - Analyze `chat_logs.csv` to identify common queries and fine-tune models.
   - Add a feedback form for users to rate response quality.
6. **Mobile Accessibility**:
   - Optimize the Streamlit interface for mobile devices.
7. **Expanded File Support**:
   - Add support for additional formats (e.g., DOCX, XLSX) to broaden applicability.
8. **Error Handling**:
   - Implement specific exception handling for file I/O, model loading, or network issues to improve debugging.

## Notes
- **File Paths**: Ensure the `docs/` directory exists and is writable.
- **Performance Tuning**: Adjust `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, and GPU layers for optimal performance.
- **Limitations**: Ollama is not suited for high-concurrency production use without significant tuning or cloud deployment.
- **Security**: Ensure temporary files are properly cleaned up to prevent data leakage.

