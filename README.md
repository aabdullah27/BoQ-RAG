# RAG Pipeline with LlamaIndex and Qdrant

This project implements a RAG (Retrieval-Augmented Generation) pipeline to process PDF documents, embed them using OpenAI, and store them in a Qdrant vector database. The pipeline is designed to be efficient, checking for new files and only processing those that haven't been added to the collection.

## Features

- **PDF Processing**: Extracts text from PDF files using `pymupdf4llm`.
- **Vector Embeddings**: Generates embeddings using OpenAI's `text-embedding-3-small` model.
- **Vector Storage**: Uses Qdrant for efficient and scalable vector storage, with support for binary quantization.
- **Incremental Updates**: Intelligently checks for new PDFs and only processes and adds them to the vector collection, avoiding redundant processing.
- **Modular Code**: The codebase is organized into modules for configuration, data loading, and vector store management for better readability and maintenance.

## Project Structure

```
RAG-Pipline/
├── .env.example
├── data/
│   └── (add your pdf files here)
├── notebooks/
│   └── sample_boq.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   └── vector_store_manager.py
├── main.py
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Access to OpenAI, Qdrant, and Groq APIs.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RAG-Pipline
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root of the project by copying the example file:

```bash
cp .env.example .env
```

Now, open the `.env` file and add your credentials. This file is included in `.gitignore` to protect your sensitive information.

**Contents of `.env.example`:**
```
# Qdrant credentials
QDRANT_URL="your_qdrant_url"
QDRANT_API_KEY="your_qdrant_api_key"

# OpenAI credentials
OPENAI_API_KEY="your_openai_api_key"

# Groq credentials
GROQ_API_KEY="your_groq_api_key"
```

### 5. Add Your Data

Place your PDF files into the `data/` directory. If this directory doesn't exist, the script will create it for you on the first run.

### 6. Run the Pipeline

Execute the `main.py` script to start the pipeline. The script will:
1.  Check for new PDFs in the `data` directory.
2.  Process any new files by extracting text and generating embeddings.
3.  Add the new documents to your Qdrant collection.
4.  If no new files are found, it will notify you that the collection is up to date.

```bash
python main.py
```

## Running the Frontend Application

To interact with your RAG pipeline through a web interface, you can run the Streamlit application.

```bash
streamlit run app.py
```

This will start a local web server and open the application in your browser, where you can ask questions and receive answers from your documents.

## How It Works

The `main.py` script orchestrates the entire process. It initializes the `VectorStoreManager`, which handles all communication with Qdrant. It checks for new files by comparing the contents of the `data` directory with a list of already-processed files stored in Qdrant's metadata. If new files are detected, `data_loader.py` is used to load and parse them, after which they are handed back to the `VectorStoreManager` for embedding and indexing.

