import os
from typing import List
import pymupdf4llm
from llama_index.core import Document

def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Loads PDF documents from the given file paths and converts them to LlamaIndex Document objects.

    Args:
        file_paths (List[str]): A list of paths to the PDF files.

    Returns:
        List[Document]: A list of LlamaIndex Document objects.
    """
    documents = []
    for pdf_path in file_paths:
        print(f"Processing {pdf_path}...")
        try:
            # Extract text as Markdown for better structural representation
            md_text = pymupdf4llm.to_markdown(pdf_path)

            # Create a LlamaIndex Document object
            doc = Document(
                text=md_text,
                metadata={"file_name": os.path.basename(pdf_path)}
            )
            documents.append(doc)
        except Exception as e:
            print(f"Failed to process {pdf_path}: {e}")

    print(f"Successfully loaded and processed {len(documents)} PDF documents.")
    return documents 