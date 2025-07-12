import os
import src.config as config
from src.data_loader import load_documents
from src.vector_store_manager import VectorStoreManager

def main():
    """
    Main function to create/update the vector store collection.
    """
    # Ensure the data directory exists
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
        print(f"Created data directory: {config.DATA_DIR}")
        print("Please add your PDF files to this directory and run the script again.")
        return

    # Check for required environment variables
    if not all([config.OPENAI_API_KEY, config.QDRANT_URL, config.QDRANT_API_KEY]):
        print("Error: Required environment variables (OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY) are not set.")
        print("Please create a .env file with these values.")
        return

    # Initialize the vector store manager
    manager = VectorStoreManager(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.COLLECTION_NAME,
        vector_size=config.VECTOR_SIZE,
        embed_model_name=config.EMBED_MODEL_NAME
    )

    # Create the collection if it doesn't exist
    manager.create_collection_if_not_exists()

    # Get existing and current filenames
    existing_files = manager.get_existing_filenames()
    try:
        current_files = set(os.listdir(config.DATA_DIR))
    except FileNotFoundError:
        current_files = set()

    # Determine new files to process
    new_files = current_files - existing_files
    new_pdf_files = [f for f in new_files if f.lower().endswith(".pdf")]

    if not new_pdf_files:
        print("All PDFs are already in the collection. No new files to add.")
        return

    print(f"Found {len(new_pdf_files)} new PDF files to add.")

    # Process and add new documents
    new_pdf_paths = [os.path.join(config.DATA_DIR, f) for f in new_pdf_files]
    documents = load_documents(new_pdf_paths)

    if documents:
        manager.add_documents(documents)
        print(f"Successfully added {len(new_pdf_files)} new PDFs to the collection.")

if __name__ == "__main__":
    main()
