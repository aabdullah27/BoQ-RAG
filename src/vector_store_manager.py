from typing import List, Set
from qdrant_client import QdrantClient, models
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

class VectorStoreManager:
    """
    Manages operations related to the Qdrant vector store.
    """
    def __init__(self, url: str, api_key: str, collection_name: str, vector_size: int, embed_model_name: str):
        """
        Initializes the VectorStoreManager.

        Args:
            url (str): The URL of the Qdrant instance.
            api_key (str): The API key for the Qdrant instance.
            collection_name (str): The name of the collection.
            vector_size (int): The size of the vectors.
            embed_model_name (str): The name of the embedding model.
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embed_model_name = embed_model_name
        self.client = QdrantClient(url=url, api_key=api_key)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
        )

    def create_collection_if_not_exists(self):
        """
        Creates the Qdrant collection with binary quantization if it doesn't exist.
        """
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                    on_disk=True
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        always_ram=True
                    )
                )
            )
            print(f"Collection '{self.collection_name}' created.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def get_existing_filenames(self) -> Set[str]:
        """
        Retrieves the set of filenames for documents already in the collection.

        Returns:
            Set[str]: A set of filenames.
        """
        response, _ = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=10000,  # Adjust if you have more than 10,000 documents
        )
        existing_files = set()
        for point in response:
            if "file_name" in point.payload:
                existing_files.add(point.payload["file_name"])
        return existing_files

    def add_documents(self, documents: List[Document]):
        """
        Adds new documents to the vector store.

        Args:
            documents (List[Document]): A list of LlamaIndex Document objects to add.
        """
        embed_model = OpenAIEmbedding(model_name=self.embed_model_name)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        print("Indexing new documents...")
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )
        print("Indexing complete.") 