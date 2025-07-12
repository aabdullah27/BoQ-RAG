import src.config as config
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.groq import Groq

def get_query_engine():
    """
    Initializes and returns a LlamaIndex query engine connected to the Qdrant vector store.

    Returns:
        RetrieverQueryEngine: The initialized query engine.
    """
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")

    # Initialize Qdrant client and vector store
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    vector_store = QdrantVectorStore(client=client, collection_name=config.COLLECTION_NAME)

    # Initialize the embedding model
    embed_model = OpenAIEmbedding(model_name=config.EMBED_MODEL_NAME)

    # Create the index from the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    # LLM for response generation
    llm = Groq(model="llama-3.3-70b-versatile")

    # Retriever to fetch relevant documents
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
    )

    # Response Synthesizer to generate a response from the retrieved context
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="compact",
    )

    # Assemble the query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    return query_engine 