import streamlit as st
import src.config as config
from src.query import get_query_engine

def main():
    st.set_page_config(page_title="Viab BoQ RAG Pipeline", layout="wide")
    st.title("Viab BoQ RAG Pipeline")

    # Check for required environment variables
    if not all([config.OPENAI_API_KEY, config.QDRANT_URL, config.QDRANT_API_KEY, config.GROQ_API_KEY]):
        st.error("Error: Required environment variables are not set.")
        st.error("Please create a .env file with OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, and GROQ_API_KEY.")
        return

    # Initialize the query engine
    try:
        with st.spinner("Initializing query engine... This may take a moment."):
            query_engine = get_query_engine()
        st.success("Query engine is ready.")
    except Exception as e:
        st.error(f"Failed to initialize query engine: {e}")
        return

    # User input
    query_text = st.text_input("Ask a question about your documents:", "")

    if st.button("Get Answer"):
        if query_text:
            with st.spinner("Searching for answers..."):
                try:
                    response = query_engine.query(query_text)
                    st.markdown("### Answer")
                    st.markdown(response.response)

                    st.markdown("### Sources")
                    for node in response.source_nodes:
                        st.write(f"- **{node.metadata.get('file_name', 'Unknown') }** (Score: {node.score:.2f})")

                    st.markdown("### Retrieved Raw Data")
                    for i, node in enumerate(response.source_nodes):
                        with st.expander(f"Source {i+1}: {node.metadata.get('file_name', 'Unknown')} (Score: {node.score:.2f})"):
                            node_dict = node.dict()
                            # The 'embedding' field is removed for better readability
                            node_dict["node"].pop("embedding", None)
                            st.json(node_dict)

                except Exception as e:
                    st.error(f"An error occurred while querying: {e}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main() 