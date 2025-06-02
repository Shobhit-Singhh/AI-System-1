# from langchain_openai import OpenAIEmbeddings
# from langchain_core.embeddings import Embeddings

# def create_embeddings(doc: str ,model_name: str = "text-embedding-3-small") -> "Embeddings":
#     """
#     Create embeddings using OpenAI's text embedding model.

#     Args:
#         doc (str): The text document to be embedded.
#         model_name (str): The name of the OpenAI embedding model to use. Default is "text-embedding-3-small".

#     Returns:
#         Embeddings: The embeddings for the text document.
#     """
#     embeddings = OpenAIEmbeddings(model=model_name)
#     return embeddings.embed_query(doc)



# if __name__ == "__main__":
#     sample_text = "This is a sample text document for embedding."
    
#     # Example usage
#     print("Creating embeddings for the sample text...")
#     embeddings = create_embeddings(sample_text, model_name="text-embedding-3-small")
    
#     print("Embeddings created successfully.")
#     print("Embedding vector:", embeddings)
#     print("Embedding vector length:", len(embeddings))