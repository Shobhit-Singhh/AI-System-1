from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


def create_vector_store(doc: list, persist_directory: str = "./../data/VDB" ) -> Chroma:
    """
    Create a vector store using Chroma for the given list of documents.

    Args:
        doc (list): A list of Document objects to be stored in the vector store.
        persist_directory (str): The directory where the vector store will be persisted. Default is "vector_store".

    Returns:
        Chroma: The created vector store.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=doc, persist_directory=persist_directory, embedding=embeddings
    )
    return vector_store


def load_vector_store(persist_directory: str = "./../data/VDB") -> Chroma:
    """
    Load a vector store from the specified directory.

    Args:
        persist_directory (str): The directory where the vector store is persisted. Default is "vector_store".

    Returns:
        Chroma: The loaded vector store.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )
    return vector_store


def retrieve(query: str, k: int = 2):
    """
    Retrieve documents from the vector store based on a query.

    Args:
        query (str): The query string to search for.
        k (int): The number of top results to return. Default is 2.

    Returns:
        list: A list of retrieved documents.
    """
    vdb = load_vector_store()
    retriever = vdb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    results = retriever.invoke(query)
    return results


def test_create():
    from Loader import load_text
    from Splitter import split_texts

    file_path = "../../data/Documents/DSM5.txt"

    documents = load_text(file_path)
    print("Document loaded successfully.")

    chunks = []
    for doc in documents:
        chunks.extend(split_texts(doc.page_content, chunk_size=1000, chunk_overlap=300))

    # Create a vector store from the chunked documents
    vector_store = create_vector_store(chunks)

    print("Vector store created and persisted successfully.")
    print("Total documents in vector store:", len(vector_store))

    id = vector_store.get()["ids"]
    print("IDs of documents in vector store:\n", id)

    print(vector_store.get(ids=[id[0]]))
    result = vector_store.get(
        ids=[id[0]], include=["embeddings", "documents", "metadatas"]
    )


def test_loader():
    vdb = load_vector_store()

    retriever = vdb.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    query = """Failure to conform to social norms with respect to lawful behaviors.
                2. Deceitfulness.
                3. Impulsivity or failure to plan ahead.
            """
    results = retriever.invoke(query)
    print("Results:\n", results)

def test_retrieve():
    query = """Failure to conform to social norms with respect to lawful behaviors.
                2. Deceitfulness.
                3. Impulsivity or failure to plan ahead.
            """
    results = retrieve(query, k=2)
    print("Results:\n", results)
    return results


if __name__ == "__main__":
    # test_create()
    test_loader()
    test_retrieve()
    
