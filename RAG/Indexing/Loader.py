from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

def load_text(file_path: str) -> "Document":
    """
    Load the text document using the TextLoader.
    Returns:
        Document: The loaded document.
    """
    
    LoadText = TextLoader( file_path = file_path, encoding = "utf-8")
    return LoadText.load()

def load_Dir(file_path: str) -> "Document":
    
    """
    Load all text documents in a directory using the TextLoader.
    
    Returns:
        Document: The loaded document.
    """
    
    LoadText = DirectoryLoader(file_path, loader_cls=TextLoader, glob="**/*.txt")
    return LoadText.load()

    


if __name__ == "__main__":
    file_path = "../../data/Documents/DSM5.txt"
    dir_path = "../../data/Documents/"

    document = load_text(file_path)
    print("Loaded Single Document Successfully:")

    documents = load_Dir(dir_path)
    for doc in documents:
        print(doc)
    print("Total Documents Loaded:", len(documents))

