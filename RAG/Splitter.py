from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_texts(document: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    Split the text document into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        document (str): The text document to be split.
        chunk_size (int): The size of each chunk. Default is 1000 characters.
        chunk_overlap (int): The number of overlapping characters between chunks. Default is 200 characters.

    Returns:
        list: A list of text chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.create_documents([document])



if __name__ == "__main__":
    from Loader import load_text
    
    sample_text = load_text("./../data/Documents/DSM5.txt")[0].page_content

    print("Splitting text into chunks...")     
    chunks = split_texts(sample_text, chunk_size=1000, chunk_overlap=300)
    
    print("Number of chunks created:", len(chunks))
    
    for index, chunk in enumerate(chunks):
        print(f"Chunk {index + 1}\n {chunk}\n{'*' * 100}\n")
        