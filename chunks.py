from doc_loader import load_wikipedia_content
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

WIKIPEDIA_URL=os.environ["WIKIPEDIA_URL"]
bs4_SoupStrainer_Class=os.environ["bs4_SoupStrainer_Class"]

def process_wikipedia_content(url=WIKIPEDIA_URL, class_name=bs4_SoupStrainer_Class, chunk_size=1500, chunk_overlap=150):
    """
    Fetch and split content from a Wikipedia page into manageable chunks.

    Parameters:
        url (str): The URL of the Wikipedia page to fetch content from.
        class_name (str): The class name of the HTML element to target for extraction.
        chunk_size (int, optional): The maximum size of each chunk (default is 1500 characters).
        chunk_overlap (int, optional): The overlap between chunks for context continuity (default is 100 characters).

    Returns:
        list: A list of document chunks as dictionaries.
    """
    # Load content from the specified Wikipedia page
    docs = load_wikipedia_content(url, class_name)

    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )

    # Split the documents into smaller chunks
    chunks = text_splitter.split_documents(docs)

    return chunks
