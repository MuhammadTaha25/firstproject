from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer

def load_wikipedia_content(url, class_name):
    """
    Load and parse a specific section of a Wikipedia page.

    Parameters:
        url (str): The URL of the Wikipedia page to load.
        class_name (str): The class name of the section to parse.

    Returns:
        list: A list of parsed documents from the specified section of the page.
    """
    loader = WebBaseLoader(
        url,
        bs_kwargs=dict(parse_only=SoupStrainer(class_=(class_name)))
    )
    return loader.load()

# Example usage
url = 'https://en.wikipedia.org/wiki/Elon_Musk'
class_name = 'mw-content-ltr mw-parser-output'
documents = load_wikipedia_content(url, class_name)

# Optional: Print the first document's content
if documents:
    print(documents[0].metadata)

