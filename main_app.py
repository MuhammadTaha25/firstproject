import streamlit as st
from audio import transcribe_audio
from langchain.chat_models import ChatOpenAI
from pineconedb import manage_pinecone_store
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
#call the function to create the chain
import os
from dotenv import load_dotenv
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#initialize the history
history=[]
retriever=manage_pinecone_store()
LLM = ChatOpenAI(
                model_name='gpt-4o-mini',
                openai_api_key=OPENAI_API_KEY,
                temperature=0)
retriever=manage_pinecone_store()
def create_expert_chain(LLM=None, retriever=retriever):
    """
    Create a chain for answering questions as an expert on Elon Musk.

    Parameters:
        llm (object): The language model to use for generating responses.
        retriever (object): A retriever for fetching relevant context based on the question.

    Returns:
        object: A configured chain for answering questions about Elon Musk.
    """
    # Define the prompt template
    prompt_str = """
    You are a highly knowledgeable and conversational chatbot specializing in providing accurate and insightful information about Elon Musk.
    Answer all questions as if you are an expert on his life, career, companies, and achievements.
    Context: {context}
    Question: {question}
    conversation_history: {chat_history}
    """
    _prompt = ChatPromptTemplate.from_template(prompt_str)

    # Chain setup
    history_fetcher=itemgetter("chat_history")
    query_fetcher = itemgetter("question")  # Extract the question from input
    
    setup = {"question": query_fetcher, "context": query_fetcher | retriever| format_docs }
 # Add your setup logic here
    if setup is None:
        raise ValueError("Setup is not properly initialized.")

    _prompt = _prompt = ChatPromptTemplate.from_template(prompt_str)

    LLM = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=OPENAI_API_KEY
    )

    parser = StrOutputParser()

# Build the chain
    chain = setup | _prompt | LLM | parser
    return chain
# Set the title of the app
st.title("Ask Anything About Elon Musk")

# Initialize components

# Chat container to display conversation
chat_container = st.container()
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_input():
    st.session_state.send_input=True
# Input field for queries

with st.container():
    query = st.text_input("Please enter a query", key="query", on_change=send_input)
    send_button = st.button("Send", key="send_btn")  # Single send button

# Chat logic
if send_button or send_input and query:
    with st.spinner("Processing... Please wait!"):  # Spinner starts here
        response = create_expert_chain().invoke({'question': query,"chat_history":"\n".join(str(history))})
        print(response)
        query="user_question:"+query
        response="ai_response:"+response
        history.append((query, response))  # Generate response
    # Update session state with user query and AI response
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", response))

with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)
