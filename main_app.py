from chainCreation import create_expert_chain
import streamlit as st
from audio import transcribe_audio
from pineconedb import manage_pinecone_store
#call the function to create the chain
manage_pinecone_store()
chain=create_expert_chain()
#initialize the history
history=[]
# Set the title of the app
st.title("Ask Anything About Elon Musk")

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
        response = chain.invoke({'question': query,"chat_history":"\n".join(str(history))})
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