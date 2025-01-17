import os
import time
import random
import requests
import streamlit as st
from io import StringIO
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.chains.combine_documents import create_stuff_documents_chain



#Api key for GROQ Inferencing
groq_api_key = "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"


#Distrubuting api calls for Google Embeddings
google_embeddings_key_dict =    {   1 : "AIzaSyBh-9JgI2cukvSy0Db3jIxAFvheYAvWWrA",
                                    2 : "AIzaSyClrQxnq6tHP6l8M2GDdQJ9NhGruwgwYP8",
                                    3 : "AIzaSyDoayRBBKUExDc62uyqLOZtiUm5t3Kj3fs",
                                    4 : "AIzaSyDA97Y1lZYcESEbKNN4HzXOyu4RkTanp8U",
                                    5 : "AIzaSyDjYPlT5Rrg8OWJeKMSByM27JI-dG52VP4",
                                    6 : "AIzaSyA0Bq8_6coTew-6stME-J3mRe84uiBUTOU",
                                    7 : "AIzaSyC1eqTc-ZY3ff6TtLGV4GI3wbiyMmlXHoI"
                                }
google_embeddings_key_index = random.randint(1, 7)
google_api_key = google_embeddings_key_dict[google_embeddings_key_index]


if not groq_api_key or not google_api_key:
    st.error("API keys for Groq and Google are required.")
    st.stop()


# Load API keys to environment variables 
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key



def load_uploaded_document(uploaded_file):
    if uploaded_file.type == "application/pdf":
        temp_file_path = os.path.join("temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        os.remove(temp_file_path)  # Clean up temp file
    else:
        text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        # Wrap text in Document objects
        docs = [Document(page_content=text)]
    
    return docs


def scrape_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Failed to retrieve the webpage.")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    # Wrap text in Document objects
    return [Document(page_content=text)]



#App Title
st.logo("VulcanLogo.png", icon_image="VulcanLogo.png")

st.set_page_config(page_title="Vulcan v1", page_icon="VulcanLogo1.jpeg")

st.title("Vulcan v1 DEPLOYED🚀")
st.header("Just Ask.")
st.write("")



#Setting default values of glabal variables
selected_model = "Llama-3.2-11b-text-preview"

option = "Normal Chat"


#Sidebar 
with st.sidebar:

    st.title("Select Model:")
    selected_model = st.sidebar.selectbox('Choose:', ["Llama-3.2-11b-text-preview", "Llama-3.2-90b-text-preview", "Llama-3.2-3b-preview", "Llama-3.2-1b-preview", "Llama-3.1-8b-instant", "Llama-3.1-70b-versatile", "Mixtral-8x7b-32768", "Gemma2-9b-it"], key='selected_model')
    
    
    st.divider()
    
    
    option = st.selectbox("Actions:", ["Normal Chat", "Upload Document", "Scrape Website"])
    
    if option=="Upload Document":
        uploaded_file = st.file_uploader("Upload your document:", type=["pdf", "txt", "csv"])
        try:
            if uploaded_file:
                docs = load_uploaded_document(uploaded_file)
                st.success("Document loaded successfully!")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    elif option=="Scrape Website":
        url = st.text_input("Enter the exact URL of the website to scrape:")

        if url:
            try:
                docs = scrape_website(url)
                st.success("Website content scraped successfully!")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
               
                
    st.divider()
    
    
    if st.button("Clear Memory", type="secondary"):
        st.session_state.history = []
        
    
    st.divider()
    
    
    st.caption("© Project by Avneesh Jadhav")
    st.caption("© Vulcan")
    
    
    st.markdown(
    """
    <a href="https://www.linkedin.com/in/avneeshjadhav/">
            <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBkPSJNMTkgMGgtMTRjLTIuNzYxIDAtNSAyLjIzOS01IDV2MTRjMCAyLjc2MSAyLjIzOSA1IDUgNWgxNGMyLjc2MiAwIDUtMi4yMzkgNS01di0xNGMwLTIuNzYxLTIuMjM4LTUtNS01em0tMTEgMTloLTN2LTExaDN2MTF6bS0xLjUtMTIuMjY4Yy0uOTY2IDAtMS43NS0uNzktMS43NS0xLjc2NHMuNzg0LTEuNzY0IDEuNzUtMS43NjQgMS43NS43OSAxLjc1IDEuNzY0LS43ODMgMS43NjQtMS43NSAxLjc2NHptMTMuNSAxMi4yNjhoLTN2LTUuNjA0YzAtMy4zNjgtNC0zLjExMy00IDB2NS42MDRoLTN2LTExaDN2MS43NjVjMS4zOTYtMi41ODYgNy0yLjc3NyA3IDIuNDc2djYuNzU5eiIvPjwvc3ZnPg==">

    </a>
    """, unsafe_allow_html=True,
    )
    
    

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name=selected_model)

# Define the prompt template with chat memory
prompt_template = ChatPromptTemplate.from_template(
"""
You are an AI assistant. Answer the questions based on the provided context and the previous conversation without explicitly mentioning that you are answering based on on the provided context and the previous conversation.
If any question is not related to the context or history, directly provide your own answer without mentioning about the context or history.
Provide the most accurate response based on the context and conversation history.
If any question is not related to the context or history, provide the most accurate response based on your own knowledge.

<context>
{context}
</context>

<conversation_history>
{history}
</conversation_history>

Questions: {input}
"""
)



# Initialize session state for chat history
if 'history' not in st.session_state:
    st.session_state.history = []
    

#Display History
for message in range(0, len(st.session_state.history), 2):
        st.write("🔸𝗬𝗼𝘂:")
        st.subheader(st.session_state.history[message][:])
        st.write("")
        if(message+1 < len(st.session_state.history)):
            st.write("🔹𝗟𝗟𝗠:")
            st.write(st.session_state.history[message+1][:])
            st.divider()



# Ensure temp directory exists
if not os.path.exists("temp"):
    os.makedirs("temp")
    


# Chat modes
if option == "Normal Chat":
    
    try:
        
        question = st.chat_input("Enter your question...")
        if question:
            start=time.process_time()
            
            st.write("🔸𝗬𝗼𝘂:")
            st.subheader(question)
            st.write("")
            st.session_state.history.append(f"{question}")
            
            document_chain = create_stuff_documents_chain(llm, prompt_template)
                        
            response = document_chain.invoke({'context': [], 'history': "\n".join(st.session_state.history), 'input': question})
                    
            st.write("🔹𝗟𝗟𝗠:")
            st.write_stream(StringIO(response))
            st.write("Response Time:", time.process_time()-start)
            st.session_state.history.append(f"{response}")
            
    except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(f"Refresh page, switch llm model and try again. [Chat history will be cleared]")
            

elif option == "Upload Document":
    
    if uploaded_file:
        
        try:

            question = st.chat_input("Enter your question...")
            if question:
                start=time.process_time()
                
                st.write("🔸𝗬𝗼𝘂:")
                st.subheader(question)
                st.write("")
                st.session_state.history.append(f"{question}")
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectors = FAISS.from_documents(docs, embeddings)
                
                document_chain = create_stuff_documents_chain(llm, prompt_template)
                retriever = vectors.as_retriever()
                relevant_docs = retriever.get_relevant_documents(question)
                
                response = document_chain.invoke({'context': relevant_docs, 'history': "\n".join(st.session_state.history), 'input': question})
                
                st.write("🔹𝗟𝗟𝗠:")
                st.write_stream(StringIO(response))
                st.write("Response Time:", time.process_time()-start)
                st.session_state.history.append(f"{response}")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(f"Refresh page, switch llm model and try again. [Chat history will be cleared]")


elif option == "Scrape Website":

    if url:
        
        try:
            
            question = st.chat_input("Enter your question...")
            if question:
                start=time.process_time()
                
                st.write("🔸𝗬𝗼𝘂:")
                st.subheader(question)
                st.write("")
                st.session_state.history.append(f"{question}")
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectors = FAISS.from_documents(docs, embeddings)
                
                document_chain = create_stuff_documents_chain(llm, prompt_template)
                retriever = vectors.as_retriever()
                relevant_docs = retriever.get_relevant_documents(question)
                
                response = document_chain.invoke({'context': relevant_docs, 'history': "\n".join(st.session_state.history), 'input': question})
                
                st.write("🔹𝗟𝗟𝗠:")
                st.write_stream(StringIO(response))
                st.write("Response Time:", time.process_time()-start)
                st.session_state.history.append(f"{response}")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(f"Refresh page, switch llm model and try again. [Chat history will be cleared]")
