import multiprocessing.process
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
import numpy as np
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
import time,multiprocessing,os,gc,threading
# import pyttsx3
from speach import speak
load_dotenv()
api_key1=os.getenv("google_key")
api_key2=os.getenv("api_key")
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=api_key1)
parser=StrOutputParser()


def add_force(url):
    

    loader=WebBaseLoader(url)
    docs=loader.load()
    vector_store=Chroma(
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L12-v2"),
        persist_directory='chroma_db1',
        collection_name='url'
    )
    vector_store.add_documents(docs)
    retriever=vector_store.get(include=["documents"])
    return retriever
    
def add(url):

    p=multiprocessing.Process(target=add_force, args=(url,))
    p.start()
    p.join()
    print("linked success")


def _force_delete(path):
    if os.path.exists(path):
        shutil.rmtree(path)
def dele():
    db_p=r"D:\internship\image_processing\ml\chroma_db1"
    p = threading.Thread(target=_force_delete, args=(db_p,))
    p.start()
    p.join()
    print("Deleted success")


url=st.text_input("Enter url here")
url = url.strip().strip("'").strip('"')
if st.button("🧿"):
    if url:
        retriever=add(url)

   


        
if st.button("DEL"):
    dele()





    





prompt=PromptTemplate(
        template="""You are a helpful ai
        check if user asks for previous question give the user question from provided transcript {chat_hist} or
    answer only from provided transcript context
    if context is insufficent ,just say you dont know.
    {context}
    Question:{Question}
    """,
    input_variables={'context','Question','chat_hist'}
)
st.title("🤖BOT")

chain=prompt | model | parser


if "human" not in st.session_state:
    st.session_state.human=[]

if "ai" not in st.session_state:
    st.session_state.ai=[]




user_input=st.chat_input("YOU:")


if user_input:

        db=r"D:\internship\image_processing\ml\chroma_db1"
        if os.path.exists(db):
            vector_store = Chroma(
                persist_directory=r"D:\internship\image_processing\ml\chroma_db1",
                collection_name="url",
                embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L12-v2")
                
                )
            retriever=vector_store.get(include=["documents"])
        else:
            retriever=None
        messages=[SystemMessage(content='You are a helpful ai assistant')]
        for i in range(len(st.session_state.human)):
            messages.append(HumanMessage(content=st.session_state.human[i]))
            messages.append(AIMessage(content=st.session_state.ai[i]))

        st.session_state.human.append(user_input)
               
        result=chain.invoke({'chat_hist':messages,'Question':user_input,'context':retriever})      
        # speak(result) 
        st.session_state.ai.append(result)

        user_input=""
        

if st.sidebar.button("clear"):
    st.session_state.human.clear()
    st.session_state.ai.clear()

for i in range(len(st.session_state.human)):
    text=st.session_state.human[i]
    atext=st.session_state.ai[i]

    
    st.write(f" YOU:{text}")
    st.write(f"AI:{atext}")
    if st.button("🔊", key=f"speak_{i}"):
        speak(atext)

             



        





 



  



