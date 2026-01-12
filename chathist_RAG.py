from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
# import pyttsx3
from speach import speak
load_dotenv()
api_key=os.getenv("google_key")

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=api_key)



parser=StrOutputParser()
#------------------------------------------------------------------------------------
# url='https://rdltech.in/biometric-authentication'
# loader=WebBaseLoader(url)
# docs=loader.load()


# from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# from dotenv import load_dotenv
# import os
# load_dotenv()
# api_key=os.getenv("google_key")
# vector_store=Chroma(
#     embedding_function=GoogleGenerativeAIEmbeddings(model='models/embedding-001',google_api_key=api_key),
#     persist_directory='chroma_db',
#     collection_name='rdl'
# )
# vector_store.add_documents(docs)
#-----------------------------------------------------------


vector_store = Chroma(
    persist_directory="chroma_db",
    collection_name="rdl",
    embedding_function=GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
)

retriever=vector_store.get(include=["documents"])


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

# if st.button("🧿"):
if user_input:
    # if user_input:
        messages=[SystemMessage(content='You are a helpful ai assistant')]
        for i in range(len(st.session_state.human)):
            messages.append(HumanMessage(content=st.session_state.human[i]))
            messages.append(AIMessage(content=st.session_state.ai[i]))

        st.session_state.human.append(user_input)
               
        result=chain.invoke({'chat_hist':messages,'Question':user_input,'context':retriever})      
        speak(result) 
        st.session_state.ai.append(result)

        user_input=""
        

if st.button("clear"):
    st.session_state.human.clear()
    st.session_state.ai.clear()

for i in range(len(st.session_state.human)):
    text=st.session_state.human[i]
    atext=st.session_state.ai[i]
    st.write(f" YOU:{text}")
    st.write(f"AI:{atext}")


 



  



