
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = st.secrets["GROQ_API_KEY"]

def CreateLLM(modelname):

    llm = ChatGroq(model=modelname, groq_api_key=api_key)

    return llm