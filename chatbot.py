from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from  dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

STRICT INSTRUCTIONS:                                       
* USE ONLY provided context to answer the user's question.
                                          
* if provided context is no sufficient just tell context is not sufficient


Context:
{context}

Question:
{question}

Answer:
""")

chatbot = prompt | llm

