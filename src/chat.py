from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a helpful AI assistant. Be concise and accurate."""

def convert_history(history):
    messages = []
    print ("inside history")
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


def plain_chat(user_input, history,llm):

    

   

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        *history,
        HumanMessage(content=user_input)
    ])

    chain = prompt | llm | StrOutputParser()

    print(chain.invoke({}),"llm response")
    return chain.invoke({})

