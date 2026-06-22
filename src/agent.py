import math
import re
from datetime import datetime

from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate


@tool
def calculator(expression:str)->str:
    """
    Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, sqrt(), log(), sin(), cos(), pi, e.
    Example input: '2 ** 10' or 'sqrt(144)' or '(3.14 * 5**2)'
    """

    allowed = {
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "abs": abs, "round": round,
        "pow": pow, "factorial": math.factorial,
    }

    try:
        clean=re.sub(r"[^0-9+\-*/().%\s_a-zA-Z]", "", expression)
        result=eval(clean,{"__builtins__": {}},allowed)
        return f"Result: {result}"
    except Exception as ex:
        return f"Could not evaluate '{expression}': {ex}"
@tool
def get_current_date_time(query: str="")->str:
    """
    Get the current date and time. Useful when the user asks about today's date,
    current time, day of week, etc. Input can be empty or any string.
    """
     
    now=datetime.now()

    return(
        f"Current date: {now.strftime('%A, %B %d, %Y')}\n"
        f"Current time: {now.strftime('%H:%M:%S')}\n"
        f"Day of week: {now.strftime('%A')}\n"
        f"Week number: {now.strftime('%W')}"
    )

@tool 
def text_analyzer(text:str)->str:
    """
    Analyze a piece of text and return statistics.
    Input should be the text you want to analyze.
    Returns: word count, sentence count, average word length, most common words.
    """

    words=text.split()

    sentences=[s.strip() for s in re.split(r'[.!?]+',text) if s.strip()]

    word_lengths=[len(w.strip(".,!?;:")) for w in words]

    stopwords={"the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
                 "for", "of", "with", "is", "are", "was", "were", "it", "this", "that"}
 
    freq={}

    for w in words:
        w_clean=w.lower().strip(".,!?;:")
        if w_clean not in stopwords and len(w_clean)>2:
            freq[w_clean]=freq.get(w_clean,0)+1
    top_words = sorted(freq.items(), key=lambda x: -x[1])[:5]
    avg_len = sum(word_lengths) / len(word_lengths) if word_lengths else 0

    return (
        f"Word count: {len(words)}\n"
        f"Sentence count: {len(sentences)}\n"
        f"Average word length: {avg_len:.1f} chars\n"
        f"Top words: {', '.join(f'{w}({c})' for w, c in top_words)}"
    )

AGENT_PROMPT="""You are a helpful research assistant with access to tools

"Availanle Tools:
{tools}

Tool Namess:{tool_names}

Use this format:

Question: the user's question
Thought: think step by step about what to do
Action: one of [{tool_names}]
Action Input: the input to the the tool
Observation: the tool result
... (repeat Thought/Action/Observation as needed)
Thought: I have enough info to answer
Final Answer: your final answer to the user

Previous Conversation:
{chat_history}

Begin!

Question:{input}
{agent_scratchpad}


"""
def build_agent(model_name:str, llm)->AgentExecutor:
    tools=[calculator, get_current_date_time, text_analyzer]

    prompt=PromptTemplate.from_template(AGENT_PROMPT)

    agent=create_react_agent(llm=llm, tools=tools,prompt=prompt)

    agent_executor=AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6
    )

    return agent_executor