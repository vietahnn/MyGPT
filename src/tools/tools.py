import math
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilyResearch
from database.database import save_memory, search_memory
from rag.rag import retrive_from_rag
load_dotenv()

CURRENT_THREAD_ID = "default"

def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id

web_search = TavilyResearch(
    max_results = 5,
    topic = "general",
    search_depth = "advanced"
)



@tool
def caculator(expression:str) -> str:
    """
    Evaluates a mathematical expression and returns the result as a string.

    This tool takes a standard mathematical expression string, parses it, 
    and computes the numerical result. It supports basic arithmetic operations 
    such as addition (+), subtraction (-), multiplication (*), and division (/).

    Args:
        expression (str): A string representing the mathematical formula 
            to evaluate (e.g., "2 * (3 + 5) - 4").

    Returns:
        str: The calculated result formatted as a string, or an error 
            message if the evaluation fails.

    Raises:
        ValueError: If the expression contains invalid characters or malformed syntax.
        ZeroDivisionError: If the expression attempts to divide a number by zero.
    """

    try:
        allowed = {
            "math" : math,
            "abs" : abs,
            "round": round,
            "min" : min,
            "max" : max,
            "sum" : sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Caculation error: {str(e)}"
    
@tool
def search_uploaded_documents(query:str) ->str:
    """
    Search uploaded documents for relevant information.
    Use this when the user asks about uploaded PDFs, DOCS, TXT, notes, files, or documents
    """
    return retrive_from_rag(
        query=query,
        thread_id=CURRENT_THREAD_ID
    )    
@tool
def remember_this(memory:str) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something
    """

    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory
    )

@tool
def recall_memory(query:str) -> str:
    """
    Recall saved long-term memories about the user or this conversation
    """

    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query = query
    )


tools = [
    caculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search
]


