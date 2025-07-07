import os
from typing import Union, List, Dict, Optional, TypedDict
from datetime import datetime
import json

from dotenv import load_dotenv

from vector_store_manager import SupabaseVectorManager

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily.tavily_crawl import TavilyCrawl
from langgraph.graph import MessagesState, END, StateGraph
from pydantic import BaseModel, Field, validator

from prompts import(
    knowledge_gap_prompt_template, rag_prompt_template,
    sufficiency_prompt_template
)

load_dotenv()

crawler = TavilyCrawl(max_depth=1)
manager = SupabaseVectorManager()

# llm = ChatOllama(model="qwen3:0.6b")
# llm = ChatOllama(model="gemma3n:e2b", temperature=0.3)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Documents(BaseModel):
    id: str
    url: str
    title: str
    content: str
    valid_at: datetime
    invalid_at: Optional[datetime]
    invalid_cause: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: Union[Dict[str, str], str]  # Allow both dict and string

    @validator('metadata', pre=True)
    def parse_metadata(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {"content": value}  # Fallback
        return value

class SearchQueries(BaseModel):
    queries: List[str] = Field(default_factory=list, description="A list of search query strings.")

# Define a simple Pydantic model for the expected metadata structure
class DocumentMetadata(BaseModel):
    title: Optional[str] = Field(None, description="The title of the document or source.")
    url: Optional[str] = Field(None, description="The URL of the document source.")
    summary: Optional[str] = Field(None, description="A brief summary of the document content.")
    # Add other potential metadata fields as needed
    
class State(TypedDict):
    query: str
    goal: str
    desired_focus: str
    target_audience: str
    search_queries: SearchQueries = Field(default_factory=SearchQueries)
    is_sufficient: bool
    documents: List[Documents]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

sufficiency_parser = StrOutputParser()
sufficiency_chain = sufficiency_prompt_template | llm | sufficiency_parser

structured_llm = llm.with_structured_output(SearchQueries)

knowledge_gap_chain = knowledge_gap_prompt_template | structured_llm

rag_chain = rag_prompt_template | llm | StrOutputParser()

def construct_query_from_state(state: State) -> str:
    """
    Constructs a detailed, context-rich query from the structured State.

    Args:
        state: The State object containing all relevant details.

    Returns:
        A formatted, descriptive string ready to be embedded for semantic search.
    """
    parts = [f"The user is requesting information about: '{state['query']}'."]

    if state.get('goal'):
        parts.append(f"The overall goal is: '{state['goal']}'.")

    if state.get('target_audience'):
        parts.append(f"The target audience for this information is: '{state['target_audience']}'.")

    if state.get('desired_focus'):
        parts.append(f"The immediate search focus is on: '{state['desired_focus']}'. This is the most important part of the query.")

    rich_query = " ".join(parts)
    print(f"Constructed Rich Query:\n---\n{rich_query}\n---")
    return rich_query

def initial_retrieve_node(state):
    print("---NODE: Initial Retrieve---")
    rich_query = construct_query_from_state(state)

    document = manager.perform_hybrid_search(rich_query, k=50)
    print(f"Retrieved {len(document)} documents initially.")
    return {"documents": document}

def check_sufficiency_node(state):
    print("---NODE: Check Sufficiency---")
    query = state["query"]
    documents = state["documents"]
    desired_focus = state["desired_focus"]

    if not documents:
        print("No documents found, deemed insufficient.")
        return {"is_sufficient": False}

    context_str = "\n\n".join([doc.page_content for doc in documents])
    response = sufficiency_chain.invoke({"query": query, "details": desired_focus, "context": context_str})
    print(f"Sufficiency check LLM response: '{response}'")
    
    is_sufficient = False # Initialize to False by default

    # First, try to extract from within a potential </think> block
    if "</think>" in response:
        parts = response.split("</think>")
        if len(parts) > 1:
            potential_decision = parts[-1].strip().lower() # Strip and lower for robustness
            if potential_decision == "true":
                is_sufficient = True
    
    stripped_output = response.strip().lower() # Strip and lower for robustness
    if stripped_output == "true" or stripped_output == "yes" or stripped_output == True:
        is_sufficient = True

    print(f"Knowledge is sufficient: {is_sufficient}")
    return {"is_sufficient": is_sufficient}

def knowledge_planner_node(state):
    print("---NODE: Knowledge Planner---")
    query = state["query"]
    documents = state["documents"]
    desired_focus = state["desired_focus"]
    context_str = "\n\n".join([doc.page_content for doc in documents])
    
    search_queries = knowledge_gap_chain.invoke({"query": query, "details": desired_focus, "context": context_str})

    print(search_queries)
    return {"search_queries": search_queries}

def gather_and_process_node(state):
    print("---NODE: Gather and Process New Knowledge---")
    
    if not state["search_queries"].queries:
        print("No new search queries planned. Skipping gathering.")
        # No new docs, so documents state remains unchanged from previous retrieval
        return {"documents": state["documents"]} 
    
    list_of_queries = state["search_queries"].queries
    print(list_of_queries)
    
    for search_query in state["search_queries"].queries:
        print(f"Gathering for: '{search_query}'...")
        manager.create_documents_from_search(search_query)
        
    # After attempting to add new knowledge, always re-retrieve to get the latest consolidated set.
    print("Re-retrieving documents after augmentation attempt.")
    query = state["query"]
    # Retrieve more documents now, as new relevant ones might have been added.
    final_documents = manager.perform_hybrid_search(query, k=50)
    print(f"Retrieved {len(final_documents)} documents after augmentation round.")
    return {"documents": final_documents}

# --- Conditional Edges ---

def decide_to_augment_or_answer(state):
    print(f"---DECISION POINT: Sufficiency check result: {state['is_sufficient']}---")
    sufficient_value = state.get("is_sufficient")
    if sufficient_value is True or (isinstance(sufficient_value, str) and sufficient_value.lower() == "true"):
        print("Sufficient, go directly to answer generation.")
        return "END"  # Sufficient, go directly to answer generation
    else:
        print("Insufficient, plan how to augment.")
        return "plan_knowledge_augmentation" # Insufficient, plan how to augment

workflow = StateGraph(State)

workflow.add_node("initial_retriever", initial_retrieve_node)
workflow.add_node("check_sufficiency", check_sufficiency_node)
workflow.add_node("plan_knowledge_augmentation", knowledge_planner_node)
workflow.add_node("gather_and_process", gather_and_process_node)

# Set the entry point
workflow.set_entry_point("initial_retriever")
workflow.add_edge("initial_retriever", "check_sufficiency")

workflow.add_conditional_edges(
    "check_sufficiency",
    decide_to_augment_or_answer,
    {
        "END" : END,
        "plan_knowledge_augmentation" : "plan_knowledge_augmentation"
    }
)

workflow.add_edge("plan_knowledge_augmentation", "gather_and_process")
workflow.add_edge("gather_and_process", END)
graph = workflow.compile()

if __name__ == "__main__":
    result = graph.invoke({
        "query": "make a course of pydantic ai",
        "goal": "To gather foundational knowledge for creating lessons for the course of Pydantic AI",
        "desired_focus": "more technical",
        "target_audience": "beginner"
    })
    print(result.get("documents"))