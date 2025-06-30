import os
from typing import Union, List, Dict, Optional, TypedDict
from datetime import datetime
import json

from dotenv import load_dotenv

from vector_store_manager import SupabaseVectorManager

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily.tavily_crawl import TavilyCrawl
from langgraph.graph import MessagesState, END, StateGraph
from pydantic import BaseModel, Field, validator

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

sufficiency_prompt_template = ChatPromptTemplate.from_template(
    """Your task is to evaluate if the provided documents contain enough information to answer the user's query.

    **User Query:** "{query}"
    **User Specific Request:** "{details}"

    **Retrieved Documents:**
    ---
    {context}
    ---

    **Instructions:**
    Based *only* on the information within the "Retrieved Documents," determine if a person with expertise in the subject area could write a comprehensive and factually correct answer to the "User Query" and "User Specific Request".

    - A **comprehensive** answer directly addresses all parts of the user's query.
    - A **factually correct** answer is one that can be fully supported by the provided text.

    Respond with only 'true' or 'false'.
    """
)

sufficiency_parser = StrOutputParser()
sufficiency_chain = sufficiency_prompt_template | llm | sufficiency_parser

structured_llm = llm.with_structured_output(SearchQueries)
knowledge_gap_prompt_template = ChatPromptTemplate.from_template(
    """Objective: Generate targeted search queries to fill knowledge gaps.

    User's Initial Question: "{query}"
    User's Specific Request: "{details}"

    Available Information (deemed insufficient):
    <context>
    {context}
    </context>

    The available information does not fully answer the "User's Initial Question" with "User's Specific Request".
    To find the missing pieces, what 1 to 3 specific questions should we ask a search engine (like Google or Tavily)?
    These questions should aim to uncover new information directly relevant to answering "{query}" that is not already present in the "Available Information."

    Think step-by-step:
    1. What aspects of "{query}" with "{details}" are not addressed by the "Available Information"?
    2. What specific facts, details, or perspectives are missing?
    3. Formulate these as concise questions or keyword-based search terms.

    Output Requirements:
    - Provide your answer ONLY as a JSON list of strings.
    - Each string must be a search query.
    - No other text or explanation.

    Make sure search queries to uncover new information directly relevant.
    """
)

knowledge_gap_chain = knowledge_gap_prompt_template | structured_llm

# Chain for RAG to generate the final answer
rag_prompt_template = ChatPromptTemplate.from_template(
"""
**[Persona]**
You are a highly capable and meticulous AI assistant. Your primary goal is to provide accurate, helpful, and well-structured answers by strictly adhering to the provided context. You do not possess any knowledge outside of the given context. When the context does not contain the answer, you will state that clearly.

**[Instructions]**
1.  **Analyze the User's Request:** Carefully examine the user's query to understand the core intent and the specific information being sought.
2.  **Contextual Grounding:** Your entire response must be based *exclusively* on the information provided in the "Context" section and your own training data. Do not introduce any external information.
3.  **Chain of Thought (CoT):** Before generating the final answer, formulate a step-by-step reasoning process to ensure accuracy and logical flow. This process should outline how you will use the context to address the user's query.
4.  **Construct the Answer:** Based on your Chain of Thought, compose a clear and concise answer.
5.  **Output Format:** The final output must be a single string.

**[Context]**
---
{context}
---

**[User Request]**
{query}

**[Chain of Thought]**
*   **Step 1: Deconstruct the User's Query:** I need to identify the key question or command in the user's request.
*   **Step 2: Scan Context for Keywords:** I will search the provided context for terms and concepts directly related to the user's query.
*   **Step 3: Synthesize Relevant Information:** I will gather the relevant sentences and data points from the context that directly answer the query.
*   **Step 4: Formulate the Response:** Based on the synthesized information, I will construct a helpful and direct answer, ensuring it strictly adheres to the provided context. If no relevant information is found, I will explicitly state that the context does not provide an answer.

**[Response]**
(Your final, user-facing answer goes here)
"""
)
# Using the main llm for answer generation
rag_chain = rag_prompt_template | llm | StrOutputParser()

metadata_prompt_template = ChatPromptTemplate.from_template(
       """Generate well-structured JSON metadata based on the following context:
       {context}
       
       Output Requirements:
       - Return valid JSON only
       - Use the exact field names from the schema
       - For datetime fields, use ISO 8601 format
       - For null values, use null (not 'None')
       - Metadata field should be a JSON object
       """
   )

def construct_rich_query(
    query: str,
    goal: Optional[str] = None,
    desired_focus: Optional[str] = None,
    target_audience: Optional[str] = None
) -> str:
    """
    Constructs a detailed, context-rich query for semantic search.

    Args:
        query: The core user query.
        goal: The user's primary objective.
        desired_focus: The specific emphasis for the results (e.g., 'technical', 'high-level').
        target_audience: The intended audience (e.g., 'beginner', 'expert').

    Returns:
        A formatted string ready to be embedded.
    """
    if not query:
        raise ValueError("The core query cannot be empty.")

    # Start with the core user query
    parts = [f"The user is searching for information about: '{query}'."]

    # Conditionally add more context if provided
    if goal:
        parts.append(f"The main goal is to {goal}.")
    
    if desired_focus:
        parts.append(f"The search results should have a strong '{desired_focus}' focus.")
    
    if target_audience:
        parts.append(f"The content is intended for a '{target_audience}' audience.")

    # Join all parts into a single, natural-sounding paragraph
    rich_query = " ".join(parts)
    
    print(f"Constructed Rich Query: {rich_query}")
    return rich_query

def initial_retrieve_node(state):
    print("---NODE: Initial Retrieve---")
    user_query = state["query"]
    goal = state["goal"]
    desired_focus = state["desired_focus"]
    target_audience = state["target_audience"]
    rich_query = construct_rich_query(
        query=user_query,
        goal=goal,
        desired_focus=desired_focus,
        target_audience=target_audience
    )
    document = manager.perform_hybrid_search(rich_query, k=25) # Adjust k as needed
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
    if stripped_output == "true":
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
    final_documents = manager.perform_hybrid_search(query, k=20)
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

import os
from typing import Union, List, Dict, Optional, TypedDict
from datetime import datetime
import json

from dotenv import load_dotenv

from vector_store_manager import SupabaseVectorManager

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily.tavily_crawl import TavilyCrawl
from langgraph.graph import MessagesState, END, StateGraph
from pydantic import BaseModel, Field, validator

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

sufficiency_prompt_template = ChatPromptTemplate.from_template(
    """Your task is to evaluate if the provided documents contain enough information to answer the user's query.

    **User Query:** "{query}"
    **User Specific Request:** "{details}"

    **Retrieved Documents:**
    ---
    {context}
    ---

    **Instructions:**
    Based *only* on the information within the "Retrieved Documents," determine if a person with expertise in the subject area could write a comprehensive and factually correct answer to the "User Query" and "User Specific Request".

    - A **comprehensive** answer directly addresses all parts of the user's query.
    - A **factually correct** answer is one that can be fully supported by the provided text.

    Respond with only 'true' or 'false'.
    """
)

sufficiency_parser = StrOutputParser()
sufficiency_chain = sufficiency_prompt_template | llm | sufficiency_parser

structured_llm = llm.with_structured_output(SearchQueries)
knowledge_gap_prompt_template = ChatPromptTemplate.from_template(
    """Objective: Generate targeted search queries to fill knowledge gaps.

    User's Initial Question: "{query}"
    User's Specific Request: "{details}"

    Available Information (deemed insufficient):
    <context>
    {context}
    </context>

    The available information does not fully answer the "User's Initial Question" with "User's Specific Request".
    To find the missing pieces, what 1 to 3 specific questions should we ask a search engine (like Google or Tavily)?
    These questions should aim to uncover new information directly relevant to answering "{query}" that is not already present in the "Available Information."

    Think step-by-step:
    1. What aspects of "{query}" with "{details}" are not addressed by the "Available Information"?
    2. What specific facts, details, or perspectives are missing?
    3. Formulate these as concise questions or keyword-based search terms.

    Output Requirements:
    - Provide your answer ONLY as a JSON list of strings.
    - Each string must be a search query.
    - No other text or explanation.

    Make sure search queries to uncover new information directly relevant.
    """
)

knowledge_gap_chain = knowledge_gap_prompt_template | structured_llm

# Chain for RAG to generate the final answer
rag_prompt_template = ChatPromptTemplate.from_template(
"""
**[Persona]**
You are a highly capable and meticulous AI assistant. Your primary goal is to provide accurate, helpful, and well-structured answers by strictly adhering to the provided context. You do not possess any knowledge outside of the given context. When the context does not contain the answer, you will state that clearly.

**[Instructions]**
1.  **Analyze the User's Request:** Carefully examine the user's query to understand the core intent and the specific information being sought.
2.  **Contextual Grounding:** Your entire response must be based *exclusively* on the information provided in the "Context" section and your own training data. Do not introduce any external information.
3.  **Chain of Thought (CoT):** Before generating the final answer, formulate a step-by-step reasoning process to ensure accuracy and logical flow. This process should outline how you will use the context to address the user's query.
4.  **Construct the Answer:** Based on your Chain of Thought, compose a clear and concise answer.
5.  **Output Format:** The final output must be a single string.

**[Context]**
---
{context}
---

**[User Request]**
{query}

**[Chain of Thought]**
*   **Step 1: Deconstruct the User's Query:** I need to identify the key question or command in the user's request.
*   **Step 2: Scan Context for Keywords:** I will search the provided context for terms and concepts directly related to the user's query.
*   **Step 3: Synthesize Relevant Information:** I will gather the relevant sentences and data points from the context that directly answer the query.
*   **Step 4: Formulate the Response:** Based on the synthesized information, I will construct a helpful and direct answer, ensuring it strictly adheres to the provided context. If no relevant information is found, I will explicitly state that the context does not provide an answer.

**[Response]**
(Your final, user-facing answer goes here)
"""
)
# Using the main llm for answer generation
rag_chain = rag_prompt_template | llm | StrOutputParser()

metadata_prompt_template = ChatPromptTemplate.from_template(
       """Generate well-structured JSON metadata based on the following context:
       {context}
       
       Output Requirements:
       - Return valid JSON only
       - Use the exact field names from the schema
       - For datetime fields, use ISO 8601 format
       - For null values, use null (not 'None')
       - Metadata field should be a JSON object
       """
   )

def construct_rich_query(
    query: str,
    goal: Optional[str] = None,
    desired_focus: Optional[str] = None,
    target_audience: Optional[str] = None
) -> str:
    """
    Constructs a detailed, context-rich query for semantic search.

    Args:
        query: The core user query.
        goal: The user's primary objective.
        desired_focus: The specific emphasis for the results (e.g., 'technical', 'high-level').
        target_audience: The intended audience (e.g., 'beginner', 'expert').

    Returns:
        A formatted string ready to be embedded.
    """
    if not query:
        raise ValueError("The core query cannot be empty.")

    # Start with the core user query
    parts = [f"The user is searching for information about: '{query}'."]

    # Conditionally add more context if provided
    if goal:
        parts.append(f"The main goal is to {goal}.")
    
    if desired_focus:
        parts.append(f"The search results should have a strong '{desired_focus}' focus.")
    
    if target_audience:
        parts.append(f"The content is intended for a '{target_audience}' audience.")

    # Join all parts into a single, natural-sounding paragraph
    rich_query = " ".join(parts)
    
    print(f"Constructed Rich Query: {rich_query}")
    return rich_query

def initial_retrieve_node(state):
    print("---NODE: Initial Retrieve---")
    user_query = state["query"]
    goal = state["goal"]
    desired_focus = state["desired_focus"]
    target_audience = state["target_audience"]
    rich_query = construct_rich_query(
        query=user_query,
        goal=goal,
        desired_focus=desired_focus,
        target_audience=target_audience
    )
    document = manager.perform_hybrid_search(rich_query, k=25) # Adjust k as needed
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
    if stripped_output == "true":
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
    final_documents = manager.perform_hybrid_search(query, k=20)
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
