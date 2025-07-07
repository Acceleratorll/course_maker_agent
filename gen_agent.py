import os

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Annotated, List, Optional, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import traceable
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode
from vector_rag import State, workflow
from langchain_core.documents import Document

from schemas import (
    CourseState, UserInputAnalysis, ObjectivesList, ModulesList, 
    SearchQuery, SearchInput, Knowledge, Lesson
)

from prompts import (
    analysis_instructions, sys_msg, module_organizer_instructions,
    get_generate_objective_instructions, get_generate_lesson_instructions,
    summary_maker
)

load_dotenv()

# Uncomment or choose your desired LLM
# llm = ChatOllama(model="qwen3:1.7b", temperature=0.3)
# llm_decider = ChatOllama(model="gemma3n:e2b", temperature=0.3)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_decider = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
vector_rag = workflow.compile()

@tool("search_web_tool", args_schema=SearchInput, return_direct=False)
def search_web_tool(
    research_need: str,
    desired_focus: Optional[str] = None,
    context: Optional[str] = None,
    goal: Optional[str] = None
) -> List[Knowledge]:
    """
    Performs a web search to find relevant material and information.
    Takes a research need and optional context/focus, generates an optimized query,
    executes the search, and returns results as a list of Knowledge objects.
    """
    # Use the globally defined LLM for query generation
    query_generation_llm = llm.with_structured_output(SearchQuery)
    tavily_search = TavilySearchResults(max_results=10)

    # Construct prompt for query generation
    prompt_parts = [
        "You are an expert at crafting precise web search queries.",
        "Generate the single best search query to find relevant information based on the following:",
        f"\n**Research Need:** \"{research_need}\""
    ]
    if desired_focus:
        prompt_parts.append(f"- **Focus:** \"{desired_focus}\"")
    if context:
        prompt_parts.append(f"- **Audience/Context:** \"{context}\"")
    if goal:
        prompt_parts.append(f"- **Information Goal:** \"{goal}\"")
    prompt_parts.append("\nReturn ONLY the search query string.")
    query_instructions = SystemMessage(content="\n".join(prompt_parts))

    # Generate the search query
    try:
        generated_query_obj = query_generation_llm.invoke([
            query_instructions,
            HumanMessage(content="Generate the query.")
        ])
        search_query = generated_query_obj.search_query
        if not search_query:
            return [Knowledge(title="Query Generation Failed", source="Internal LLM", content="LLM did not return a valid search query.")]
    except Exception as e:
        return [Knowledge(title="Query Generation Error", source="Internal LLM", content=f"Error generating search query: {e}")]

    print(f"Executing web search for query: '{search_query}'")
    # Execute the search
    try:
        search_results = tavily_search.invoke(search_query)
    except Exception as e:
        print(f"Error during web search: {e}")
        return [Knowledge(title="Search Execution Error", source="Tavily", content=f"Failed to execute search for query '{search_query}'. Error: {e}")]

    # Process search results into Knowledge objects
    knowledge_list = []
    if not search_results:
        knowledge_list.append(Knowledge(title="No Results Found", source="Tavily", content=f"No search results found for query: '{search_query}'."))
    else:
        for i, result in enumerate(search_results, start=1):
            content = result.get("content", "Content not available.")
            if not isinstance(content, str):
                content = str(content)
            knowledge_list.append(
                Knowledge(
                    title=result.get("title", f"Search Result {i}") or f"Search Result {i}",
                    source=result.get("url", "Unknown source"),
                    content=content
                )
            )
    return knowledge_list

tools = [search_web_tool]
llm_with_tools = llm.bind_tools(tools)

@traceable(name="analyze_user_input")
def analyze_user_input(state: CourseState):
    """
    Analyze the user's input and extract key course parameters:
    subject, title, target_audience, user_goal, language and added_details.
    """
    command = state['messages']
    
    structured_llm = llm_with_tools.with_structured_output(UserInputAnalysis)
    result = structured_llm.invoke([
        analysis_instructions,
        HumanMessage(content=f"{command[-1].content}")
    ])
    print(f"Analyzed User Input")
    return {
        "subject": result.subject,
        "title": result.title,
        "language": result.language,
        "target_audience": result.target_audience,
        "added_details": result.added_details
    }

@traceable(name="generate_core_course")
def generate_core_course(state: CourseState):
    """ Generate a core element of making a course """
    try:
        subject = state["subject"]
        title = state["title"]
        target_audience = state["target_audience"]
        added_details = state["added_details"]
        language = state['language']
    except KeyError as e:
        return {"messages": state.get("messages", []) + [AIMessage(content=f"Error: Missing key {e} in state.")]}

    structured_llm = llm_with_tools.with_structured_output(ObjectivesList)
    instructions = get_generate_objective_instructions(subject, title, language, target_audience, added_details)
    try:
        objectives_instance = structured_llm.invoke([
            instructions,
            HumanMessage(content="Generate the list of learning objectives for the course.")
        ])
    except Exception as e:
        return {"messages": state.get("messages", []) + [AIMessage(content=f"Error during generation of objectives: {e}")]}

    if not hasattr(objectives_instance, "objectives"):
        return {"messages": state.get("messages", []) + [AIMessage(content="Error: The LLM output is missing the 'objectives' field.")]}

    print(f"Successfully generated objectives.")
    return {"objective": objectives_instance.objectives}

@traceable(name="course_knowledge_gatherer_node")
def course_knowledge_gatherer_node(state: CourseState):
    """
    Gathers knowledge for the course by utilizing the search_web_tool.
    Formulates search queries based on course objectives and updates the state with results.
    """
    if not state.get("objective") or not state.get("subject"):
        print("Error: Objectives or subject not found in state for knowledge gathering.")
        return {"messages": state.get("messages", []) + [AIMessage(content="I need course objectives and subject to gather knowledge. Something went wrong.")]}

    objective_goals = "; ".join([obj.goal for obj in state["objective"]])
    research_topic = f"Key concepts and information for a course on '{state['subject']}' (titled '{state['title']}') with objectives: {objective_goals}"
    # Use string representation of target_audience as context
    context = str(state.get("target_audience"))
    print(f"Initiating knowledge gathering for: {research_topic}")
    print(f"Context: {context}")
    print(f"Desired focus: {state.get('added_details')}")

    # Enhanced goal for vector_rag to emphasize accuracy and completeness
    knowledge_gathering_goal = (
        f"To gather comprehensive, accurate, and highly relevant foundational knowledge "
        f"for creating detailed and authoritative lessons for the course '{state['title']}'. "
        f"This knowledge will serve as the primary source for the course content."
    )

    try:
        gathered_knowledge_result = vector_rag.invoke({
            "query": research_topic,
            "target_audience": context,
            "goal": knowledge_gathering_goal, # Use the enhanced goal
            "desired_focus": state.get("added_details")
        })
        gathered_knowledge = gathered_knowledge_result.get("documents")
    except Exception as e:
        error_message = f"An error occurred during knowledge gathering: {e}"
        print(error_message)
        return {"messages": state.get("messages", []) + [AIMessage(content=error_message)]}

    if not gathered_knowledge:
        error_message = "Knowledge gathering completed, but no relevant resources were found. This might impact course quality."
        print(error_message)
        return {"messages": state.get("messages", []) + [AIMessage(content=error_message)]}
    print(f"Successfully gathered {len(gathered_knowledge)} knowledge resources.")
    return {
        "knowledge": gathered_knowledge,
        "messages": state.get("messages", []) + [AIMessage(content=f"I've gathered some initial knowledge for the course. Found {len(gathered_knowledge)} resources.")]
    }

def objectives_to_str(objectives) -> str:
    if not objectives:
        return "No learning objectives provided."
    
    lines = []
    for i, obj in enumerate(objectives, 1):
        lines.append(f"   Goal: {obj.goal}")
        lines.append(f"   Description: {obj.description}")
        lines.append(f"   Scope: {obj.scope}")
        lines.append("")
    return "\n".join(lines)

@traceable(name="module_organizer_node")
def module_organizer_node(state: CourseState):
    """
    Organizes the course into modules and lessons based on objectives, target audience and gathered knowledge.
    """
    
    instructions = module_organizer_instructions(
        language=state["language"], 
        title=state["title"], 
        target_audience=str(state["target_audience"]), 
        objective=objectives_to_str(state["objective"]), 
        added_details=state.get("added_details")
    )
    
    structured_llm = llm.with_structured_output(ModulesList)
    modules_instance = structured_llm.invoke([instructions, HumanMessage(content="Generate the module structure.")])
    print(f"Successfully generated modules. Found {len(modules_instance.modules)} modules.")
    return {"modules": modules_instance.modules}


@traceable(name="lesson_writer")
def lesson_writer(state: CourseState):
    """ Generate lessons for each module based on course knowledge and module structure """
    print("Generating lessons for each module...")
    knowledge = state["knowledge"]
    target_audience = state["target_audience"]
    modules = state["modules"]
    language = state["language"]
    course_title = state["title"] # Get course_title from state

    if not modules:
        print("Error: No modules found in state for lesson writing.")
        return {"messages": state.get("messages", []) + [AIMessage(content="I need the module structure to write lessons. Something went wrong.")]}

    updated_modules = []
    for module in modules:
        print(f"Generating lessons for {module.number} - {module.title}")
        for lesson in module.lessons:
            structured_llm = llm_with_tools.with_structured_output(Lesson)
            instructions = get_generate_lesson_instructions(
                course_title=course_title, # Pass course_title
                language=language,
                module_number=module.number.split(" ")[-1], # Extract number from "Module X"
                module_title=module.title,
                module_achieved=module.goal,
                relevant_knowledge_base=knowledge,
                course_target_audience=target_audience,
                lesson=lesson # Pass the lesson object
            )
            try:
                lesson_content = structured_llm.invoke([
                    instructions,
                    HumanMessage(content=f"Generate content for lesson {lesson.number}: {lesson.title} based on the provided details and knowledge.")
                ])
                # Update the lesson object with the generated content
                lesson.explanation = lesson_content.explanation
                lesson.case_study = lesson_content.case_study
                lesson.idea = lesson_content.idea
                lesson.reflection_questions = lesson_content.reflection_questions
                print(f"Generated content for lesson {lesson.number}: {lesson.title}.")
            except Exception as e:
                print(f"Error generating content for lesson {lesson.number}: {e}")
                # Optionally add an error lesson or message to the module
                lesson.explanation = f"Could not generate content for this lesson due to an error: {e}"
                lesson.case_study = ""
                lesson.idea = ""
                lesson.reflection_questions = ""


        updated_modules.append(module)

    print(f"Successfully generated lessons for each modules.")
    return {"modules": updated_modules}

@traceable(name="finalize_course")
def finalize_course(state: CourseState):
    """ Combine course components to write a Course Summary """
    modules = state["modules"]
    language = state["language"]
    
    llm_response = llm_with_tools.invoke([
        summary_maker(modules, language), 
        HumanMessage(content="Write a summary for the course.")
    ])
    return {"summary": llm_response.content}

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Routing
def entry_point_passthrough(state):
    """Passes the state through. Acts as a named entry for conditional routing."""
    return

def is_user_want_make_a_course(state):
    user_input = str(state["messages"]) or "assistant"
    instruction = SystemMessage(content="""You are an AI decider. 
                                Your purpose is to decide the next node to take. is user want to make a course or learning book?
                                The output should be:
                                'assistant' if the user dont want to make a course
                                'analyze_user_input' if the user want to make a course
                                Dont add any additional text, explanations, or markdown formatting.
                               """)
    result = llm_decider.invoke([instruction, HumanMessage(content=f"{user_input}")])
    raw_output = result.content
    
    decision_word = None
    
    if "</think>" in raw_output:
        parts = raw_output.split("</think>")
        if len(parts) > 1:
            # Get the text after </think> and strip it
            potential_decision = parts[-1].strip()
            if potential_decision == "assistant":
                decision_word = "assistant"
            elif potential_decision == "analyze_user_input":
                decision_word = "analyze_user_input"
    
    if decision_word is None:
        stripped_output = raw_output.strip()
        if stripped_output == "assistant":
            decision_word = "assistant"
        elif stripped_output == "analyze_user_input":
            decision_word = "analyze_user_input"

    if decision_word:
        return decision_word
    else:
        print(f"Warning: LLM output could not be reliably parsed to 'assistant' or 'analyze_user_input'. Raw output: '{raw_output}'. Defaulting to 'assistant'.")
        return "assistant"

course_maker = StateGraph(CourseState)
course_maker.add_node("entry_point_passthrough", entry_point_passthrough)
course_maker.add_node("assistant", assistant)
course_maker.add_node("analyze_user_input", analyze_user_input)
course_maker.add_node("generate_core_course", generate_core_course)
course_maker.add_node("course_knowledge_gatherer_node", course_knowledge_gatherer_node)
course_maker.add_node("module_organizer_node", module_organizer_node)
course_maker.add_node("lesson_writer", lesson_writer)
course_maker.add_node("finalize_course", finalize_course)
course_maker.add_node("tools", ToolNode(tools))

course_maker.add_edge(START, "entry_point_passthrough")
course_maker.add_conditional_edges("entry_point_passthrough", is_user_want_make_a_course, ["assistant", "analyze_user_input"])
course_maker.add_edge("analyze_user_input", "generate_core_course")
course_maker.add_edge("generate_core_course", "course_knowledge_gatherer_node")
course_maker.add_edge("course_knowledge_gatherer_node", "module_organizer_node")
course_maker.add_edge("module_organizer_node", "lesson_writer")
course_maker.add_edge("lesson_writer", "finalize_course")
course_maker.add_edge("finalize_course", END)

course_maker.add_conditional_edges("assistant", tools_condition)
course_maker.add_edge("tools", "assistant")

graph = course_maker.compile()