import os
import operator

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Annotated, List, Optional, TypedDict
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.constants import Send
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import traceable
from langchain_core.tools import tool

class TargetAudience(BaseModel):
    age_range: Optional[str] = None
    experience_level: Optional[str] = None
    prior_knowledge: Optional[str] = None
    interests: Optional[str] = None
    learning_style: Optional[str] = None
    goals: str = Field(description="The goals of the target audience")
    pain_points: Optional[str] = None
    demographics: Optional[str] = None

class Objective(BaseModel):
    goal: str
    description: str = Field(description="The description of the goal")
    scope: str = Field(description="The scope of the goal")

class ObjectivesList(BaseModel):
    objectives: List[Objective]

class Knowledge(BaseModel):
    title: str
    source: str = Field(description="The source of the knowledge")
    content: str = Field(description="The content of the gathered knowledge")
    relevance_score: Optional[float] = Field(None, description="An optional score indicating relevance if provided by the search source.")

class Lesson(BaseModel):
    number: str = Field(description="The number of the lesson(e.g., Lesson 1.1). format:module[number].lesson[number]")
    title: str = Field(description="The title of the lesson")
    explanation: str = Field(description="The explanation of the lesson")
    case_study: str = Field(description="The case study of the lesson")
    idead: str = Field(description="Ideas for simple interactive exercises (even text-based ones initially)")
    reflection_questions: str = Field(description="Reflection questions for the lesson")
    
class LessonsList(BaseModel):
    lessons: List[Lesson]

class Homework(BaseModel):
    task: str = Field(description="The task to be completed.")
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        min_items=1,
        default_factory=list
    )

class Prerequisite(BaseModel):
    content: str = Field(description="The content of the prerequisite")

class Modules(BaseModel):
    number: str = Field(description="The number of the module(e.g., Module 1)")
    title: str = Field(description="The title of the module(e.g., Fundamental of AI Agent)")
    topic: str = Field(description="The topic of the module")
    lessons: List[Lesson] = Field(description="The lessons in the module")

class ModulesList(BaseModel):
    modules: List[Modules]

class CourseState(MessagesState):
    title: str
    subject: str
    added_details: Optional[str] = None
    target_audience: TargetAudience = Field(description="The target audience of the course")
    user_input: str = Field(description="The user input or command")
    prerequisites: List[Prerequisite] = Field(description="The prerequisites of the course")
    objective: List[Objective] = Field(default_factory=list, description="The objective of the course")
    modules: List[Modules] = Field(default_factory=list, description="The modules of the course")
    lesson: List[Lesson] = Field(default_factory=list, description="The lessons of the course")
    knowledge: List[Knowledge] = Field(default_factory=list, description="The knowledge of the lesson")
    summary: str
    description: str

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class UserInputAnalysis(BaseModel):
    title: str
    subject: str
    target_audience: TargetAudience
    user_goal: Optional[str] = None
    added_details: Optional[str] = None

load_dotenv()

# Uncomment or choose your desired LLM
# llm = ChatOllama(model="qwen3:1.7b", temperature=0.3)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")

class SearchInput(BaseModel):
    research_need: str = Field(description="The primary topic, question, or information need that requires web research. This is the core of the search.")
    desired_focus: Optional[str] = Field(None, description="Specific aspects, keywords, or angles to focus on or prioritize in the search. Helps narrow down results.")
    context: Optional[str] = Field(None, description="Brief context about who this information is for, if it influences the type or depth of information needed (e.g., 'for a beginner', 'for a technical expert').")
    goal: Optional[str] = Field(None, description="What will this information be used for? (e.g., 'to answer a specific user question', 'to get a general overview', 'to find supporting data').")

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
    subject, title, target_audience, user_goal and added_details.
    """
    command = state['messages']
    analysis_instructions = SystemMessage(
        content=(
            "You are a highly intelligent AI assistant dedicated to analyze user requests for course creation. "
            "Your primary objective is to analyze the user's command and extract specific parameters and requirements for a new course. "
            "You MUST identify and extract the following key pieces of information and structure your output as a JSON object conforming to the `UserInputAnalysis` schema, which includes a nested `TargetAudience` object:"
            "\n\n"
            "1.  **title**: The desired title for the course. "
            "    - If the user explicitly states a title, use that exact title. "
            "    - If not explicitly stated, infer a concise and relevant working title based on the extracted 'subject', 'target_audience' details (if available), and the overall user request."
            "\n\n"
            "2.  **subject**: The main subject matter or academic domain of the course."
            "\n\n"
            "3.  **target_audience**: Details about the specific group of learners this course is intended for. Extract as many of the following details as possible into a nested object:"
            "    - `age_range`: The typical age range of the learners (e.g., 'adults', 'teenagers', 'university students')."
            "    - `experience_level`: Their prior experience with the subject (e.g., 'beginners', 'intermediate', 'advanced', 'no prior knowledge')."
            "    - `prior_knowledge`: Specific knowledge or skills they are expected to have before starting the course."
            "    - `interests`: Their general interests that might be relevant to the course subject."
            "    - `learning_style`: Any indicated preferences for how they learn (e.g., 'visual', 'hands-on', 'lecture-based')."
            "    - `goals`: What they hope to achieve by taking the course (distinct from the overall `user_goal`)."
            "    - `pain_points`: Challenges or problems they are facing that the course could help solve."
            "    - `demographics`: Other relevant demographic information (e.g., 'professionals', 'students', 'hobbyists')."
            "    - If specific details are not mentioned, infer them based on the subject and user request, or leave the field as null."
            "\n\n"
            "4.  **user_goal**: The primary goal or outcome the user (the one requesting the course) wants the target audience to achieve by completing the course. This is the high-level purpose of the course from the user's perspective."
            "\n\n"
            "5.  **added_details**: Any additional specific instructions, requirements, or details the user provides regarding the course content, structure, or specific elements to include. Capture these details verbatim or summarize them accurately."
            "\n\n"
            "Your output MUST be ONLY a JSON object matching the `UserInputAnalysis` schema, without any additional text, explanations, or markdown formatting outside the JSON."
        )
    )
    structured_llm = llm_with_tools.with_structured_output(UserInputAnalysis)
    result = structured_llm.invoke([
        analysis_instructions,
        HumanMessage(content=f"{command[-1].content}")
    ])
    return {
        "subject": result.subject,
        "title": result.title,
        "target_audience": result.target_audience,
        "added_details": result.added_details
    }

@traceable(name="generate_objective")
def get_generate_objective_instructions(subject, title, target_audience, added_details: Optional[str] = None):
    target_audience_str = str(target_audience)
    return SystemMessage(
        content=f"""You are an expert AI instructional designer with a specialization in curriculum development.
        Your primary mission is to brainstorm the Course Details and then, craft a list of clear, compelling, and actionable learning objectives for a course.

        **Course Details to Brainstorm:**
        1.  **Course Subject:** "{subject}"
        2.  **Course Title:** "{title}"
        3.  **Target Audience Profile:** {target_audience_str}
        4.  **Added Details:** "{added_details}"

        **Core Task: Generate Learning Objectives**

        You must generate a list of approximately 3-5 key learning objectives. Each objective must clearly define what a learner will be able to **DO** upon successful completion of this course or a significant module within it.

        **Detailed Instructions for Each Objective:**

        For each learning objective, you will populate three fields: `goal`, `description`, and `scope`.

        1.  **`goal` (The Actionable Statement):**
            *   This is the most critical part. It **MUST** start with the phrase: "Upon successful completion of this course, learners will be able to..."
            *   Follow this phrase with a **strong, specific, action-oriented verb** that is observable and measurable (e.g., Analyze, Design, Implement, Evaluate, Create, Compare, Explain, Differentiate, Troubleshoot). Avoid vague verbs like "understand" or "learn about."
            *   The rest of the sentence should clearly state the skill, knowledge, or competency the learner will acquire or demonstrate.
            *   *Example `goal`*: "Upon successful completion of this course, learners will be able to design a responsive user interface for a mobile application using Figma."

        2.  **`description` (Elaboration and Context):**
            *   Provide a brief (1-2 sentences) elaboration of the `goal`.
            *   This should clarify the context, the 'what,' or the 'why' of the objective. It can add detail to the skill or knowledge being imparted.
            *   It helps to explain the importance or relevance of achieving this specific `goal`.
            *   *Example `description` (for the goal above)*: "This involves applying principles of mobile-first design and utilizing Figma's prototyping tools to create interactive mockups that adapt to various screen sizes."

        3.  **`scope` (Boundaries and Focus):**
            *   Define the breadth or limits of this objective (1-2 sentences).
            *   Indicate what is included or excluded, or the specific context/conditions under which the skill will be applied or demonstrated.
            *   It helps to set realistic expectations and defines the focus area for that particular objective.
            *   *Example `scope` (for the goal above)*: "The focus will be on core UI components and navigation patterns for common mobile app use cases, excluding advanced animations or backend integration."

        **Guiding Principles for Objective Formulation:**

        *   **Alignment:** Ensure EVERY objective is directly and logically aligned with the provided **Course Subject**, **Course Title**, and **Target Audience Profile**.
        *   **Specificity:** Objectives should be precise and unambiguous.
        *   **Measurability:** The action verb should allow for assessment of whether the learner has achieved the objective.
        *   **Attainability (for the Target Audience):** Consider the `Target Audience Profile`. Objectives for 'beginners' should focus on foundational skills, while objectives for 'advanced' learners can target more complex synthesis or specialized knowledge or evaluation. The language and complexity should be appropriate.
        *   **Relevance:** Each objective should contribute meaningfully to the overall learning outcomes of the course.

        **Output Format:**

        *   Your response **MUST** be **ONLY** a single, valid JSON object.
        *   This JSON object **MUST** strictly adhere to the `ObjectivesList` schema, which looks like this:
            {{ "objectives": [ {{ "goal": "...", "description": "...", "scope": "..." }}, ... ] }}
        *   Do **NOT** include any additional text, explanations, or markdown formatting outside of this JSON object.
        """
    )

@traceable(name="generate_core_course")
def generate_core_course(state):
    """ Generate a core element of making a course """
    try:
        subject = state["subject"]
        title = state["title"]
        target_audience = state["target_audience"]
        added_details = state["added_details"]
    except KeyError as e:
        return {"messages": state.get("messages", []) + [AIMessage(content=f"Error: Missing key {e} in state.")]}

    structured_llm = llm_with_tools.with_structured_output(ObjectivesList)
    instructions = get_generate_objective_instructions(subject, title, target_audience, added_details)
    try:
        objectives_instance = structured_llm.invoke([
            instructions,
            HumanMessage(content="Generate the list of learning objectives for the course.")
        ])
    except Exception as e:
        return {"messages": state.get("messages", []) + [AIMessage(content=f"Error during generation of objectives: {e}")]}

    if not hasattr(objectives_instance, "objectives"):
        return {"messages": state.get("messages", []) + [AIMessage(content="Error: The LLM output is missing the 'objectives' field.")]}

    return {"objective": objectives_instance.objectives}

def get_generate_lesson_instructions(module_number, module_title, module_topic, course_objectives, relevant_knowledge_base, course_target_audience, course_title):
    return SystemMessage(
        content=f"""You are an Expert AI Instructional Designer and Educational Content Creator.
        Your mission is to develop detailed lesson plans for a specific module within a larger course. Generate multiple lesson plans based on the module's topic, the overall course objectives, and a provided knowledge base.
        These lesson plans should be tailored for the specified target audience and fit within the overall course context.
        Prioritize incorporating active learning strategies, where students engage with the material (e.g., discussions, problem-solving, hands-on activities).

        Phase 1: Understand the Context
            1. Overall Course Title: {course_title}
            2. Target Audience: {course_target_audience}
            3. Overall Course Objectives: {course_objectives}
            4. Module Number: {module_number}
            5. Module Title: {module_title}
            6. Module Topic: {module_topic}
            7. Knowledge Base: ```{relevant_knowledge_base}```

        Phase 2: Design the Lesson Plans for Module {module_number} - "{module_title}"
        Based on the Module Topic, Course Objectives, and Knowledge Base, design a sequence of lessons for this specific module.
        Each lesson plan MUST include:
            1. Lesson Number: Format as ModuleNumber.LessonNumber (e.g., "1.1", "1.2", "2.1").
            2. Lesson Title: A concise and engaging title.
            3. Explanation: Detailed content for the lesson.
            4. Case Study: A relevant case study or example.
            5. Idead: Ideas for simple interactive exercises (even text-based ones initially).
            6. Reflection Questions: Questions to prompt learner reflection.

        Output:
            - Provide a JSON object matching the LessonsList schema, for example:
            {{ "lessons": [{{"number": "{module_number}.1", "title": "...", "explanation": "...", "case_study": "...", "idead": "...", "reflection_questions": "..."}}, ...] }}
        Do not include any additional text, explanations, or markdown formatting outside the JSON.
        """
    )

@traceable(name="lesson_writer")
def lesson_writer(state: CourseState):
    """ Generate lessons for each module based on course knowledge and module structure """
    objective = state["objective"]
    knowledge = state["knowledge"]
    target_audience = state["target_audience"]
    title = state["title"]
    modules = state["modules"]

    if not modules:
        print("Error: No modules found in state for lesson writing.")
        return {"messages": state.get("messages", []) + [AIMessage(content="I need the module structure to write lessons. Something went wrong.")]}

    updated_modules = []
    for module in modules:
        print(f"Generating lessons for {module.number} - {module.title}")
        structured_llm = llm_with_tools.with_structured_output(LessonsList)
        instructions = get_generate_lesson_instructions(
            module_number=module.number.split(" ")[-1], # Extract number from "Module X"
            module_title=module.title,
            module_topic=module.topic,
            course_objectives=objective,
            relevant_knowledge_base=knowledge,
            course_target_audience=target_audience,
            course_title=title
        )
        try:
            lessons_instance = structured_llm.invoke([
                instructions,
                HumanMessage(content=f"Generate lessons for {module.number} based on the provided details and knowledge.")
            ])
            module.lessons = lessons_instance.lessons
            print(f"Generated {len(module.lessons)} lessons for {module.number}.")
        except Exception as e:
            print(f"Error generating lessons for {module.number}: {e}")
            # Optionally add an error lesson or message to the module
            module.lessons = [Lesson(number=f"{module.number.split(' ')[-1]}.0", title="Error generating lessons", explanation=f"Could not generate lessons for this module due to an error: {e}", case_study="", idead="", reflection_questions="")]


        updated_modules.append(module)

    # Update the modules field in the state with the lessons added
    return {"modules": updated_modules}

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
    gathered_knowledge = search_web_tool.invoke({
        "research_need": research_topic,
        "context": context,
        "goal": f"To gather foundational knowledge for creating lessons for the course '{state['title']}'.",
        "desired_focus": state.get("added_details")
    })
    if not gathered_knowledge or any(k.title in ["Query Generation Failed", "Query Generation Error", "Search Execution Error", "No Results Found"] for k in gathered_knowledge):
        error_message = f"I tried to gather knowledge but encountered an issue: {gathered_knowledge[0].content if gathered_knowledge else 'Unknown error during search tool invocation.'}"
        print(error_message)
        return {"messages": state.get("messages", []) + [AIMessage(content=error_message)]}
    print(f"Successfully gathered {len(gathered_knowledge)} knowledge resources.")
    return {
        "knowledge": gathered_knowledge,
        "messages": state.get("messages", []) + [AIMessage(content=f"I've gathered some initial knowledge for the course. Found {len(gathered_knowledge)} resources.")]
    }

@traceable(name="module_organizer_node")
def module_organizer_node(state: CourseState):
    """
    Organizes the course into modules based on objectives and gathered knowledge.
    """
    if not state.get("objective") or not state.get("knowledge"):
        print("Error: Objectives or knowledge not found in state for module organization.")
        return {"messages": state.get("messages", []) + [AIMessage(content="I need course objectives and knowledge to organize modules. Something went wrong.")]}

    objective = state["objective"]
    knowledge = state["knowledge"]
    title = state["title"]
    target_audience = state["target_audience"]

    # Create LLM instructions for module organization
    module_organizer_instructions = SystemMessage(
        content=f"""You are an expert AI curriculum designer.
        Your task is to structure a course into logical modules based on the provided objectives and gathered knowledge.

        **Course Title:** {title}
        **Target Audience:** {target_audience}
        **Course Objectives:**
        {chr(10).join([f"- {obj.goal}" for obj in objective])}

        **Gathered Knowledge:**
        {chr(10).join([f"- {k.title}: {k.content[:200]}..." for k in knowledge])}

        Analyze the objectives and knowledge to define the overall module structure.
        Generate a list of modules, where each module has a number (e.g., "Module 1", "Module 2"), a concise title, and a brief topic description.
        The lessons list within each module should be empty at this stage.

        Your output MUST be ONLY a JSON object matching the `ModulesList` schema, without any additional text, explanations, or markdown formatting outside the JSON.
        Example: {{ "modules": [ {{ "number": "Module 1", "title": "Introduction", "topic": "Overview of the subject", "lessons": [] }}, {{ "number": "Module 2", "title": "Core Concepts", "topic": "Deep dive into key ideas", "lessons": [] }} ] }}
        """
    )

    structured_llm = llm_with_tools.with_structured_output(ModulesList)

    try:
        modules_instance = structured_llm.invoke([
            module_organizer_instructions,
            HumanMessage(content="Generate the module structure for the course.")
        ])
    except Exception as e:
        return {"messages": state.get("messages", []) + [AIMessage(content=f"Error during module organization: {e}")]}

    if not hasattr(modules_instance, "modules"):
         return {"messages": state.get("messages", []) + [AIMessage(content="Error: The LLM output for modules is missing the 'modules' field.")]}

    print(f"Successfully generated {len(modules_instance.modules)} modules.")
    return {
        "modules": modules_instance.modules,
        "messages": state.get("messages", []) + [AIMessage(content=f"I've organized the course into {len(modules_instance.modules)} modules.")]
    }

@traceable(name="lesson_writer")
def lesson_writer(state: CourseState):
    """ Generate lessons for each module based on course knowledge and module structure """
    objective = state["objective"]
    knowledge = state["knowledge"]
    target_audience = state["target_audience"]
    title = state["title"]
    modules = state["modules"]

    if not modules:
        print("Error: No modules found in state for lesson writing.")
        return {"messages": state.get("messages", []) + [AIMessage(content="I need the module structure to write lessons. Something went wrong.")]}

    updated_modules = []
    for module in modules:
        print(f"Generating lessons for {module.number} - {module.title}")
        structured_llm = llm_with_tools.with_structured_output(LessonsList)
        instructions = get_generate_lesson_instructions(
            module_number=module.number.split(" ")[-1], # Extract number from "Module X"
            module_title=module.title,
            module_topic=module.topic,
            course_objectives=objective,
            relevant_knowledge_base=knowledge,
            course_target_audience=target_audience,
            course_title=title
        )
        try:
            lessons_instance = structured_llm.invoke([
                instructions,
                HumanMessage(content=f"Generate lessons for {module.number} based on the provided details and knowledge.")
            ])
            module.lessons = lessons_instance.lessons
            print(f"Generated {len(module.lessons)} lessons for {module.number}.")
        except Exception as e:
            print(f"Error generating lessons for {module.number}: {e}")
            # Optionally add an error lesson or message to the module
            module.lessons = [Lesson(number=f"{module.number.split(' ')[-1]}.0", title="Error generating lessons", explanation=f"Could not generate lessons for this module due to an error: {e}", case_study="", idead="", reflection_questions="")]


        updated_modules.append(module)

    # Update the modules field in the state with the lessons added
    return {"modules": updated_modules}

summary_maker = SystemMessage(
    content="""You are an AI teacher.
Your goal is to generate a well-structured summary for the course based on the lessons. Analyze the lessons and produce a concise course summary.
"""
)

@traceable(name="finalize_course")
def finalize_course(state: CourseState):
    """ Combine course components to write a Course Summary """
    # Finalize course now uses modules with nested lessons
    modules = state["modules"]
    objective = state["objective"]
    knowledge = state["knowledge"]

    # Format lessons from modules for the summary maker
    all_lessons_content = []
    for module in modules:
        all_lessons_content.append(f"## {module.number} - {module.title}")
        for lesson in module.lessons:
            all_lessons_content.append(f"### {lesson.number} - {lesson.title}")
            all_lessons_content.append(f"Explanation: {lesson.explanation}")
            all_lessons_content.append(f"Case Study: {lesson.case_study}")
            all_lessons_content.append(f"Ideas: {lesson.idead}")
            all_lessons_content.append(f"Reflection Questions: {lesson.reflection_questions}")
            all_lessons_content.append("---") # Separator between lessons

    formatted_str_lessons = "\n\n".join(all_lessons_content)
    formatted_str_objective = "\n\n---\n\n".join([f"{item}" for item in objective])
    formatted_str_knowledge = "\n\n---\n\n".join([f"{item}" for item in knowledge])

    summary = llm_with_tools.invoke([
        summary_maker,
        HumanMessage(content=f"Write a summary for the course based on the following:\nModules and Lessons:\n{formatted_str_lessons}\n\nObjectives: {formatted_str_objective}\n\nKnowledge: {formatted_str_knowledge}")
    ])
    return {"summary": summary.content} # Extract content from AIMessage

summary_maker = SystemMessage(
    content="""You are an AI teacher.
Your goal is to generate a well-structured summary for the course based on the lessons. Analyze the lessons and produce a concise course summary.
"""
)

@traceable(name="finalize_course")
def finalize_course(state: CourseState):
    """ Combine course components to write a Course Summary """
    modules = state["modules"]
    objective = state["objective"]
    knowledge = state["knowledge"]

    formatted_str_modules = "\n\n---\n\n".join([f"{item}" for item in modules])
    formatted_str_objective = "\n\n---\n\n".join([f"{item}" for item in objective])
    formatted_str_knowledge = "\n\n---\n\n".join([f"{item}" for item in knowledge])
    summary = llm_with_tools.invoke([
        summary_maker, 
        HumanMessage(content=f"Write a summary for the course based on the following:\nModules: {formatted_str_modules}\nObjectives: {formatted_str_objective}\nKnowledge: {formatted_str_knowledge}")
    ])
    return {"summary": summary}

# System message
sys_msg = SystemMessage(content="You are a helpful assistant.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Routing
def entry_point_passthrough(state):
    """Passes the state through. Acts as a named entry for conditional routing."""
    return

def is_user_want_make_a_course(state):
    user_input = state["messages"][-1].content
    keywords = ["course", "class", "lesson", "teach", "education", "module", "curriculum"]
    if any(keyword in user_input for keyword in keywords):
        return "analyze_user_input"
    else:
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

course_maker.add_edge(START, "entry_point_passthrough")
course_maker.add_conditional_edges("entry_point_passthrough", is_user_want_make_a_course, ["assistant", "analyze_user_input"])
course_maker.add_edge("analyze_user_input", "generate_core_course")
course_maker.add_edge("generate_core_course", "course_knowledge_gatherer_node")
course_maker.add_edge("course_knowledge_gatherer_node", "module_organizer_node")
course_maker.add_edge("module_organizer_node", "lesson_writer")
course_maker.add_edge("lesson_writer", "finalize_course")
course_maker.add_edge("finalize_course", END)

graph = course_maker.compile()
