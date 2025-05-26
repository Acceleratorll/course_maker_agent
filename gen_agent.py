import os

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Annotated, List, Optional, TypedDict
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import traceable
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode

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
    explanation: Optional[str] = Field(description="The detail explanation of the lesson")
    case_study: Optional[str] = Field(description="The case study of the lesson")
    idea: Optional[str] = Field(description="Ideas for simple interactive exercises (even text-based ones initially)")
    reflection_questions: Optional[str] = Field(description="Reflection questions for the lesson")
    
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
    achieved: str = Field(description="What you get after completing the module")
    lessons: List[Lesson] = Field(description="The lessons in the module")

class ModulesList(BaseModel):
    modules: List[Modules]

class CourseState(MessagesState):
    title: str
    subject: str
    language: str
    added_details: Optional[str] = None
    target_audience: TargetAudience = Field(description="The target audience of the course")
    user_input: str = Field(description="The user input or command")
    prerequisites: List[Prerequisite] = Field(description="The prerequisites of the course")
    objective: List[Objective] = Field(default_factory=list, description="The objective of the course")
    modules: List[Modules] = Field(default_factory=list, description="The modules of the course")
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
    language: str

load_dotenv()

# Uncomment or choose your desired LLM
# llm = ChatOllama(model="qwen3:4b", temperature=0.3)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

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
    subject, title, target_audience, user_goal, language and added_details.
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
            "6.  **language**: Language used for the course"
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
        "language": result.language,
        "target_audience": result.target_audience,
        "added_details": result.added_details
    }

@traceable(name="generate_objective")
def get_generate_objective_instructions(subject, title, language, target_audience, added_details: Optional[str] = None):
    target_audience_str = str(target_audience)
    return SystemMessage(
        content=f"""You are an expert AI instructional designer with a specialization in curriculum development.
        Your primary mission is to brainstorm the Course Details and then, craft a list of clear, compelling, and actionable learning objectives for a course.

        **Course Details to Brainstorm:**
        1.  **Course Subject:** "{subject}"
        2.  **Course Title:** "{title}"
        3.  **Target Audience Profile:** {target_audience_str}
        4.  **Added Details:** "{added_details}"
        5.  **Main Language:** "{language}"

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

    return {"objective": objectives_instance.objectives}

def get_generate_lesson_instructions(language, course_title, module_number, module_title, module_achieved, relevant_knowledge_base, course_target_audience, lesson):
    return SystemMessage(
        content=f"""You are an AI Expert Writer Educational Content Creator.
        Your primary mission is to write detailed, engaging, and effective lesson content for a specific module within a larger course.
        Crucially, prioritize the integration of active learning strategies and practical application throughout the lessons.

        Phase 1: Understand the Course and Module Context

            1. Overall Course Title: "{course_title}"
            2. Target Audience Profile: {course_target_audience}
            3. Main Language: {language}
            4. Module Number: {module_number}
            5. Module Title: "{module_title}"
            6. Module Learning Outcome (Achieved): "{module_achieved}"
            7. Relevant Knowledge Base: ```{relevant_knowledge_base}``` - Utilize this information to inform and enrich the lesson content. Synthesize key points and examples from this knowledge base.
            8. Lesson Title: "{lesson.title}"
            9. Lesson Number: "{lesson.number}"

        Phase 2: Write Detailed Content for Lesson "{lesson.title}" (Number: {lesson.number})

        Based on the context above, write the content for this lesson, ensuring it is clear, accurate, and engaging for the "{course_target_audience}".

        The lesson content MUST be a JSON object with the following fields:

            1.  "explanation": string - Detailed, well-written content explaining the lesson's concepts. This should be the core instructional material, tailored to the target audience's level of understanding and prior knowledge.
                - For beginner audiences, use analogies, real-world examples, and avoid technical jargon.
                - For expert audiences, use precise language, technical details, and assume a high level of prior knowledge.
            2.  "case_study": string (Optional) - A relevant, real-world case study, detailed example, or practical scenario that directly illustrates and helps learners apply the concepts taught in the "explanation". This should provide concrete context and demonstrate the practical relevance of the lesson. If a case study is not applicable or necessary for a specific lesson, provide an empty string or null. Consider varying the format or complexity of case studies across lessons to maintain engagement.
            3.  "idea": string (Optional) - A concrete idea for a simple interactive exercise or activity related to the lesson content. This could be:
                - A discussion prompt
                - A small problem to solve
                - A mini-quiz idea
                - A hands-on task (even if text-based initially)
                - A coding challenge
                - A design exercise
                - A role-playing scenario
                Focus on activities that promote active engagement and practical application of the concepts. If not applicable, provide an empty string or null.
            4.  "reflection_questions": string - 1-3 thought-provoking questions designed to prompt learners to reflect on the lesson's content, connect it to their own experience, or think critically about the topic. If not applicable, provide an empty string or null.

        Output Constraints:
        - Do NOT include any additional text, explanations, or markdown formatting outside of the JSON object.
        - Ensure the content within the JSON is well-formatted and readable.
        """
    )

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
    Organizes the course into modules and lessons based on objectives, target audience and gathered knowledge.
    """
    if not state.get("objective") or not state.get("knowledge"):
        print("Error: Objectives or knowledge not found in state for module organization.")
        return {"messages": state.get("messages", []) + [AIMessage(content="I need course objectives and knowledge to organize modules. Something went wrong.")]}

    objective = state["objective"]
    knowledge = state["knowledge"]
    title = state["title"]
    target_audience = state["target_audience"]
    language = state["language"]

    # Create LLM instructions for module organization
    module_organizer_instructions = SystemMessage(
        content=f"""You are an expert AI curriculum designer.
        Your task is to structure a comprehensive course into logical, progressive, and easy-to-understand modules and lessons. Base the structure on the provided course objectives and synthesize the gathered knowledge to inform the content and flow.

        **Course Main Language:** {language}
        **Course Title:** {title}
        **Target Audience:** {target_audience}
        **Course Objectives:**
        {chr(10).join([f"- {obj.goal}" for obj in objective])}

        **Gathered Knowledge:**
        {chr(10).join([f"- {k.title}: {k.content[:200]}..." for k in knowledge])}

        Analyze the course objectives and the gathered knowledge to define a logical overall module structure. Ensure the modules build upon each other progressively.

        Generate a list of modules. For each module:
        - Provide a `number` (e.g., "Module 1", "Module 2").
        - Provide a concise and descriptive `title` that clearly indicates the module's main topic.
        - Provide an `achieved` field: A brief summary (1-2 sentences) of the key learning outcomes or skills a learner will gain specifically from completing this module. This should align with the overall course objectives.
        - Generate a list of `lessons` within the module. For each lesson:
            - Provide a `number` that includes the module number (e.g., "1.1", "1.2", "2.1").
            - Provide a concise and informative `title` that reflects the lesson's specific topic.
            - Only fill 'number' and 'title' fields.'

        Ensure the lessons within each module follow a logical sequence and contribute to achieving the module's learning outcomes.

        Your output MUST be ONLY a JSON object matching the `ModulesList` schema, without any additional text, explanations, or markdown formatting outside the JSON.
        Example: {{ "modules": [
        {{ "number": "Module 1", "title": "Introduction to AI Agents", "achieved": "Understand the fundamental concepts and applications of AI agents.", "lessons": [{{"number": "1.1", "title": "What are AI Agents?"}}, {{"number": "1.2", "title": "History and Evolution"}}] }},
        {{ "number": "Module 2", "title": "Core Agent Architectures", "achieved": "Identify and differentiate between various core architectures used in AI agent design.", "lessons": [{{"number": "2.1", "title": "Deliberative Agents"}}, {{"number": "2.2", "title": "Reactive Agents"}}] }}
        ]}}
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
                module_achieved=module.achieved,
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

    # Update the modules field in the state with the lessons added
    return {"modules": updated_modules}

summary_maker = SystemMessage(
    content="""You are an AI writer skilled in crafting concise and impactful course conclusions.

Your task is to generate a brief, narrative summary that serves as a conclusion for the entire course. This summary should synthesize the core knowledge and skills presented across all modules, providing a compact overview of what the learner has achieved and can now do.

Follow these guidelines to create the concluding summary:

1.  **Distill Core Concepts:** Review the provided course modules and identify the most essential concepts, skills, and overall journey the learner undertakes.
2.  **Weave a Narrative:** Combine these distilled points into a cohesive paragraph or two. The summary should flow as a narrative, not a list or a simple concatenation of module topics. It should tell a story of the learner's progression or the knowledge built.
3.  **Highlight Overall Value & Transformation:** Emphasize the overall transformation, key capabilities, or understanding the learner gains from the entire course journey. What is the main takeaway or the ultimate benefit of completing all modules?
4.  **Be Conclusive:** Frame the summary as a look back at what has been covered and achieved, providing a sense of closure and completeness.

The summary should be:

*   **Concise and Compact:** Aim for a focused piece of text (e.g., 150-250 words, or adjust as needed) that captures the essence of the course.
*   **Holistic:** Reflect the integrated knowledge from all modules, rather than a module-by-module breakdown.
*   **Outcome-Oriented:** Focus on the final understanding, abilities, or perspective the learner has gained.
*   **Engaging and Coherent:** Provide a smooth, readable, and insightful conclusion.

**Important Instructions:**

*   Do **not** break down the summary by explicitly listing individual modules or their titles.
*   Do **not** include separate sections like "Target Audience" or "Key Learning Objectives" as distinct bullet points or paragraphs. If these aspects are crucial, weave them naturally into the concluding narrative.
*   The output should be **only the summary text itself**. Do not include any additional text, explanations, headers, or markdown formatting that isn't part of the narrative flow of the conclusion.
"""
)

@traceable(name="finalize_course")
def finalize_course(state: CourseState):
    """ Combine course components to write a Course Summary """
    modules = state["modules"]
    language = state["language"]

    formatted_str_modules = "\n\n---\n\n".join([f"{item}" for item in modules])
    
    summary_request_content = f"Write a concluding summary for the course based on the following:\nModules: {formatted_str_modules}\nLanguage: {language}\nAdhere strictly to the instructions provided in the system message for crafting a concluding summary."
    
    llm_response = llm_with_tools.invoke([
        summary_maker, 
        HumanMessage(content=summary_request_content)
    ])
    return {"summary": llm_response.content}

# System message
sys_msg = SystemMessage(content="You are a helpful ai assistant with tools in your arsenal.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Routing
def entry_point_passthrough(state):
    """Passes the state through. Acts as a named entry for conditional routing."""
    return

def is_user_want_make_a_course(state):
    user_input = state["messages"][-1].content
    instruction = SystemMessage(content="""You are an AI decider. 
                                Your purpose is to decide the next node to take. is user want to make a course?
                                The output should be:
                                'assistant' if the user dont want to make a course
                                'analyze_user_input' if the user want to make a course
                                Dont add any additional text, explanations, or markdown formatting.
                               """)
    result = llm_with_tools.invoke([instruction, HumanMessage(content=user_input)])
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