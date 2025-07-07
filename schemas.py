from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.documents import Document
from langgraph.graph import MessagesState

# --- Input and State Schemas ---

class TargetAudience(BaseModel):
    id: str = Field(description="Unique identifier for the target audience")
    age_range: Optional[str] = Field(None, description="The age range of the target audience")
    experience_level: Optional[str] = Field(None, description="The experience level of the target audience (e.g., beginner, intermediate)")
    prior_knowledge: Optional[str] = Field(None, description="Specific prior knowledge the audience is expected to have")
    interests: Optional[str] = Field(None, description="Interests of the target audience relevant to the course")
    learning_style: Optional[str] = Field(None, description="Preferred learning style of the audience (e.g., visual, hands-on)")
    goals: str = Field(description="The goals of the target audience")
    pain_points: Optional[str] = Field(None, description="Pain points or challenges the audience faces")
    demographics: Optional[str] = Field(None, description="Demographic information of the target audience")

class Objective(BaseModel):
    id: str = Field(description="Unique identifier for the objective")
    goal: str = Field(description="The learning goal of the objective")
    description: str = Field(description="The description of the goal")
    scope: str = Field(description="The scope of the goal")

class ObjectivesList(BaseModel):
    id: str = Field(description="Unique identifier for the list of objectives")
    objectives: List[Objective] = Field(description="A list of learning objectives")

class Knowledge(BaseModel):
    id: str = Field(description="Unique identifier for the knowledge entry")
    title: str = Field(description="The title of the knowledge source")
    source: str = Field(description="The source of the knowledge")
    content: str = Field(description="The content of the gathered knowledge")
    relevance_score: Optional[float] = Field(None, description="An optional score indicating relevance if provided by the search source.")

class Lesson(BaseModel):
    number: str = Field(description="The number of the lesson(e.g., Lesson 1.1). format:module[number].lesson[number]")
    title: str = Field(description="The title of the lesson")
    explanation: Optional[str] = Field(description="The detail explanation of the lesson")
    case_study: Optional[str] = Field(description="The case study of the lesson")
    idea: Optional[str] = Field(description="Ideas for simple interactive exercises (even text-based ones initially)")
    goal: str = Field(description="The goal or main learning outcome of the lesson")
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
    goal: str = Field(description="What you get after completing the module")
    lessons: List[Lesson] = Field(description="The lessons in the module")

class ModulesList(BaseModel):
    modules: List[Modules]

class CourseState(MessagesState):
    id: str = Field(description="Unique identifier for the course state")
    title: str = Field(description="The title of the course")
    subject: str = Field(description="The subject of the course")
    language: str = Field(description="The language of the course")
    added_details: Optional[str] = Field(None, description="Additional details or special requests for the course")
    target_audience: TargetAudience = Field(description="The target audience of the course")
    user_input: str = Field(description="The user input or command")
    prerequisites: List[Prerequisite] = Field(description="The prerequisites of the course")
    objective: List[Objective] = Field(default_factory=list, description="The objective of the course")
    modules: List[Modules] = Field(description="The modules of the course")
    knowledge: List[Document] = Field(default_factory=list, description="The knowledge of the lesson")
    summary: str = Field(description="A summary of the course")
    description: str = Field(description="A detailed description of the course")

class SearchQuery(BaseModel):
    id: str = Field(description="Unique identifier for the search query")
    search_query: str = Field(None, description="Search query for retrieval.")

class UserInputAnalysis(BaseModel):
    id: str = Field(description="Unique identifier for the user input analysis")
    title: str = Field(description="The title of the course, derived from user input")
    subject: str = Field(description="The subject of the course, derived from user input")
    target_audience: TargetAudience = Field(description="The target audience profile, derived from user input")
    user_goal: Optional[str] = Field(None, description="The user's primary goal for the course")
    added_details: Optional[str] = Field(None, description="Additional details from the user's request")
    language: str = Field(description="The language for the course, derived from user input")
    
class SearchInput(BaseModel):
    id: str = Field(description="Unique identifier for the search input")
    research_need: str = Field(description="The primary topic, question, or information need that requires web research. This is the core of the search.")
    desired_focus: Optional[str] = Field(None, description="Specific aspects, keywords, or angles to focus on or prioritize in the search. Helps narrow down results.")
    context: Optional[str] = Field(None, a="Brief context about who this information is for, if it influences the type or depth of information needed (e.g., 'for a beginner', 'technical expert').")
    goal: Optional[str] = Field(None, description="What will this information be used for? (e.g., 'to answer a specific user question', 'to get a general overview', 'to find supporting data').")
    