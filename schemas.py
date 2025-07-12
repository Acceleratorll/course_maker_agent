import json
from typing import Union, List, Dict, Optional, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field, validator
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

class SearchQueries(BaseModel):
    semantic_queries: List[str] = Field(description="A list of full-sentence, natural language queries designed to capture the user's intent for semantic search.")
    keyword_queries: List[str] = Field(description="A list of short, concise keyword-based queries ideal for lexical search or database lookups.")
    web_queries: List[str] = Field(default_factory=list, description="A list of queries optimized for web search engines, often combining keywords with operators or common search phrases.")

class IdentifyKnowledge(BaseModel):
    is_sufficient: bool = Field(default_factory=bool)
    reasoning: str = Field(default_factory=str)
    confidence_score: float = Field(default_factory=float)
    identified_gaps: List[str] = Field(default_factory=list)

class DocumentMetadata(BaseModel):
    title: Optional[str] = Field(None, description="The title of the document or source.")
    url: Optional[str] = Field(None, description="The URL of the document source.")
    summary: Optional[str] = Field(None, description="A brief summary of the document content.")

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
    # Add other potential metadata fields as needed
    @validator('metadata', pre=True)
    def parse_metadata(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {"content": value}  # Fallback
        return value

class DataState(TypedDict):
    query: str
    goal: str
    desired_focus: str
    target_audience: str
    search_queries: SearchQueries
    is_sufficient: bool
    documents: List[Documents]

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
    knowledge: List[Documents] = Field(default_factory=list, description="The knowledge of the lesson")
    summary: str = Field(description="A summary of the course")
    description: str = Field(description="A detailed description of the course")
    semantic_queries: List[str] = Field(description="A list of full-sentence, natural language queries designed to capture the user's intent for semantic search.")
    keyword_queries: List[str] = Field(description="A list of short, concise keyword-based queries ideal for lexical search or database lookups.")
    web_queries: List[str] = Field(default_factory=list, description="A list of queries optimized for web search engines, often combining keywords with operators or common search phrases.")
    is_sufficient: bool = Field(default_factory=bool)
    identify_knowledge: IdentifyKnowledge
    # search_queries: SearchQueries

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
    desired_focus: Optional[str] = Field(None, description="Specific aspects, keywords, or angles to focus on or prioritize in the search. Helps narrow down results.")
    context: Optional[str] = Field(None, a="Brief context about who this information is for, if it influences the type or depth of information needed (e.g., 'for a beginner', 'technical expert').")
    goal: Optional[str] = Field(None, description="What will this information be used for? (e.g., 'to answer a specific user question', 'to get a general overview', 'to find supporting data').")
