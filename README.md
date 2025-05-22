# AI Course Generation Agent 🤖📚

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Tyullix14/course_maker)
<!-- Optional: Add a GitHub repo link if you have one -->
<!-- [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-lightgrey)](YOUR_GITHUB_REPO_LINK_HERE) -->

This project is an AI-powered agent that automatically generates structured course content based on a given topic and target audience. It leverages **LangGraph** to orchestrate a series of steps involving Large Language Models (LLMs) and web search to create objectives, gather knowledge, write lessons, and summarize the course.

The agent is deployed on Hugging Face Spaces with a **Gradio** frontend for easy interaction.

## ✨ Features

*   **Automated Objective Creation:** Generates clear learning objectives based on the course topic and target audience.
*   **Web-Powered Knowledge Retrieval:** Searches the web (via Tavily Search) for relevant and up-to-date information related to the course objective.
*   **Structured Lesson Generation:** Creates detailed lessons including:
    *   Title
    *   Introduction
    *   Benefits
    *   Core Content
    *   Conclusion
    *   Description
    *   Actionable Tips
*   **Course Summarization:** Provides a concise summary of the generated course content.
*   **Modular & Stateful Workflow:** Built with LangGraph for a clear, debuggable, and extensible multi-step process.
*   **Interactive Frontend:** Easy-to-use Gradio interface hosted on Hugging Face Spaces.

## 🚀 Live Demo

You can try out the AI Course Generation Agent live on Hugging Face Spaces:
[➡️ Click here to access the Demo](https://huggingface.co/spaces/Tyullix14/course_maker)

## 🛠️ How It Works (The Agent's Workflow)

The agent uses LangGraph to define and execute a stateful graph. Each node in the graph represents a step in the course generation process:

1.  **`generate_core_course` (Objective Definition):**
    *   **Input:** Course topic, target audience.
    *   **Action:** Uses an LLM (Google `gemini-2.5-flash-preview-04-17`) to define the primary `Objective` of the course.
    *   **Output:** An `Objective` object.

2.  **`search_web` (Knowledge Gathering):**
    *   **Input:** The generated `Objective` and course topic.
    *   **Action:**
        *   Generates a search query using the LLM based on the objective.
        *   Uses Tavily Search API to find relevant web documents.
        *   Formats the search results into a `Knowledge` structure (though in the current code, it returns a list of formatted strings; this could be enhanced to populate the `Knowledge` Pydantic model more directly).
    *   **Output:** A list of knowledge snippets/documents.

3.  **`lesson_writer` (Lesson Creation):**
    *   **Input:** Course topic, `Objective`, and gathered `Knowledge`.
    *   **Action:** Instructs the LLM to write a comprehensive `Lesson` based on the provided inputs.
    *   **Output:** A `Lesson` object.

4.  **`finalize_course` (Course Summary):**
    *   **Input:** The generated `Lesson`(s), `Objective`, and `Knowledge`.
    *   **Action:** The LLM creates a final `Summary` for the course.
    *   **Output:** The course summary string.

The `CourseState` (a Pydantic `BaseModel`) holds all the information as it's generated and passed between these nodes.

## 💻 Technology Stack

*   **Orchestration:** LangGraph
*   **LLM Interaction & Core Logic:** LangChain
*   **Large Language Model (LLM):** Google Gemini
*   **Web Search:** Tavily Search API
*   **Data Validation & Structuring:** Pydantic
*   **Frontend:** Gradio
*   **Deployment:** Hugging Face Spaces
*   **Language:** Python
*   **Environment Management:** `python-dotenv`
*   **Observability (Optional):** LangSmith (configured via `@traceable`)

## 📖 Usage (Gradio Interface)

1.  Navigate to the [Hugging Face Space](https://huggingface.co/spaces/Tyullix14/course_maker).
2.  You will see input fields for:
    *   **Course Topic:** (e.g., "Introduction to Quantum Computing", "Sustainable Gardening Basics")
    *   **Target Audience:** (e.g., "Beginners with no prior knowledge", "Software developers looking to upskill")
    *   *(Your Gradio app might have a "User Input" field as well, clarify its purpose if it's actively used)*
3.  Fill in the required information.
4.  Click the "Generate Course" (or similar) button.
5.  The agent will process the request through the different stages. This might take a few moments.
6.  The generated course components (Objective, Knowledge snippets, Lesson, Summary) will be displayed on the page.

## ⚙️ Local Setup (Optional)

If you want to run this agent locally:

1.  **Clone the repository:**
    ```bash
    git clone YOUR_REPOSITORY_LINK_HERE # Replace with your Git repo link
    cd YOUR_REPOSITORY_DIRECTORY
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure your `requirements.txt` file lists all necessary packages: `langchain`, `langgraph`, `langchain-openai`, `langchain-ollama`, `pydantic`, `python-dotenv`, `tavily-python`, `gradio`, `langsmith` etc.)*

4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory of the project and add your API keys:
    ```env
    TAVILY_API_KEY="your_tavily_api_key"
    GOOGLE_API_KEY="your-api-key"

    # Optional: For LangSmith Tracing (highly recommended for debugging LangGraph)
    # LANGCHAIN_TRACING_V2="true"
    # LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    # LANGCHAIN_API_KEY="your_langsmith_api_key"
    # LANGCHAIN_PROJECT="your_project_name_in_langsmith"
    ```
5.  **Run the Gradio Application:**
    Assuming your main Gradio script is `app.py` (or similar):
    ```bash
    python app.py
    ```
    This will typically start a local web server, and you can access the Gradio interface in your browser (usually at `http://127.0.0.1:7860`).

## 💡 Future Enhancements & Potential Improvements

*   **Full `Knowledge` Model Integration:** Ensure the `search_web` node populates the `Knowledge` Pydantic model completely (title, source, content) for better data structuring.
*   **Implement `Prerequisite` and `Homework` Generation:** Activate and integrate nodes for generating prerequisites and homework assignments into the `CourseState` and workflow.
*   **Iterative Refinement:** Allow for user feedback at intermediate steps to refine content (e.g., after objective generation, suggest edits).
*   **Multiple Lessons:** Extend the graph to generate multiple lessons for a course, perhaps based on sub-topics derived from the main objective.
*   **More Knowledge Sources:** Integrate other knowledge sources like Wikipedia, ArXiv, or even uploaded documents.
*   **Error Handling & Resilience:** Add more robust error handling within graph nodes.
*   **User Input Analysis:** Fully implement the `analyze_user_input` node to allow for more complex or nuanced course requests.
*   **Model Selection:** Allow users to select different Ollama models or even other LLM providers through the Gradio interface.

*Generated with the help of AI, fine-tuned by a human.*
