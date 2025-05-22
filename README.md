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
2.  You will see a single input for prompt:
3.  Enter Prompt(make sure contains e.g., course, lessons, class, modules).
4.  Click the "Generate Course" (or similar) button.
5.  The agent will process the request through the different stages. This might take a few moments.
6.  The generated course components (Objective, Knowledge snippets, Lesson, Summary) will be displayed on the page.

## ⚙️ Local Setup (Optional)

If you want to run this agent locally:

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_LINK_HERE # Replace with your Git repo link](https://github.com/Acceleratorll/course_maker_agent.git)
    cd course_maker_agent
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
    *(Ensure your `requirements.txt` file lists all necessary packages: `langchain`, `langgraph`, `langchain-ollama`, `pydantic`, `python-dotenv`, `tavily-python`, `gradio`, `langsmith` etc.)*

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
    ```bash
    python app.py
    ```
    This will start a local web server, and you can access the Gradio interface in your browser (usually at `http://127.0.0.1:7860`).

## 💡 Future Enhancements & Potential Improvements

*   Refine parsing data to PDF
*   Can Add Knowledge via user input(or choose maybe)
*   **Implement `Prerequisite` and `Homework` Generation:** Activate and integrate nodes for generating prerequisites and homework assignments into the `CourseState` and workflow.
*   **Error Handling & Resilience:** Add more robust error handling within graph nodes.

*Generated with the help of AI, fine-tuned by a human.*
