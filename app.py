import sys
import gradio as gr
import pypandoc
from gen_agent import graph
import os
import re
from datetime import datetime

class MockObject:
    """A simple class to mimic objects with attributes."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"MockObject({self.__dict__})"

def get_pandoc_config(course_title):
    """Returns the pandoc configuration for a high-quality PDF."""

    header_includes = r"""
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\rhead{""" + course_title + r"""}
\cfoot{Page \thepage\ of \pageref{LastPage}}
\usepackage{lastpage}

% --- START: Title Page Break Fix (NEW) ---
% Redefine \maketitle to automatically add a new page after it's done
\let\oldmaketitle\maketitle
\renewcommand{\maketitle}{%
  \oldmaketitle%
  \newpage%
}
% --- END: Title Page Break Fix ---

% --- START: TOC Page Break Fix ---
% Redefine \tableofcontents to automatically add a new page after it's done
\let\oldtableofcontents\tableofcontents
\renewcommand{\tableofcontents}{%
  \oldtableofcontents%
  \newpage%
}
% --- END: TOC Page Break Fix ---
"""

    pandoc_vars = {
        'title': course_title,
        'date': datetime.now().strftime("%B %d, %Y"),
        'fontsize': '11pt',
        'geometry': 'margin=1in',
        'mainfont': 'TeX Gyre Termes',   # Professional serif font (Times New Roman alternative)
        'sansfont': 'TeX Gyre Heros',     # Clean sans-serif font for headers
        'monofont': 'TeX Gyre Cursor',    # Clear monospace font for code
        'linkcolor': 'blue',              # Make hyperlinks blue and clickable
        'urlcolor': 'blue',
        'header-includes': header_includes # Inject our custom header/footer LaTeX
    }

    extra_args = [
        '--pdf-engine=xelatex',
        '--toc',                        # Generate Table of Contents
        '--toc-depth=3',                # TOC includes up to ### headers
        '--number-sections',            # Crucial for structured documents
        '--number-offset=1',            # <<< THE FIX: Start numbering from 1 instead of 0
        '--highlight-style=pygments'    # Code syntax highlighting
    ]

    # Build the final list of arguments
    for key, value in pandoc_vars.items():
        extra_args.extend(['-V', f'{key}={value}'])

    return extra_args

# --- Part 2: Tidy and Structured Markdown Generation ---
# This class handles the conversion of your course data into clean Markdown.
# --- Part 2: Tidy and Structured Markdown Generation (Corrected Version) ---
# This class handles the conversion of your course data into clean Markdown.
class MarkdownCourseGenerator:
    """A class to generate structured Markdown from course data."""
    def __init__(self, course_data):
        self.data = course_data
        self.markdown_parts = []
        self.extracted_sources = []
        self.header_regex = re.compile(r'^\s*#{1,6}\s')

    def _escape_latex_special_chars(self, text):
        """
        Escapes LaTeX special characters in a given string in the correct order.
        """
        # Ensure text is a string to prevent errors
        if not isinstance(text, str):
            text = str(text)
            
        replacements = {
            '\\': r'\textbackslash{}',
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
        }
        
        text = text.replace('\\', replacements['\\'])
        
        for char, escaped in replacements.items():
            if char != '\\':
                text = text.replace(char, escaped)
        return text

    def _process_content_with_headers(self, content):
        """
        NEW: Processes content line by line. It preserves Markdown headers
        and applies LaTeX escaping to all other lines.
        """
        if not content:
            return ""

        processed_lines = []
        for line in content.split('\n'):
            # Check if the start of the line matches the header pattern
            if self.header_regex.match(line):
                # This is a header, so add it without escaping
                processed_lines.append(line)
            else:
                # This is regular content, so escape it
                escaped_line = self._escape_latex_special_chars(line)
                processed_lines.append(escaped_line)
        
        return "\n".join(processed_lines)

    def _generate_objectives(self):
        objectives = self.data.get("objective")
        if not objectives: return

        self.markdown_parts.append("## Learning Objectives {-}\n")
        if isinstance(objectives, list):
            for obj in objectives:
                # Use the new processing function here as well, just in case
                goal_text = getattr(obj, 'goal', f"Could not process: {obj}")
                processed_text = self._process_content_with_headers(goal_text)
                self.markdown_parts.append(f"* {processed_text}\n")
        
        self.markdown_parts.append("\n\\newpage\n")

    def _generate_modules(self):
        modules = self.data.get("modules")
        if not modules or not isinstance(modules, list): return
        
        for module in modules:
            mod_title = getattr(module, 'title', 'Untitled Module')
            self.markdown_parts.append(f"# {mod_title}\n")
            
            achieved_text = getattr(module, 'achieved', '')
            if achieved_text:
                # Process this text too
                processed_achieved = self._process_content_with_headers(achieved_text)
                self.markdown_parts.append(f"{processed_achieved}\n\n")
            
            lessons = getattr(module, 'lessons', [])
            if isinstance(lessons, list):
                for lesson in lessons:
                    les_title = getattr(lesson, 'title', 'Untitled Lesson')
                    self.markdown_parts.append(f"## {les_title}\n\n")
                    
                    content_sections = [
                        getattr(lesson, 'explanation', ''),
                        getattr(lesson, 'important_areas', ''),
                        getattr(lesson, 'case_study', '')
                    ]
                    for content in content_sections:
                        # THE FIX: Use the new intelligent processing function
                        processed_content = self._process_content_with_headers(content)
                        self.markdown_parts.append(f"{processed_content}\n\n")

        self.markdown_parts.append("\n\\newpage\n")

    def _generate_summary(self):
        summary = self.data.get("summary")
        if not summary: return

        self.markdown_parts.append("## Summary {-}\n")
        content = getattr(summary, 'content', str(summary))
        # Use the new processing function on the summary content
        processed_content = self._process_content_with_headers(content)
        self.markdown_parts.append(f"{processed_content}\n")
        
        self.markdown_parts.append("\n\\newpage\n")

    def _generate_sources(self):
        # This function deals with URLs, so it doesn't need the header processing logic
        knowledge_list = self.data.get("knowledge")
        if not knowledge_list or not isinstance(knowledge_list, list): return
        
        self.markdown_parts.append("## Sources {-}\n")
        # ... (rest of your _generate_sources function is fine and does not need changes)
        sources_found = False
        translation_table = str.maketrans('', '', '"/[]`')
        flat_documents = []
        if knowledge_list:
            for item in knowledge_list:
                if isinstance(item, list):
                    flat_documents.extend(item)
                else:
                    flat_documents.append(item)
        
        for doc in flat_documents:
            if hasattr(doc, 'metadata') and doc.metadata:
                url = doc.metadata.get('url')
                if url and url.lower().translate(translation_table).strip() not in {'n/a', 'unknown', 'url_not_provided', 'not specified', 'no url provided', '', 'source url not provided in context'}:
                    self.markdown_parts.append(f"* <{url}>\n")
                    self.extracted_sources.append({"url": url})
                    sources_found = True
        
        if not sources_found:
            self.markdown_parts.append("* No external sources were cited for this document.\n")
        self.markdown_parts.append("\n")

    def generate(self):
        """Generates the full Markdown string and returns it."""
        self._generate_objectives()
        self._generate_modules()
        self._generate_summary()
        self._generate_sources()
        return "".join(self.markdown_parts)

# --- Part 3: The Main PDF Creation Function (Now clean and tidy) ---
def create_course_pdf(course_data, filename_suffix=".pdf"):
    """
    Orchestrates the generation of a high-quality PDF from course data.
    """
    course_title = course_data.get("title", "Generated Course")

    # 1. Generate clean Markdown content using our dedicated class
    generator = MarkdownCourseGenerator(course_data)
    full_markdown = generator.generate()
    extracted_sources = generator.extracted_sources

    # 2. Get the professional Pandoc configuration
    extra_args = get_pandoc_config(course_title)

    # 3. Define a safe and descriptive filename
    safe_title = re.sub(r'[\\/*?:"<>|]', "", course_title).replace(" ", "_")
    pdf_path = os.path.join(os.getcwd(), f"{safe_title}{filename_suffix}")

    # 4. Convert to PDF
    try:
        pypandoc.convert_text(
            full_markdown,
            'pdf',
            format='markdown+raw_tex+smart',
            outputfile=pdf_path,
            extra_args=extra_args
        )
    except Exception as e:
        print(f"Error building PDF document with pypandoc: {e}")
        if "xelatex not found" in str(e):
            raise RuntimeError(
                "Failed to build PDF. The 'xelatex' engine was not found. "
                "Please ensure it is installed as part of your LaTeX distribution (e.g., MiKTeX, TeX Live)."
            )
        raise RuntimeError(f"Failed to build PDF with pypandoc: {e}")

    return pdf_path, extracted_sources

# --- Part 4: Gradio Interface (Unchanged, but now calls the improved function) ---
async def generate_course(prompt):
    """
    Invokes the LangGraph agent, generates a PDF, and returns the path.
    """
    initial_state = {"messages": [("human", prompt)]}
    try:
        # Assuming `graph.invoke` returns the full state dictionary
        final_state = graph.invoke(initial_state)
        
        # Check for errors returned from the agent
        last_message = final_state.get("messages", [])[-1]
        if isinstance(last_message, tuple) and "error" in last_message[0].lower():
             return None, f"Agent processing error: {last_message[1]}"
        if isinstance(last_message, str) and "error" in last_message.lower():
             return None, f"Agent processing error: {last_message}"

        # The agent's final output is now in the state, not just returned
        course_data_for_pdf = final_state
        pdf_path, extracted_sources = create_course_pdf(course_data_for_pdf)
        print("Extracted Sources:", extracted_sources)
        return pdf_path, "PDF generated successfully!"

    except Exception as e:
        # Catch errors from both agent invocation and PDF creation
        return None, f"A critical error occurred: {e}"

interface = gr.Interface(
    fn=generate_course,
    inputs=gr.Textbox(label="Enter your course generation prompt:"),
    outputs=[
        gr.File(label="Download Generated Course PDF:"),
        gr.Textbox(label="Status Message:")
    ],
    title="Course Generation Agent",
    description="Enter a prompt to generate a course outline, lessons, and summary as a downloadable PDF using the LangGraph agent.\n\nPrompt Example:\n('make a course of ai agent concepts (use recent reference like open ai paper, anthropic paper, google paper, etc) for web developer')"
)

def run_test():
    """
    Runs a test of the PDF generation logic using sample data.
    This bypasses the Gradio UI and the AI agent.
    """
    print("--- Running PDF Generation Test ---")

    # 1. Create mock course data that mimics the AI agent's output
    sample_course_data = {
        "title": "My Awesome Test Course",
        "objective": [
            MockObject(goal="Learn how to test code effectively."),
            MockObject(goal="Understand LaTeX special characters like #, $, and %."),
        ],
        "modules": [
            MockObject(
                title="Module 1: Getting Started",
                achieved="Introduction to testing.",
                lessons=[
                    MockObject(
                        title="Lesson 1.1: The Basics",
                        explanation="""This is the main content. It includes special characters that need escaping: #, $, %, &, _, {, }, ~, ^, \\.
### What Makes a System Truly Intelligent?

Have you ever wondered how a smart thermostat knows when to turn on the heat, or how a chatbot seems to understand your questions and provide helpful answers? It's not just magic; it's the fundamental concept of an **AI Agent** at work. In this lesson, we'll peel back the layers to understand what an AI agent is, its core components, and the continuous cycle that enables its intelligent behavior.

### Why This Matters: Your Foundation in AI

Understanding AI agents isn't just academic; it's the bedrock for comprehending nearly every advanced AI system you'll encounter. From self-driving cars to personalized recommendation engines, the underlying principles of how these systems perceive their surroundings and make decisions are rooted in the AI agent model. Grasping these fundamentals will empower you to analyze, design, and even troubleshoot AI solutions in the real world.
                        """,
                        important_areas="Focus on the `_escape_latex_special_chars` function.",
                        case_study="A case study of a successful test."
                    )
                ]
            )
        ],
        "summary": MockObject(content="This is the course summary."),
        "knowledge": [
            [MockObject(metadata={'url': 'https://www.example.com/source1'})],
            MockObject(metadata={'url': 'https://www.example.com/source2'})
        ]
    }
    
    try:
        # 1. Call the PDF creation function directly
        print("Generating test PDF...")
        # Make sure to provide a valid path for saving the test file
        test_output_dir = "D:/Assets/Project/langgraph/course_maker_agent/test"
        os.makedirs(test_output_dir, exist_ok=True) # Create the directory if it doesn't exist
        
        pdf_path, sources = create_course_pdf(sample_course_data, filename_suffix="_TEST.pdf")
        
        # Move the file to your desired test directory
        final_pdf_path = os.path.join(test_output_dir, os.path.basename(pdf_path))
        os.rename(pdf_path, final_pdf_path)
        pdf_path = final_pdf_path

        # 2. Verify the output
        if os.path.exists(pdf_path):
            print(f"✅ SUCCESS: PDF created at '{pdf_path}'")
            print(f"✅ SUCCESS: Extracted {len(sources)} sources.")
        else:
            print(f"❌ FAILURE: PDF file was not created.")
            
    except Exception as e:
        print(f"❌ FAILURE: An error occurred during the test: {e}")
    # finally:
    #     # 4. Clean up the generated file
    #     if pdf_path and os.path.exists(pdf_path):
    #         os.remove(pdf_path)
    #         print(f"--- Cleaned up test file: '{pdf_path}' ---")


if __name__ == "__main__":
    # Check if the '--test' argument was passed
    if "--test" in sys.argv:
        run_test()
    else:
        # If no '--test' argument, launch the Gradio interface
        print("--- Launching Gradio Interface ---")
        interface.launch()