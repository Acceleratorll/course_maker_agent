import gradio as gr
import pypandoc
from gen_agent import graph
import os
import re
from datetime import datetime

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

    def _generate_objectives(self):
        objectives = self.data.get("objective")
        if not objectives: return

        # This section will not have a page break before it, allowing it to
        # follow the Table of Contents.
        self.markdown_parts.append("## Learning Objectives {-}\n")
        if isinstance(objectives, list):
            for obj in objectives:
                text = getattr(obj, 'goal', f"Could not process: {obj}")
                self.markdown_parts.append(f"* {text}\n")
        
        # Add a page break AFTER Learning Objectives
        self.markdown_parts.append("\n\\newpage\n")

    def _generate_modules(self):
        modules = self.data.get("modules")
        if not modules or not isinstance(modules, list): return
        
        for module in modules:
            mod_title = getattr(module, 'title', 'Untitled Module')
            # FIX 1: Promote Module titles to Level 1 Header (#)
            self.markdown_parts.append(f"# {mod_title}\n")
            self.markdown_parts.append(f"{getattr(module, 'achieved', '')}\n\n")
            
            lessons = getattr(module, 'lessons', [])
            if isinstance(lessons, list):
                for lesson in lessons:
                    les_title = getattr(lesson, 'title', 'Untitled Lesson')
                    # FIX 1: Promote Lesson titles to Level 2 Header (##)
                    self.markdown_parts.append(f"## {les_title}\n\n")
                    
                    content_sections = [
                        getattr(lesson, 'explanation', ''),
                        getattr(lesson, 'important_areas', ''),
                        getattr(lesson, 'case_study', '')
                    ]
                    self.markdown_parts.extend(f"{content}\n\n" for content in content_sections if content)
            
        # A final page break after all modules are done.
        # This also serves as the page break BEFORE the summary.
        self.markdown_parts.append("\n\\newpage\n")

    def _generate_summary(self):
        summary = self.data.get("summary")
        if not summary: return

        # The page break BEFORE summary is handled by the end of _generate_modules.
        self.markdown_parts.append("## Summary {-}\n")
        content = getattr(summary, 'content', str(summary))
        self.markdown_parts.append(f"{content}\n")
        
        # FIX 2: Add page break AFTER summary.
        self.markdown_parts.append("\n\\newpage\n")

    def _generate_sources(self):
        knowledge_list = self.data.get("knowledge")
        if not knowledge_list or not isinstance(knowledge_list, list): return
        
        self.markdown_parts.append("## Sources {-}\n")
        sources_found = False
        translation_table = str.maketrans('', '', '"/[]`')
        for item in knowledge_list:
            # url = getattr(item, 'source', None)
            url = item.metadata.get('url')
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

if __name__ == "__main__":
    interface.launch()