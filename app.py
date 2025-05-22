import gradio as gr
from fpdf import FPDF
import gen_agent
from gen_agent import CourseState, graph

import os

class PDF(FPDF):
    def header(self):
        # Remove header content as title is in the body
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def heading1(self, title):
        """Add a Heading 1 style (largest heading for main title)"""
        self.set_font('Arial', 'B', 16)  # Bold, size 16
        # Ensure title text is properly encoded for FPDF
        safe_title = str(title).encode('latin-1', errors='replace').decode('latin-1')
        # Center the main title
        self.cell(0, 15, safe_title, 0, 1, 'C')
        self.ln(5)  # Add space after heading

    def heading2(self, title):
        """Add a Heading 2 style (for major sections)"""
        self.set_font('Arial', 'B', 14)  # Bold, size 14
        # Ensure title text is properly encoded for FPDF
        safe_title = str(title).encode('latin-1', errors='replace').decode('latin-1')
        self.cell(0, 10, safe_title, 0, 1, 'L')
        self.ln(3)  # Add space after heading

    def heading3(self, title):
        """Add a Heading 3 style (for sub-sections)"""
        self.set_font('Arial', 'B', 12)  # Bold, size 12
        # Ensure title text is properly encoded for FPDF
        safe_title = str(title).encode('latin-1', errors='replace').decode('latin-1')
        self.cell(0, 8, safe_title, 0, 1, 'L')
        self.ln(2)  # Add space after heading

    def chapter_title(self, title):
        """Legacy method - now redirects to heading2"""
        self.heading2(title)

    def chapter_body(self, body):
        """Add standard body text"""
        self.set_font('Arial', '', 11)  # Regular, size 11
        # Ensure body is a string before encoding and handle potential encoding errors
        encoded_body = str(body).encode('latin-1', errors='replace').decode('latin-1')
        self.multi_cell(0, 8, encoded_body)
        self.ln(3)  # Space after paragraph

    def bullet_list_item(self, text, level=0):
        """Add a bullet list item with specified indentation level"""
        indent = 8 * level  # Indent based on level (0, 1, 2, etc.)
        self.set_x(10 + indent)  # Set position with indentation
        self.set_font('Arial', '', 11)
        # Use a dash instead of unicode bullet to avoid encoding issues
        self.cell(6, 6, '-', 0, 0, 'C')
        # Ensure text is properly encoded for FPDF
        safe_text = str(text).encode('latin-1', errors='replace').decode('latin-1')
        self.multi_cell(0, 6, safe_text)
        self.ln(2)  # Small space between items

    def numbered_list_item(self, number, text, level=0):
        """Add a numbered list item with specified indentation level"""
        indent = 8 * level  # Indent based on level
        self.set_x(10 + indent)  # Set position with indentation
        self.set_font('Arial', '', 11)
        # Add number and text
        self.cell(10, 6, f"{number}.", 0, 0, 'R')
        # Ensure text is properly encoded for FPDF
        safe_text = str(text).encode('latin-1', errors='replace').decode('latin-1')
        self.multi_cell(0, 6, safe_text)
        self.ln(2)  # Small space between items

    def indented_text(self, text, indent=1):
        """Add text with specified level of indentation"""
        margin = 10 * indent  # Calculate margin based on indent level
        self.set_x(margin)  # Set position with indentation
        self.set_font('Arial', '', 11)
        text_width = self.w - margin - 10  # Adjust width for right margin
        # Ensure text is properly encoded for FPDF
        safe_text = str(text).encode('latin-1', errors='replace').decode('latin-1')
        self.multi_cell(text_width, 8, safe_text)
        self.ln(2)  # Small space after

# New function: render_markdown
def render_markdown(pdf, markdown_text):
    """
    A simple Markdown renderer that supports:
      - Headings: '# ' for Heading1, '## ' for Heading2, '### ' for Heading3
      - Bullet lists: lines starting with '- '
      - Blank lines for spacing, and plain text paragraphs.
    """
    lines = markdown_text.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("### "):
            pdf.heading3(line[4:])
        elif line.startswith("## "):
            pdf.heading2(line[3:])
        elif line.startswith("# "):
            pdf.heading1(line[2:])
        elif line.startswith("- "):
            pdf.bullet_list_item(line[2:])
        elif line == "":
            pdf.ln(3)
        else:
            pdf.chapter_body(line)

def create_course_pdf(course_data, filename="generated_course.pdf"):
    """
    Generates a PDF from the course data.
    """
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Course Title (as Heading 1, centered)
    if course_data.get("title"):
        pdf.heading1(course_data.get("title", "Generated Course"))

    # Objectives (as Heading 2, with bullet list items)
    if course_data.get("objective"):
        pdf.heading2("Learning Objectives")
        objectives_list = course_data["objective"]
        if isinstance(objectives_list, list):
            for obj in objectives_list:
                # Check if obj is a Pydantic object and has the expected attributes
                if hasattr(obj, 'goal') and hasattr(obj, 'description') and hasattr(obj, 'scope'):
                    # Use bullet list for each objective
                    pdf.bullet_list_item(f"Goal: {str(obj.goal)}")
                    
                    # Description and scope as indented text
                    pdf.indented_text(f"Description: {str(obj.description)}", indent=1)
                    pdf.indented_text(f"Scope: {str(obj.scope)}", indent=1)
                    pdf.ln(3) # Extra space between objectives
                else:
                    pdf.bullet_list_item(f"Could not process objective data: {str(obj)}")
            pdf.ln(5) # Space after objectives section
        else:
             pdf.chapter_body(f"Could not process objectives list: {str(objectives_list)}")

    # Modules and Lessons
    if course_data.get("modules"):
        pdf.heading2("Course Modules")
        modules_list = course_data["modules"]
        if isinstance(modules_list, list):
            for module in modules_list:
                if hasattr(module, 'number') and hasattr(module, 'title') and hasattr(module, 'topic'):
                    pdf.heading3(f"{module.number}: {module.title}")
                    pdf.indented_text(f"Topic: {module.topic}", indent=1)
                    pdf.ln(2)

                    if hasattr(module, 'lessons') and isinstance(module.lessons, list):
                        for lesson in module.lessons:
                            if hasattr(lesson, 'number') and hasattr(lesson, 'title') and hasattr(lesson, 'explanation') and hasattr(lesson, 'case_study') and hasattr(lesson, 'idead') and hasattr(lesson, 'reflection_questions'):
                                pdf.heading3(f"{lesson.number}: {lesson.title}")
                                pdf.indented_text(f"Explanation: {lesson.explanation}", indent=2)
                                pdf.indented_text(f"Case Study: {lesson.case_study}", indent=2)
                                pdf.indented_text(f"Ideas: {lesson.idead}", indent=2)
                                pdf.indented_text(f"Reflection Questions: {lesson.reflection_questions}", indent=2)
                                pdf.ln(3)
                            else:
                                pdf.chapter_body(f"Could not process lesson data: {str(lesson)}")
                        pdf.ln(5)
                    else:
                        pdf.indented_text("No lessons in this module.", indent=1)
                    pdf.ln(5)
                else:
                    pdf.chapter_body(f"Could not process module data: {str(module)}")
        else:
            pdf.chapter_body(f"Could not process modules list: {str(modules_list)}")

    # Summary (as Heading 2)
    if course_data.get("summary"):
        pdf.heading2("Summary")
        summary_data = course_data["summary"]
        # Extract content if it's a message object, otherwise use as is
        summary_text = summary_data.content if hasattr(summary_data, 'content') else str(summary_data)
        render_markdown(pdf, summary_text)

    # Sources (without title, just bullet list)
    if course_data.get("knowledge"):
        knowledge_list = course_data["knowledge"]
        if isinstance(knowledge_list, list):
            for source in knowledge_list:
                if hasattr(source, 'title') and hasattr(source, 'source'):
                    # Format each source as a bullet point without a heading
                    pdf.bullet_list_item(f"{str(source.title)} ({str(source.source)})")
        else:
            pdf.chapter_body(f"Could not process knowledge list: {str(knowledge_list)}")


    # Save the PDF
    pdf_path = os.path.join(os.getcwd(), filename)
    pdf.output(pdf_path)
    return pdf_path


async def generate_course(prompt):
    """
    Invokes the LangGraph agent with the user's prompt, generates a PDF, and returns the PDF path.
    """
    initial_state = {"messages": [("human", prompt)]}

    try:
        result = graph.invoke(initial_state)
        # Assuming the result dictionary contains the necessary keys for PDF generation
        # Need to ensure the result is a dictionary before passing to create_course_pdf
        if isinstance(result, dict):
            # Check if the result dictionary contains an error message instead of course data
            # This is a heuristic based on previous error output
            # Check if 'messages' key exists and the last message is an error string
            if "messages" in result and isinstance(result["messages"][-1], str) and "An error occurred:" in result["messages"][-1]:
                 return f"Agent processing error: {result['messages'][-1]}"
            # Also check if the result itself is an error string (less likely with LangGraph state)
            elif isinstance(result, str) and "An error occurred:" in result:
                 return result
            else:
                print("Course data received by create_course_pdf:")
                print(result) # Add print statement to inspect data
                pdf_path = create_course_pdf(result)
                return pdf_path
        else:
            # Handle cases where the graph might return something other than a dictionary state
            # This could be an error message string from the agent itself
            return f"Unexpected output format from agent: {result}"
    except Exception as e:
        # Catch exceptions during graph invocation
        return f"An error occurred during agent invocation: {e}"

# Define the Gradio interface
# Input: Textbox for the user prompt
# Output: File component for PDF download
interface = gr.Interface(
    fn=generate_course,
    inputs=gr.Textbox(label="Enter your course generation prompt:"),
    outputs=gr.File(label="Download Generated Course PDF:"),
    title="Course Generation Agent",
    description="Enter a prompt to generate a course outline, lessons, and summary as a downloadable PDF using the LangGraph agent."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
