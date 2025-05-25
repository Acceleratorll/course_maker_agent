import gradio as gr
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT # Import TA_CENTER
from gen_agent import CourseState, graph
import os

def create_course_pdf(course_data, filename="generated_course.pdf"):
    """
    Generates a PDF from the course data using ReportLab.
    """
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=60, bottomMargin=18)
    styles = getSampleStyleSheet()
    style_heading1 = styles['Heading1']
    style_heading2 = styles['Heading2']
    style_heading3 = styles['Heading3']
    style_body = styles['BodyText']

    style_body.firstLineIndent = 18 # Example: 18 points for indent
    style_body.alignment = TA_JUSTIFY
    style_body.fontSize = 11

    # Style with no first line indent (for bullet list and sources)
    style_no_indent = ParagraphStyle(
        'NoIndent',
        parent=style_body,
        alignment=TA_LEFT,
        firstLineIndent=0
    )

    # Modify heading styles
    style_heading1.alignment = TA_CENTER # Center Heading 1
    style_heading1.italic = False # Make Heading 1 non-italic
    style_heading2.italic = False # Make Heading 2 non-italic
    style_heading3.italic = False # Make Heading 3 non-italic


    elements = []
    
    # Course Title
    if course_data.get("title"):
        elements.append(Paragraph(course_data.get("title", "Generated Course"), style_heading1))
        elements.append(Spacer(1, 12))
    
    # Objectives
    if course_data.get("objective"):
        elements.append(Paragraph("Learning Objectives", style_heading2))
        elements.append(Spacer(1, 12))
        objectives_list = course_data["objective"]
        if isinstance(objectives_list, list):
            objective_items = [] # List to hold text for bullet points
            for obj in objectives_list:
                if hasattr(obj, 'goal'):
                    objective_items.append(str(obj.goal)) # Add objective goal to the list
                else:
                    objective_items.append(f"Could not process objective data: {str(obj)}") # Add error message to the list

            if objective_items: # Only add the list if there are items
                elements.append(Spacer(1, 12))
                # Create a bulleted list from the collected items
                bullet_list = ListFlowable(
                    [Paragraph(item, style_no_indent) for item in objective_items],
                    bulletType='bullet', # Use bullet points
                    # Add indentation if needed (adjust leftIndent and bulletIndent)
                    leftIndent=20,
                    bulletIndent=10
                )
                elements.append(bullet_list)
                elements.append(Spacer(1, 12)) # Space after the list
        else:
            elements.append(Paragraph(f"Could not process objectives list: {str(objectives_list)}", style_body))
            elements.append(Spacer(1, 12))
    
    # Modules and Lessons
    if course_data.get("modules"):
        modules_list = course_data["modules"]
        if isinstance(modules_list, list):
            for module in modules_list:
                if hasattr(module, 'number') and hasattr(module, 'title') and hasattr(module, 'topic'):
                    mod_title = f"{module.number}: {module.title}"
                    elements.append(Paragraph(mod_title, style_heading2))
                    elements.append(Spacer(1, 12))
                    elements.append(Paragraph(str(module.topic), style_body))
                    elements.append(Spacer(1, 12))
                    if hasattr(module, 'lessons') and isinstance(module.lessons, list):
                        for lesson in module.lessons:
                            if (hasattr(lesson, 'number') and hasattr(lesson, 'title') and 
                                hasattr(lesson, 'explanation') and hasattr(lesson, 'reflection_questions')):
                                les_title = f"{lesson.number}: {lesson.title}"
                                elements.append(Paragraph(les_title, style_heading3))
                                elements.append(Spacer(1, 12))
                                # Markdown parsing: * for bullets, ** for bold
                                import re
                                explanation_raw = str(lesson.explanation)
                                # Replace **text** with <b>text</b>
                                explanation_raw = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", explanation_raw)
                                # Split into lines
                                lines = [line.strip() for line in explanation_raw.split('\n') if line.strip()]
                                bullet_items = []
                                normal_paragraphs = []
                                for line in lines:
                                    if line.startswith('* '):
                                        bullet_items.append(Paragraph(line[2:], style_no_indent))
                                    else:
                                        normal_paragraphs.append(Paragraph(line, style_body))
                                if bullet_items:
                                    bullet_list = ListFlowable(
                                        bullet_items,
                                        bulletType='bullet',
                                        leftIndent=20,
                                        bulletIndent=10
                                    )
                                    elements.append(bullet_list)
                                for para in normal_paragraphs:
                                    elements.append(para)
                                if getattr(lesson, 'case_study', None):
                                    elements.append(Spacer(1, 12))
                            else:
                                elements.append(Paragraph(f"Could not process lesson data: {str(lesson)}", style_body))
                        elements.append(Spacer(1, 12))
                    else:
                        elements.append(Paragraph("No lessons in this module.", style_body))
                        elements.append(Spacer(1, 12))
                else:
                    elements.append(Paragraph(f"Could not process module data: {str(module)}", style_body))
        else:
            elements.append(Paragraph(f"Could not process modules list: {str(modules_list)}", style_body))
    
    # Summary
    if course_data.get("summary"):
        elements.append(PageBreak())
        elements.append(Paragraph("Summary", style_heading2))
        elements.append(Spacer(1, 12))
        summary_data = course_data["summary"]
        summary_text = summary_data.content if hasattr(summary_data, 'content') else str(summary_data)
        import re
        summary_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", summary_text)
        lines = [line.strip() for line in summary_text.split('\n') if line.strip()]
        bullet_items = []
        normal_paragraphs = []
        for line in lines:
            if line.startswith('* '):
                bullet_items.append(Paragraph(line[2:], style_no_indent))
            else:
                normal_paragraphs.append(Paragraph(line, style_body))
        if bullet_items:
            bullet_list = ListFlowable(
                bullet_items,
                bulletType='bullet',
                leftIndent=20,
                bulletIndent=10
            )
            elements.append(bullet_list)
        for para in normal_paragraphs:
            elements.append(para)
        elements.append(Spacer(1, 12))
    
    # Sources
    if course_data.get("knowledge"):
        knowledge_list = course_data["knowledge"]
        if isinstance(knowledge_list, list):
            elements.append(PageBreak())
            elements.append(Paragraph("Sources", style_heading2))
            elements.append(Spacer(1, 12))
            for source in knowledge_list:
                if hasattr(source, 'title') and hasattr(source, 'source'):
                    text = f"{str(source.title)} ({str(source.source)})"
                    elements.append(Paragraph(text, style_no_indent))
        else:
            elements.append(Paragraph(f"Could not process knowledge list: {str(knowledge_list)}", style_no_indent))
    
    doc.build(elements)
    pdf_path = os.path.join(os.getcwd(), filename)
    return pdf_path

async def generate_course(prompt):
    """
    Invokes the LangGraph agent with the user's prompt, generates a PDF using ReportLab, and returns the PDF path.
    """
    initial_state = {"messages": [("human", prompt)]}
    try:
        result = graph.invoke(initial_state)
        if isinstance(result, dict):
            if ("messages" in result 
                and isinstance(result["messages"][-1], str) 
                and "An error occurred:" in result["messages"][-1]):
                return f"Agent processing error: {result['messages'][-1]}"
            elif isinstance(result, str) and "An error occurred:" in result:
                return result
            else:
                print("Course data received by create_course_pdf:")
                print(result)
                pdf_path = create_course_pdf(result)
                return pdf_path
        else:
            return f"Unexpected output format from agent: {result}"
    except Exception as e:
        return f"An error occurred during agent invocation: {e}"

interface = gr.Interface(
    fn=generate_course,
    inputs=gr.Textbox(label="Enter your course generation prompt:"),
    outputs=gr.File(label="Download Generated Course PDF:"),
    title="Course Generation Agent",
    description="Enter a prompt to generate a course outline, lessons, and summary as a downloadable PDF using the LangGraph agent.\n\nPrompt Example:\n('make a course of ai agent concepts (use recent reference like open ai paper, anthropic paper, google paper, etc) for web developer')"
)

if __name__ == "__main__":
    interface.launch()
