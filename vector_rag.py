import json
import sys
from typing import List

from dotenv import load_dotenv

from vector_store_manager import SupabaseVectorManager

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily.tavily_crawl import TavilyCrawl
from langgraph.graph import END, StateGraph
from pydantic import validator
from schemas import(
    CourseState, UserInputAnalysis, SearchQueries, IdentifyKnowledge
)
from prompts import(
    knowledge_gap_prompt_template, rag_prompt_template,
    sufficiency_prompt_template, web_query_generation_prompt, keyword_query_generation_prompt,
    semantic_query_generation_prompt
)

load_dotenv()

crawler = TavilyCrawl(max_depth=1)
manager = SupabaseVectorManager()

# llm = ChatOllama(model="qwen3:0.6b")
# llm = ChatOllama(model="gemma3n:e2b", temperature=0.3)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)


structured_llm = llm.with_structured_output(SearchQueries)
identify_llm = llm.with_structured_output(IdentifyKnowledge)

sufficiency_chain = sufficiency_prompt_template | identify_llm
web_query_chain = web_query_generation_prompt | structured_llm
keyword_query_chain = keyword_query_generation_prompt | structured_llm
semantic_query_chain = semantic_query_generation_prompt | structured_llm
knowledge_gap_chain = knowledge_gap_prompt_template | structured_llm
rag_chain = rag_prompt_template | llm | StrOutputParser()

def initial_retrieve_node(state):
    print("---NODE: Initial Retrieve---")
    
    context = {
          "title": state["title"],
          "subject": state["subject"],
          "target_audience": state["target_audience"],
          "user_goal": state["objective"],
          "added_details": state["added_details"],
          "language": state["language"]
      }

    keyword_result = keyword_query_chain.invoke({"context": context})
    semantic_result = semantic_query_chain.invoke({"context": context})
    
    keyword_list = keyword_result.keyword_queries
    semantic_list = semantic_result.semantic_queries

    print("Keyword Queries: ", keyword_list)
    print("Semantic Queries: ", semantic_list)

    keyword_queries = " ".join(keyword_list) if keyword_list else ""
    
    all_documents = []
    seen_doc_ids = set()
    
    custom_q = "how to structure online course modules and lessons effectively"
    k_q = "designing engaging structure online course lessons effectively"
    print(f"\nPerform Search for: '{custom_q}' with keyword queries: '{k_q}'")
    documents = manager.perform_hybrid_search(q=custom_q, k_q=k_q, k=10)
    print(f"Retrieved {len(documents)} documents after performing search.")
    for item in documents:
        if isinstance(item, list):
            all_documents.extend(item)
        else:
            all_documents.append(item)
    # for doc in documents:
        # if doc.metadata.get('id') not in seen_doc_ids:
        #     all_documents.append(doc)
        #     seen_doc_ids.add(doc.metadata['id'])

    for query in semantic_list:
        print(f"\nPerform Search for: '{query}' with keyword queries: '{keyword_queries}'")
        documents = manager.perform_hybrid_search(q=query, k_q=keyword_queries, k=10)
        print(f"Retrieved {len(documents)} documents after performing search.")
        all_documents.append(documents)
        for item in documents:
            if isinstance(item, list):
                all_documents.extend(item)
            else:
                all_documents.append(item)
        # for doc in documents:
        #     if doc.metadata.get('id') not in seen_doc_ids:
        #         all_documents.append(doc)
        #         seen_doc_ids.add(doc.metadata['id'])

    print(f"Retrieved {len(all_documents)} documents initially.")
    return {"knowledge": all_documents, "keyword_queries":keyword_result, "semantic_queries":semantic_list}

def check_sufficiency_node(state):
    print("---NODE: Check Sufficiency---")
    title = state["title"]
    target_audience = state["target_audience"]
    objective = state["objective"]
    documents_from_state = state["knowledge"]
    desired_focus = state["added_details"]

    flat_documents = []
    for item in documents_from_state:
        if isinstance(item, list):
            flat_documents.extend(item)
        else:
            flat_documents.append(item)
            
    context_str = "\n\n".join([doc.page_content for doc in flat_documents])
    response = sufficiency_chain.invoke({
        "course_title": title,
        "target_audience": target_audience,
        "learning_objectives": objective,
        "context": context_str,
        "details": desired_focus
    })
    print(f"Sufficiency check LLM response: '{response}'")
    
    return {"identify_knowledge": response, "is_sufficient": response.is_sufficient}

def knowledge_planner_node(state):
    print("---NODE: Knowledge Planner---")
    
    search_queries = knowledge_gap_chain.invoke({
            "course_title": state["title"], 
            "learning_objectives": state["objective"], 
            "target_audience": state["target_audience"], 
            "identified_gaps": state["identify_knowledge"].identified_gaps,
            "details": state["added_details"],
        })

    print("Web Queries: ",search_queries.web_queries)
    return {"web_queries": search_queries.web_queries}

def gather_and_process_node(state):
    print("---NODE: Gather and Process New Knowledge---")
    
    if not state["web_queries"]:
        print("No new search queries planned. Skipping gathering.")
        # No new docs, so documents state remains unchanged from previous retrieval
        return {"documents": state["documents"]} 
    
    for search_query in state["web_queries"]:
        print(f"Gathering for: '{search_query}'...")
        manager.create_documents_from_search(search_query)
        
    print("Gathering complete.")
        
# --- Conditional Edges ---

def decide_to_augment_or_answer(state):
    print(f"---DECISION POINT: Sufficiency check result: {state['is_sufficient']}---")
    sufficient_value = state["is_sufficient"]
    if sufficient_value is True or (isinstance(sufficient_value, str) and sufficient_value.lower() == "true"):
        print("Sufficient, go directly to answer generation.")
        return "END"  # Sufficient, go directly to answer generation
    elif sufficient_value is False:
        print("Insufficient, plan how to augment.")
        return "plan_knowledge_augmentation" # Insufficient, plan how to augment
    else:
        print(f"is_sufficient is null")
        sys.exit("is_sufficient is null")

workflow = StateGraph(CourseState)

workflow.add_node("initial_retriever", initial_retrieve_node)
workflow.add_node("check_sufficiency", check_sufficiency_node)
workflow.add_node("plan_knowledge_augmentation", knowledge_planner_node)
workflow.add_node("gather_and_process", gather_and_process_node)

# Set the entry point
workflow.set_entry_point("initial_retriever")
workflow.add_edge("initial_retriever", "check_sufficiency")

workflow.add_conditional_edges(
    "check_sufficiency",
    decide_to_augment_or_answer,
    {
        "END" : END,
        "plan_knowledge_augmentation" : "plan_knowledge_augmentation"
    }
)

workflow.add_edge("plan_knowledge_augmentation", "gather_and_process")
workflow.add_edge("gather_and_process", "initial_retriever")
graph = workflow.compile()

if __name__ == "__main__":
    # result = graph.invoke({
    #     "user_input": "make simplest course (1 module 1 lesson only) of homeworkout",
    #     "title": "Simple Home Workout Guide",
    #     "subject": "Home Workout",
    #     "language": "English",
    #     "added_details": "Course must be extremely simple, consisting of only one module and one lesson.",
    #     "target_audience": {
    #         "id": "audience_homeworkout_001",
    #         "age_range": "Adults",
    #         "experience_level": "Beginner",
    #         "interests": "Fitness, health, convenience of home exercise",
    #         "goals": "To start exercising at home, to learn a basic and effective home workout routine, to improve general fitness without a gym",
    #         "pain_points": "Lack of time, no gym access, intimidation by complex workout routines, not knowing how to start exercising",
    #         "demographics": "Individuals seeking an easy entry point into fitness from home"
    #     },
    #     "objective": [
    #     {
    #         "id": "hw_obj_01",
    #         "goal": "Identify the essential components of a safe and effective home workout.",
    #         "description": "This objective is vital because it equips you with the foundational knowledge to understand what makes a workout effective and how to stay safe, preventing injury and maximizing your results from home.",
    #         "scope": "This includes recognizing the importance of warm-up, main exercises (bodyweight), and cool-down. It excludes detailed physiological explanations, advanced exercise science, or complex equipment use."
    #     },
    #     {
    #         "id": "hw_obj_02",
    #         "goal": "Execute a basic, full-body home workout routine with correct form.",
    #         "description": "Mastering this objective will enable you to confidently perform a practical workout routine independently, directly addressing your goal of starting to exercise effectively from the comfort of your home.",
    #         "scope": "This involves demonstrating proper form for foundational bodyweight exercises such as squats, modified push-ups, and planks. It does not cover advanced variations, progressive overload techniques, or specific muscle group targeting."
    #     }
    #     ]
    # })
    
    result = graph.invoke({
        "user_input": "Buatkan buku pembelajaran 'Bahasa Inggris' untuk anak SD kelas 6 yang mudah untuk dipelajari dan mencakup materi-materi yang akan di ujikan pada 'Ujian Nasional 2026'",
        "title": "Buku Pembelajaran Bahasa Inggris untuk SD Kelas 6: Persiapan Ujian Nasional 2026",
        "subject": "English Language",
        "language": "Indonesian",
        "added_details": "The book must be easy to learn and cover all material to be tested in the 'Ujian Nasional 2026' (National Exam 2026).",
        "target_audience": {
            "id": "audience-001",
            "age_range": "11-12 years old",
            "experience_level": "Intermediate for elementary students",
            "prior_knowledge": "Basic English vocabulary and grammar suitable for elementary school",
            "interests": "Engaging activities, stories, and practical language use",
            "learning_style": "Easy to understand, potentially visual and interactive methods",
            "goals": "Master English concepts required for the Ujian Nasional 2026 and achieve proficiency in basic English communication.",
            "pain_points": "Difficulty with English, stress of preparing for a national exam",
            "demographics": "6th-grade elementary school students in Indonesia"
        },
        "objective": [
            {
            "id": "obj-001",
            "goal": "Mengidentifikasi dan menggunakan struktur tata bahasa dasar serta kosakata penting dalam konteks yang relevan.",
            "description": "Tujuan ini penting agar kamu bisa membangun fondasi bahasa Inggris yang kuat, membantu memahami kalimat dan percakapan sehari-hari, serta menjawab soal ujian dengan tepat.",
            "scope": "Fokus pada penggunaan tenses sederhana (Present Simple, Past Simple, Future Simple), parts of speech (kata benda, kerja, sifat), dan kosakata umum yang sering muncul dalam ujian nasional dan kehidupan sehari-hari. Tidak termasuk tenses kompleks atau kosakata akademik yang jarang digunakan di tingkat dasar."
            },
            {
            "id": "obj-002",
            "goal": "Menganalisis dan menemukan informasi spesifik serta ide pokok dari berbagai jenis teks pendek.",
            "description": "Menguasai ini akan membantumu memahami cerita, pengumuman, atau surat pendek dalam bahasa Inggris, yang sangat penting untuk menjawab soal pemahaman bacaan di Ujian Nasional.",
            "scope": "Meliputi teks deskriptif, naratif sederhana, dan pengumuman singkat. Pembelajar akan fokus pada identifikasi informasi detail, makna kata dari konteks, dan tujuan teks. Tidak mencakup analisis teks sastra kompleks atau esai panjang."
            },
            {
            "id": "obj-003",
            "goal": "Menyusun kalimat dan paragraf sederhana untuk tujuan komunikasi dasar serta menerapkan strategi pengerjaan soal Ujian Nasional.",
            "description": "Ini akan memberimu kemampuan untuk mengungkapkan pikiranmu dalam tulisan sederhana dan mempersiapkanmu secara mental untuk menghadapi format soal ujian, mengurangi kecemasan saat Ujian Nasional tiba.",
            "scope": "Meliputi penulisan kalimat deskriptif dan naratif sangat sederhana, serta mengisi bagian rumpang. Fokus juga pada pengenalan jenis-jenis soal Ujian Nasional (pilihan ganda, esai singkat) dan teknik menjawabnya secara efektif. Tidak termasuk penulisan esai formal atau debat."
            }
        ]
    })
    
    print(result["knowledge"])