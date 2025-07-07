import os
from dotenv import load_dotenv
from typing import List

# --- Langchain & Supabase Imports ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from pydantic import BaseModel, Field
from supabase.client import Client, create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = os.getenv("TABLE")
FUNCTION_NAME = os.getenv("FUNCTION")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class DocumentMetadata(BaseModel):
    """Schema for LLM-generated metadata for a document chunk."""
    title: str = Field(..., description="A concise, descriptive title for the text chunk.")
    url: str = Field(..., description="Source URL of the text chunk.")
    content: str = Field(..., description="The refined text content of the chunk.")
    mentions: List[str] = Field(default_factory=list, description="List of named entities or acronyms mentioned.")
    related_to: List[str] = Field(default_factory=list, description="List of broader concepts or topics.")

class SupabaseVectorManager:
    """
    A manager class to handle CRUD operations for a Supabase Vector Store
    using Langchain, with content sourced from Tavily Search.
    """
    def __init__(self):
        """
        Initializes all necessary components: Embeddings, Tools, Splitter,
        Supabase client, and the Vector Store itself.
        """
        print("Initializing Supabase Vector Manager...")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.llm_with_structured_output = self.llm.with_structured_output(DocumentMetadata)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_query")
        self.search_tool = TavilySearchResults(max_results=3)
        self.metadata_prompt_template = ChatPromptTemplate.from_template(
             # Your existing prompt template...
             """As an expert data analyst, your task is to carefully read the following text chunk and generate structured metadata for it.
            Your output will be used to create a high-quality entry in a vector database for later retrieval.

            **Context Chunk to Analyze:**
            ---
            {context}
            ---

            **Instructions for Generating Fields:**

            1.  **title**: Create a concise, descriptive title specifically for THIS text chunk, not the entire original article. The title should capture the main point of the text.
            2.  **content**: Your primary task. Refine the original text chunk into a clear, well-structured paragraph. Correct any grammatical errors and improve the flow, but **do not add new information or hallucinate facts**. The goal is a high-quality, self-contained version of the original content.
            3.  **mentions**: Identify and list any specific named entities (e.g., "OpenAI", "GPT-4", "LangChain", "Transformer architecture") or important acronyms (e.g., "RAG", "LLM") mentioned in the text. If no specific entities are mentioned, return an empty list.
            4.  **related_to**: Identify and list 3 to 5 broader concepts or topics that this text is about. These should be useful for categorizing the document (e.g., "Artificial Intelligence", "Natural Language Processing", "Machine Learning Techniques").

            Generate the structured output based *only* on the provided context chunk and these instructions.
            """
        )
        self.generate_metadata = self.metadata_prompt_template | self.llm_with_structured_output
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
        )
        self.supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # This vector_store object is now mainly for utility, like deleting.
        # Adding will be done via direct client upsert, and searching via RPC.
        self.vector_store = SupabaseVectorStore(
            client=self.supabase_client,
            embedding=self.embeddings,
            table_name=TABLE_NAME,
            query_name=FUNCTION_NAME # A placeholder, we won't use it for hybrid search
        )
        print("Initialization complete.")

    def create_documents_from_search(self, topic: str) -> List[int]:
        print(f"\n--- [CREATE] Fetching and processing content for topic: '{topic}' ---")
        
        search_results = self.search_tool.invoke(topic)
        if not search_results:
            print("No results found from Tavily search.")
            return []

        rows_to_insert = []
        for result in search_results:
            if not result.get("content"):
                continue
            
            content_chunks = self.text_splitter.split_text(result["content"])
            for chunk in content_chunks:
                try:
                    llm_metadata = self.generate_metadata.invoke({"context": chunk})
                    llm_metadata_dict = llm_metadata.model_dump()
                    
                    embedding = self.embeddings.embed_query(llm_metadata_dict["content"])
                    
                    row = {
                        "content": llm_metadata_dict["content"],
                        "embedding": embedding,
                        "metadata": { # This will be stored in your JSONB metadata column
                            "title": llm_metadata_dict.get('title', result.get('title', "No Title")),
                            "url": llm_metadata_dict.get('url', result.get('url', 'no_url_provided')),
                            "source": "tavily_search",
                            "llm_generated": {
                                "mentions": llm_metadata_dict.get("mentions", []),
                                "related_to": llm_metadata_dict.get("related_to", [])
                            }
                        }
                    }
                    rows_to_insert.append(row)
                except Exception as e:
                    print(f"Warning: Could not process a chunk. Error: {e}")
        
        if not rows_to_insert:
            print("No document chunks could be processed for insertion.")
            return []

        print(f"Adding {len(rows_to_insert)} document chunks to Supabase table '{TABLE_NAME}'...")
        response = self.supabase_client.from_(TABLE_NAME).upsert(rows_to_insert).execute()
        
        if response.data:
            added_ids = [item['id'] for item in response.data]
            print(f"Successfully added documents. IDs: {added_ids}")
            return added_ids
        else:
            print(f"Error inserting documents: {response.error or 'No data returned'}")
            return []

    def perform_hybrid_search(self, q: str, k: int = 15) -> List[Document]:
        """
        Performs a hybrid search by calling the custom 'hybrid_search' RPC function
        in Supabase, combining full-text and semantic search.
        """
        print(f"\n--- [HYBRID SEARCH] Searching for documents related to: '{q}' ---")

        # 1. Embed the query text
        query_vector = self.embeddings.embed_query(q)

        # 2. Define the parameters for the RPC call
        rpc_params = {
            "query_text": q,
            "query_embedding": query_vector,
            "match_count": k,
        }

        # 3. Call the RPC function
        print(f"Executing '{FUNCTION_NAME}' RPC function in Supabase...")
        response = self.supabase_client.rpc(FUNCTION_NAME, rpc_params).execute()

        if response.data:
            # 4. Convert the raw dictionary results into LangChain Document objects
            matched_docs = [
                Document(
                    page_content=row.get("content", ""),
                    metadata={
                         "id": row.get("id"),
                         **(row.get("metadata") if isinstance(row.get("metadata"), dict) else {})
                    },
                )
                for row in response.data
            ]
            print(f"Successfully found {len(matched_docs)} documents.")
            return matched_docs
        else:
            print(f"Error performing hybrid search: {getattr(response, 'error', 'No data returned')}")
            return []
    
    def delete_documents(self, doc_ids: List[int]):
        """Deletes documents by their integer IDs."""
        if not doc_ids:
            print("No document IDs provided for deletion.")
            return
        
        print(f"\n--- [DELETE] Deleting {len(doc_ids)} documents ---")
        # Use the base client for deletion
        self.supabase_client.from_(TABLE_NAME).delete().in_("id", doc_ids).execute()
        print("Deletion command executed successfully.")
        
class DatabaseManager:
    """
    A manager class for handling generic CRUD operations with a Supabase table.
    This class provides a robust, error-handling layer over the Supabase-py client.
    """
    def __init__(self):
        """
        Initializes the DatabaseManager and the Supabase client.
        """
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("DatabaseManager initialized.")

    def create_data(self, table_name: str, data: any) -> List[dict]:
        """
        Inserts one or more rows into a specified table.

        Args:
            table_name: The name of the table to insert data into.
            data: A dictionary for a single row or a list of dictionaries for multiple rows.

        Returns:
            A list of dictionaries representing the inserted data.

        Raises:
            ValueError: If the insertion fails.
        """
        record_count = len(data) if isinstance(data, list) else 1
        print(f"Attempting to insert {record_count} record(s) into '{table_name}'...")
        try:
            response = self.client.from_(table_name).insert(data).execute()
            if response.error:
                raise ValueError(f"API Error: {response.error.message}")
            print(f"Successfully inserted {len(response.data)} record(s).")
            return response.data
        except Exception as e:
            print(f"Error creating data in '{table_name}': {e}")
            raise ValueError(f"Could not create data: {e}")

    def read_data(self, table_name: str, select: str = "*", filters: dict = None, limit: int = 100) -> List[dict]:
        """
        Reads data from a table with optional filtering and limits.

        Args:
            table_name: The name of the table to read from.
            select: The columns to select (e.g., "*", "id, title").
            filters: A dictionary of filters to apply (e.g., {"column_name": "value"}).
            limit: The maximum number of rows to return.

        Returns:
            A list of dictionaries representing the query results.
        
        Raises:
            ValueError: If the read operation fails.
        """
        print(f"Reading data from '{table_name}' with select='{select}' and limit={limit}.")
        try:
            query = self.client.from_(table_name).select(select).limit(limit)
            if filters:
                for column, value in filters.items():
                    query = query.eq(column, value)
            
            response = query.execute()
            if response.error:
                raise ValueError(f"API Error: {response.error.message}")
            print(f"Successfully retrieved {len(response.data)} record(s).")
            return response.data
        except Exception as e:
            print(f"Error reading data from '{table_name}': {e}")
            raise ValueError(f"Could not read data: {e}")

    def update_data(self, table_name: str, record_id: any, data: dict) -> dict:
        """
        Updates a specific record in a table by its ID.

        Args:
            table_name: The name of the table to update.
            record_id: The ID of the record to update.
            data: A dictionary containing the data to update.

        Returns:
            A dictionary representing the updated record.

        Raises:
            ValueError: If the update fails or no record is updated.
        """
        print(f"Updating record '{record_id}' in table '{table_name}'...")
        try:
            response = self.client.from_(table_name).update(data).eq("id", record_id).execute()
            if response.error:
                raise ValueError(f"API Error: {response.error.message}")
            if not response.data:
                raise ValueError(f"Update failed: No record found with ID '{record_id}'.")
            print(f"Successfully updated record '{record_id}'.")
            return response.data[0]
        except Exception as e:
            print(f"Error updating data in '{table_name}': {e}")
            raise ValueError(f"Could not update data: {e}")

    def delete_data(self, table_name: str, record_ids: any) -> List[dict]:
        """
        Deletes one or more records from a table by their IDs.

        Args:
            table_name: The name of the table to delete from.
            record_ids: A single ID or a list of IDs to delete.

        Returns:
            A list of dictionaries representing the deleted records.

        Raises:
            ValueError: If the deletion fails.
        """
        if not isinstance(record_ids, list):
            record_ids = [record_ids]
        
        print(f"Attempting to delete {len(record_ids)} record(s) from '{table_name}'...")
        try:
            response = self.client.from_(table_name).delete().in_("id", record_ids).execute()
            if response.error:
                raise ValueError(f"API Error: {response.error.message}")
            print(f"Successfully deleted {len(response.data)} record(s).")
            return response.data
        except Exception as e:
            print(f"Error deleting data from '{table_name}': {e}")
            raise ValueError(f"Could not delete data: {e}")