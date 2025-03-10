from langchain_groq import ChatGroq
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class TextExtractionAgent:
    """Extracts text from a document and sends it to an LLM for structured analysis."""

    def __init__(self, file_path="srd.docx"):
        """Initialize the Text Extraction Agent."""
        self.file_path = file_path

        # Load and process the document
        self.documents = self.load_document()
        print(f"✅ Loaded {len(self.documents)} documents.")

        # Split document into smaller chunks
        self.text_chunks = self.split_texts()
        print(f"✅ Split document into {len(self.text_chunks)} chunks.")

        # Initialize Groq LLM (Llama3-8B)
        self.llm = ChatGroq(model_name="llama3-8b-8192", api_key=GROQ_API_KEY)

    def load_document(self):
        """Loads the .docx document and extracts text."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"⚠️ Document not found: {self.file_path}")

        loader = Docx2txtLoader(self.file_path)
        return loader.load()

    def split_texts(self):
        """Splits the document into smaller chunks for better processing."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        return text_splitter.split_documents(self.documents)

    def retrieve_answer(self, query):
        """Sends the extracted text to the LLM and retrieves structured response in JSON format."""
        # Combine all chunks into a single document
        document_text = "\n\n".join([text.page_content for text in self.text_chunks])

        # Define structured prompt for Groq LLM
        prompt = PromptTemplate.from_template(
            """You are an AI specializing in Software Requirements Specification (SRS) analysis.

            Extract the following structured information from the given document:
            - **UI Components** (e.g., buttons, modals, navigation bars, tiles).
            - **State Management Requirements** (e.g., session handling, API integration).
            - **API Endpoints** (including method, headers, parameters, response).
            - **User Roles & Permissions** (e.g., General Users vs. Managers).
            - **Styling & Branding Guidelines** (if applicable).

            The response **must** be in **valid JSON format**.

            Example JSON output:
            ```json
            {{
                "UI_Components": ["Dashboard", "Leave Request Form", "Pods Management"],
                "State_Management": ["NgRx", "Session Storage"],
                "API_Endpoints": [
                    {{
                        "endpoint": "/api/lms/leaves/apply",
                        "method": "POST",
                        "headers": {{ "Authorization": "Bearer <token>" }},
                        "request_body": {{ "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "reason": "string" }},
                        "response": {{ "message": "Success", "status": "pending" }}
                    }}
                ],
                "User_Roles": {{
                    "General_User": ["Apply for leave", "View leave balance"],
                    "Manager": ["Approve/reject leave", "View team reports"]
                }},
                "Styling": {{
                    "Primary_Color": "#FF5733",
                    "Font": "Roboto"
                }}
            }}
            ```

            Extract and return the details in this JSON format.
            
            Context: {document_text}

            **Strictly output only JSON. Do not add any extra text.**
            """
        )


        # Generate response using Groq LLM
        formatted_prompt = prompt.format(document_text=document_text)
        response = self.llm.invoke(formatted_prompt)

        # Ensure the output is in proper JSON format
        try:
            structured_output = json.loads(response.content.strip())  # Strip any whitespace
        except json.JSONDecodeError:
            structured_output = {
                "error": "Failed to parse response into JSON",
                "raw_response": response.content
            }

        return structured_output

# Test the Enhanced Agent
if __name__ == "__main__":
    agent = TextExtractionAgent()

    print("\n🔹 Extracted Structured Data from Groq:")
    extracted_data = agent.retrieve_answer("Extract UI and API details")
    print(json.dumps(extracted_data, indent=4))