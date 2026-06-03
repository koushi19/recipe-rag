import json
import os
import argparse
import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from langchain_google_vertexai import ChatVertexAI
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import torch
from PIL import Image

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

class RecipeRAG:
    def __init__(
        self,
        json_dir: str,
        db_dir: str = str(Path(__file__).parent.parent / "data" / "recipe_db_local"),
        llm_model: str = "llama3.2",
        retrieval_k: int = 5,
    ):
        self.json_dir = Path(json_dir)
        self.db_dir = db_dir
        self.llm_model = llm_model
        self.retrieval_k = retrieval_k

        # Set up Google Application Credentials for Vertex AI
        creds_path = Path(__file__).parent.parent / "gcp_credentials.json"
        if creds_path.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

        self._embeddings = None  # lazy-loaded on first access
        self.vector_store = None


    @property
    def embeddings(self):
        """Lazy-load CLIP model on first access instead of at startup."""
        if self._embeddings is None:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self._embeddings = OpenCLIPEmbeddings(
                    model_name="ViT-B-32",
                    checkpoint="openai"
                )
        return self._embeddings

    @staticmethod
    def _to_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [v.strip() for v in value.split("\n") if v.strip()]
        return [str(value).strip()]

    @staticmethod
    def _get_first(data: dict, keys, default="N/A"):
        for key in keys:
            value = data.get(key)
            if value not in (None, ""):
                return value
        return default

    def load_recipes(self):
        """Convert JSON files into LangChain Documents."""
        documents = []
        if not self.json_dir.exists():
            print(f"Error: Folder {self.json_dir} not found.")
            return []

        json_files = list(self.json_dir.glob("*.json"))
        print(f"Loading {len(json_files)} recipes...")
        
        for file_path in json_files:
            with open(file_path, "r") as f:
                data = json.load(f)

            ingredients = self._to_list(data.get("ingredients"))
            instructions = self._to_list(data.get("instructions"))
            servings = self._get_first(data, ["serving_size", "servings", "serves"])
            prep_time = self._get_first(data, ["prep_time", "preparation_time", "time"])
            calories = self._get_first(data, ["kcal", "calories", "nutrition_kcal"])
            
            content = f"""
Recipe Name: {data.get('name')}
Serving Size: {servings}
Prep Time: {prep_time}
Nutrition (Calories): {calories}
Ingredients: {', '.join(ingredients)}
Instructions: {' '.join(instructions)}
        """.strip()

            metadata = {
                "source": data.get('url', 'N/A'),
                "name": data.get('name', 'Unknown'),
                "serving_size": str(servings),
                "prep_time": str(prep_time),
                "calories": str(calories),
                "image_path": str(Path(self.json_dir).parent / data.get('image_path')) if data.get('image_path') else None
            }
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def build_index(self):
        """Build or load the local vector database."""
        if os.path.exists(self.db_dir) and len(os.listdir(self.db_dir)) > 0:
            print("Loading existing local index...")
            self.vector_store = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
        else:
            print("Creating new multimodal local index...")
            docs = self.load_recipes()
            if not docs: return
            
            # index text documents
            print(f"Indexing {len(docs)} text documents...")
            self.vector_store = Chroma.from_documents(
                documents=docs, 
                embedding=self.embeddings, 
                persist_directory=self.db_dir
            )
            
            # index image documents
            image_docs = []
            image_paths = []
            valid_docs = []
            
            print("Preparing image documents...")
            for doc in docs:
                img_path = doc.metadata.get("image_path")
                if img_path and os.path.exists(img_path):
                    image_paths.append(img_path)
                    valid_docs.append(doc)
            
            if image_paths:
                print(f"Generating embeddings for {len(image_paths)} images...")
                # clip image embeddings
                img_embeddings = self.embeddings.embed_image(image_paths)
                
                for i, doc in enumerate(valid_docs):
                    img_doc = Document(
                        page_content=f"Image of {doc.metadata.get('name')}",
                        metadata={**doc.metadata, "type": "image"}
                    )
                    image_docs.append(img_doc)
                
                # add image embeddings to the vector store
                self.vector_store.add_texts(
                    texts=[d.page_content for d in image_docs],
                    metadatas=[d.metadata for d in image_docs],
                    embeddings=img_embeddings
                )
            
            print("Index created successfully.")

    def query(self, question: str, image_b64: str = None, history: list = None):
        if not self.vector_store:
            self.build_index()
        if not self.vector_store:
            return {"result": "I couldn't load any recipes. Please check the JSON folder path."}
        
        question = question or "What is this food? Describe it and suggest similar recipes."
        search_query = question

        if image_b64:
            import base64
            import tempfile
            # save base64 image to a temp file so CLIP can read it
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_img.write(base64.b64decode(image_b64))
            temp_img.close()
            query_image_path = temp_img.name

            print("Generating image embedding for retrieval...")
            query_embedding = self.embeddings.embed_image([query_image_path])[0]
            os.unlink(query_image_path)

            # use image embedding to find relevant documents
            print("Searching vector store using image embedding...")
            docs = self.vector_store.similarity_search_by_vector(query_embedding, k=self.retrieval_k)
        else:
            print(f"Searching vector store using text: {question}")
            retriever = self.vector_store.as_retriever(search_kwargs={"k": self.retrieval_k})
            docs = retriever.invoke(question)

        # Direct recipe filename match fallback
        import re
        matched_recipe_doc = None
        match = re.search(r"'(.*?)'", question)
        if match:
            recipe_name = match.group(1).strip()
            print(f"Extracted recipe name from query: '{recipe_name}'")
            found_file = None
            if self.json_dir.exists():
                for file_path in self.json_dir.glob("*.json"):
                    if file_path.stem.lower() == recipe_name.lower():
                        found_file = file_path
                        break
            if found_file:
                print(f"Direct matching JSON file found: {found_file.name}")
                try:
                    with open(found_file, "r") as f:
                        data = json.load(f)
                    
                    ingredients = self._to_list(data.get("ingredients"))
                    instructions = self._to_list(data.get("instructions"))
                    servings = self._get_first(data, ["serving_size", "servings", "serves"])
                    prep_time = self._get_first(data, ["prep_time", "preparation_time", "time"])
                    calories = self._get_first(data, ["kcal", "calories", "nutrition_kcal"])
                    
                    content = f"""
Recipe Name: {data.get('name')}
Serving Size: {servings}
Prep Time: {prep_time}
Nutrition (Calories): {calories}
Ingredients: {', '.join(ingredients)}
Instructions: {' '.join(instructions)}
                    """.strip()

                    metadata = {
                        "source": data.get('url', 'N/A'),
                        "name": data.get('name', 'Unknown'),
                        "serving_size": str(servings),
                        "prep_time": str(prep_time),
                        "calories": str(calories),
                        "image_path": str(Path(self.json_dir).parent / data.get('image_path')) if data.get('image_path') else None
                    }
                    matched_recipe_doc = Document(page_content=content, metadata=metadata)
                except Exception as e:
                    print(f"Error loading matched recipe: {e}")

        if matched_recipe_doc:
            # prepend to docs and filter out any existing docs with the same name (case-insensitive)
            merged_docs = [matched_recipe_doc]
            matched_name_lower = matched_recipe_doc.metadata.get("name", "").lower()
            for doc in docs:
                if doc.metadata.get("name", "").lower() != matched_name_lower:
                    merged_docs.append(doc)
            docs = merged_docs[:self.retrieval_k]

        model_name = self.llm_model
        if not model_name or "llama" in model_name.lower() or "gemini" in model_name.lower():
            model_name = "gemini-2.5-flash"
        
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or "cvit-edu-1"
        llm = ChatVertexAI(
            model_name=model_name,
            temperature=0,
            project=project_id,
            location="us-central1"
        )

        context_blocks = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "N/A")
            name = doc.metadata.get("name", "Unknown")
            context_blocks.append(
                f"[{i}] Name: {name}\nURL: {source}\n{doc.page_content}"
            )

        context = "\n\n".join(context_blocks)

        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        system_instructions = f"""You are an expert recipe and nutrition assistant.

Goal:
- Answer food and cooking questions using the recipe context provided below.
- If the requested recipe is present in the context, you MUST use that recipe's information.
- If the requested recipe is NOT present in the context, you should generate a high-quality recipe for it using your general knowledge, ensuring it is realistic, accurate, and structured.

Response requirements:
- Be concise and practical.
- Do not include any conversational preamble or introductory text (like "Here is the recipe..." or "I could not find it in the database but here is..."). Start directly with the recipe title header `## [Recipe Name]`.
- Use the following clean Markdown structure when providing a recipe:
  
  ## [Recipe Name]
  
  ### Quick Details
  - Prep Time: [Prep Time]
  - Servings: [Serves]
  - Calories: [Calories]

  ### Ingredients
  - [Ingredient 1]
  - [Ingredient 2]

  ### Preparation Steps
  1. [Step 1]
  2. [Step 2]
  
  ### Sources
  - [Source Name]: [URL]

- Do not include recipe numbers like "Recipe 1".
- If you generated the recipe using general knowledge and no source URL is available, use "General Cooking Knowledge" for Source Name and "https://cooking.example.com" or a similar placeholder for URL.
- If the user asks general or comparison questions, keep the formatting structured with clear headers and bullet points.

Context:
{context}"""

        messages_list = [SystemMessage(content=system_instructions)]

        if history:
            for msg in history:
                if isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "user":
                        messages_list.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages_list.append(AIMessage(content=content))

        if image_b64:
            messages_list.append(HumanMessage(
                content=[
                    {"type": "text", "text": f"Question: {question}\n\nUse the context and the image to answer."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                ]
            ))
        else:
            messages_list.append(HumanMessage(content=f"Question: {question}"))

        response = llm.invoke(messages_list)

        return {"result": response.content}


def build_cli_parser():
    parser = argparse.ArgumentParser(description="Ask recipe questions using local RAG + Ollama.")
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--json-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "swiggy_recipe_json"),
        help="Directory with recipe JSON files.",
    )
    parser.add_argument(
        "--db-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "recipe_db_local"),
        help="Directory for local Chroma vector DB.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model name (e.g., llama3.2, mistral).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of retrieved recipes per query.",
    )
    return parser


def run_interactive(rag: RecipeRAG):
    print("Recipe RAG CLI")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        response = rag.query(question)
        print("\nANSWER:\n")
        print(response["result"])
        print()

if __name__ == "__main__":
    parser = build_cli_parser()
    args = parser.parse_args()

    rag = RecipeRAG(
        json_dir=args.json_dir,
        db_dir=args.db_dir,
        llm_model=args.model,
        retrieval_k=max(1, args.k),
    )

    if args.question:
        question = " ".join(args.question).strip()
        response = rag.query(question)
        print("\nANSWER:\n")
        print(response["result"])
    else:
        run_interactive(rag)