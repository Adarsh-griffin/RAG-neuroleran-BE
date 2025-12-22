# Importing Libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os
import PyPDF2
import uuid
from pymongo import MongoClient
import torch
from groq import Groq   # ✅ Groq client

# =====================================================================================================
# MongoDB Setup
# =====================================================================================================

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # update with your connection string
db = client["neurolearn"]
collection = db["files"]

def get_latest_pdf_doc():
    """Fetch the latest uploaded PDF document from MongoDB."""
    file_doc = collection.find_one({}, sort=[("uploadDate", -1)])  # latest file
    if file_doc:
        return file_doc
    else:
        print("⚠️ No PDF found in MongoDB")
        return None
    

def save_explanation_to_mongo(file_path, explanation):
    """Update MongoDB document with generated explanation using filePath."""
    result = collection.update_one(
        {"filePath": file_path},
        {"$set": {"explanation": explanation}}
    )
    if result.modified_count > 0:
        print(f"✅ Explanation saved successfully for {file_path}")
    else:
        print(f"⚠️ Failed to update document for {file_path}")


# =====================================================================================================
# Groq Setup
# =====================================================================================================

# Groq Setup with Rotation
GROQ_API_KEYS_STR = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS = [k.strip() for k in GROQ_API_KEYS_STR.split(',') if k.strip()]

if not GROQ_API_KEYS:
    # Fallback to single key if comma split fails or not present (backward compat)
    single_key = os.getenv("GROQ_API_KEY")
    if single_key:
        GROQ_API_KEYS = [single_key]
    else:
        raise ValueError("GROQ_API_KEY environment variable is required.")

_current_key_index = 0

def get_current_groq_client():
    global _current_key_index
    return Groq(api_key=GROQ_API_KEYS[_current_key_index])

def rotate_key():
    global _current_key_index
    _current_key_index = (_current_key_index + 1) % len(GROQ_API_KEYS)
    print(f"Rotating to Groq API key index: {_current_key_index}")

def execute_with_retry(func, *args, **kwargs):
    max_retries = len(GROQ_API_KEYS)
    last_exception = None
    for attempt in range(max_retries):
        try:
            client = get_current_groq_client()
            return func(client, *args, **kwargs)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with key index {_current_key_index}: {e}")
            last_exception = e
            rotate_key()
    raise last_exception

def groq_generate(prompt, max_tokens=700, temperature=0.7):
    """Send prompt to Groq API and return response text."""
    
    def _do_generate(client, p, mt, temp):
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": p}],
            temperature=temp,
            max_completion_tokens=mt,
            top_p=1,
            reasoning_effort="low",
            stream=False
        )
        return completion.choices[0].message.content.strip()

    try:
        return execute_with_retry(_do_generate, prompt, max_tokens, temperature)
    except Exception as e:
        print(f"Error generating response after retries: {e}")
        return None


# =====================================================================================================
# PDF Extraction
# =====================================================================================================
def extract_text_from_pdf(pdf_path):
    """Extracts all text from a given PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except FileNotFoundError:
        print(f"Error: The file at {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return None
    return text


# ------------------------
# Token-aware chunk splitter (sliding window)
# ------------------------
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into overlapping chunks for sliding window processing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


# =====================================================================================================
# LangChain Output Parser Setup
# =====================================================================================================
response_schemas = [
    ResponseSchema(
        name="explanation",
        description="A concise, readable summary of the provided chunks."
    ),
    ResponseSchema(
        name="links",
        description="A list of 3 reliable reference links to resources about the main concepts (Wikipedia, official docs, educational articles)."
    )
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


# ------------------------
# Prompt with CoT + Parser
# ------------------------
def prompt_summary_and_links(chunk, prev_summary=""):
    return f"""
You are an AI tutor.

Your task:
1. Read the following chunk of text.
2. Create a concise summary of the chunk.
3. Generate exactly 3 reference links (reliable sources like Wikipedia, official docs, or research articles) related to the main concepts.

Previous Summary: {prev_summary}

[Current Text Chunk]:
{chunk}

{format_instructions}

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""



# ------------------------
# Sliding context processor
# ------------------------
def process_with_sliding_window(pdf_path, chunk_size=3000, chunk_overlap=200):
    print(f"Step 1: Extracting text from {pdf_path}...")
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        return None

    print("Step 2: Splitting text into chunks (sliding window)...")
    chunks = split_text_into_chunks(pdf_text, chunk_size, chunk_overlap)
    print(f"✅ Document split into {len(chunks)} overlapping chunks.")

    results = []
    prev_summary = ""

    for i, chunk in enumerate(chunks[:2]):  # demo: limit to 2 chunks
        print(f"\n🔹 Processing Chunk {i+1}/{len(chunks)}...")
        final_prompt = prompt_summary_and_links(chunk, prev_summary)
        response = groq_generate(final_prompt)

        try:
            parsed_output = output_parser.parse(response)
        except Exception as e:
            print("⚠️ Parsing failed, raw response:", response)
            parsed_output = {"explanation": response, "links": []}

        results.append(parsed_output)

        # Update sliding context with summary only
        prev_summary = parsed_output.get("explanation", "")[-800:]

    print("\n✅ All chunks processed with sliding context window.")
    return results



# =====================================================================================================
# Run
# =====================================================================================================
print("===========🚀 Starting processing...===========")

file_doc = get_latest_pdf_doc()

if file_doc:
    print("file_doc:", file_doc, type(file_doc))
    pdf_path = os.path.normpath(file_doc["filePath"])  # normalize path
    print("Successfully retrieved file path:")

    # Run processing
    output = process_with_sliding_window(pdf_path)

    # Save explanation back using filePath (now as structured JSON)
    save_explanation_to_mongo(file_doc["filePath"], output)

    print("\n📦 Final Structured Output:\n", output)

    # --- IMAGE FINDER INTEGRATION ---
    from image_finder import process_images_for_file
    
    # Extract text from the output for keyword generation
    # Output is a list of dicts, we combine explanations
    combined_summary = " ".join([item.get("explanation", "") for item in output])
    
    if combined_summary:
        print(f"\n🚀 Triggering Image Finder for {pdf_path}...")
        process_images_for_file(file_doc["filePath"], combined_summary)
    else:
        print("⚠️ No summary text available for image finding.")

else:
    print("⚠️ No file found in MongoDB")
