import os
import textwrap
import re
import sys
import subprocess
from datetime import datetime
import json
import requests
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from retrieval import get_relevant_docs
# from faster_whisper import WhisperModel  # Local STT disabled; using Groq hosted Whisper

# ---------------- Groq Setup ----------------
from groq import Groq

# Initialize Groq clients (supports multiple API keys for rotation)
# Load API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

groq_client = Groq(api_key=GROQ_API_KEY)

def groq_generate(prompt, max_tokens=512, temperature=0.7):
    """Send a prompt to Groq and return the response text."""
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",  # You can change to other Groq models if needed
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            reasoning_effort="low",
            stream=False  # non-streaming for API responses
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

llm = groq_generate

from langchain.llms.base import LLM
from typing import Any, List, Optional

class GroqLLM(LLM):
    """LangChain wrapper for Groq's groq_generate function."""

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return groq_generate(prompt)

    @property
    def _identifying_params(self) -> dict:
        return {"name": "GroqLLM"}

    @property
    def _llm_type(self) -> str:
        return "groq"

# ---------------- Flask Setup ----------------
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ---------------- Configuration for Uploads and MongoDB ----------------
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
TTS_OUTPUT_FOLDER = 'collections'  # reuse existing public folder
if not os.path.exists(TTS_OUTPUT_FOLDER):
    os.makedirs(TTS_OUTPUT_FOLDER)

# ---------------- fal.ai Kokoro TTS Config (manual overrides) ----------------
# You can hardcode your fal.ai API key and endpoint here. If left blank, the app
# will fall back to environment variables FAL_API_KEY/FAL_KEY and FAL_KOKORO_URL.
FAL_API_KEY = "11984d35-b73b-4797-9697-d4e91775aff7:e705f60c28ccf229d7235ff0b6f860c0"  # Format: <id>:<secret> used with 'Key ' header on fal.run
FAL_KOKORO_URL = "https://api.fal.ai/kokoro/tts"  # Change if your endpoint differs

try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['neurolearn']
    files_collection = db['files']
    client.server_info()
    print("✅ MongoDB connection successful.")
except Exception as e:
    print(f"❌ Could not connect to MongoDB: {e}")
    client = None

# ---------------- Helper ----------------
def extract_helpful_answer(text):
    match = re.search(r"Helpful Answer:\s*(.*?)(?:\n\s*Question:|\Z)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# ===================================================================================================================================
# ---------------- Upload ----------------

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if client:
            if files_collection.find_one({"originalName": file.filename}):
                print(f"File '{file.filename}' already exists in DB. Skipping metadata insert.")
            else:
                metadata = {
                    "originalName": file.filename,
                    "filePath": filepath,
                    "fileType": file.content_type,
                    "fileSize": os.path.getsize(filepath),
                    "uploadDate": datetime.utcnow()
                }
                files_collection.insert_one(metadata)
        else:
            return jsonify({"error": "Database connection is not available."}), 500

        try:
            # Run ingest.py
            print(f"Starting subprocess for ingest: {filepath}")
            subprocess.Popen([sys.executable, 'ingest.py', filepath])

            # Run pass2main.py
            print(f"Starting subprocess for passmain_groq")
            subprocess.Popen([sys.executable, 'passmain_groq.py'])

        except Exception as e:
            print(f"Error starting subprocess: {e}")
            return jsonify({"error": "Failed to start background processes"}), 500

        return jsonify({
            "message": f"File '{filename}' uploaded. Processing started (analysis + explanation).",
            "filename": filename,
            "status": "processing"
        }), 202

# ===================================================================================================================================
# --- Processing Status ---

@app.route('/api/processing-status/<filename>', methods=['GET'])
def check_processing_status(filename):
    """Check if file processing is complete by looking for explanation in MongoDB."""
    try:
        if not client:
            return jsonify({"error": "Database connection is not available."}), 500
        
        # Find the file document
        file_doc = files_collection.find_one({"originalName": filename})
        
        if not file_doc:
            return jsonify({"status": "not_found", "message": "File not found"}), 404
        
        # Check if explanation exists (indicates processing is complete)
        explanation = file_doc.get("explanation")
        
        if explanation:
            return jsonify({
                "status": "completed",
                "message": "Processing completed successfully",
                "hasExplanation": True
            })
        else:
            return jsonify({
                "status": "processing",
                "message": "File is still being processed",
                "hasExplanation": False
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===================================================================================================================================
# --- Q&A ---

@app.route('/api/files', methods=['GET'])
def get_files():
    if not client:
        return jsonify({"error": "Database connection is not available."}), 500
    try:
        # Find all documents and only return the originalName field, sorted by date
        files = list(files_collection.find({}, {"_id": 0, "originalName": 1}).sort("uploadDate", -1))
        file_names = [f['originalName'] for f in files]
        return jsonify(file_names)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/qa', methods=['POST'])
def ask_question():
    """
    This endpoint handles AI queries for both qa.html and learning.html
    """
    data = request.json
    query = data.get('question', '')

    # --- START OF ADDED CODE ---
    file_name = data.get('fileName', '')

    if not query:
        return jsonify({'error': 'No question provided'}), 400
    if not file_name:
        return jsonify({'error': 'No file name provided for context'}), 400
        
    collection_name = os.path.splitext(os.path.basename(file_name))[0].lower().replace(" ", "_")
    # --- END OF ADDED CODE ---

    try:
        retriever = get_relevant_docs(query, collection_name)

        qa = RetrievalQA.from_chain_type(
            llm=GroqLLM(),   # ✅ fixed: pass wrapped Groq LLM
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )

        template = '''You are an AI tutor designed to assist students by providing
        clear, concise, and easy-to-understand answers to their queries.

        Your task:
        1. Retrieve relevant information from the knowledge base.
        2. Provide an informative yet simple response.
        3. Use examples and analogies when necessary.
        4. Avoid unnecessary details and technical jargon.

        Now answer the following question:
        {query}'''

        prompt = PromptTemplate(template=template, input_variables=["query"])
        formatted_prompt = prompt.format(query=query)

        response = qa.invoke({"query": formatted_prompt})  # ✅ pass dict with query
        result_text = response['result'].strip()
        helpful_answer = extract_helpful_answer(result_text)
        wrapped_response = textwrap.fill(helpful_answer, width=80)

        return jsonify({'response': wrapped_response})
        

    except Exception as e:
        import traceback
        traceback.print_exc()  # ✅ log full error in Flask console
        return jsonify({'error': str(e)}), 500
    

#=================================================================================================================================
# ---------------- Learning ----------------

def get_latest_pdf_doc():
    return files_collection.find_one({}, sort=[("uploadDate", -1)])

@app.route("/api/get_text")
def get_text():
    """Return the latest uploaded text from MongoDB."""
    file_doc = files_collection.find_one({}, sort=[("uploadDate", -1)])  # latest document

    if not file_doc:
        return jsonify({"script_text": "No content available."})

    exp = file_doc.get("explanation")
    if not exp:
        return jsonify({"script_text": "No content available."})

    texts = []
    if isinstance(exp, list):
        for item in exp:
            if isinstance(item, dict):
                texts.append(str(item.get("explanation", "")).strip())
            else:
                texts.append(str(item).strip())
    elif isinstance(exp, dict):
        texts.append(str(exp.get("explanation", "")).strip())
    else:
        texts.append(str(exp).strip())

    combined_text = " ".join([t for t in texts if t])
    return jsonify({"script_text": combined_text or "No content available."})
   

@app.route("/api/get_links")
def get_links():
    """Return the latest uploaded links from MongoDB."""
    file_doc = files_collection.find_one({}, sort=[("uploadDate", -1)])  # latest document

    if not file_doc:
        return jsonify({"links": []})

    exp = file_doc.get("explanation")
    if not exp:
        return jsonify({"links": []})

    all_links = []
    def push_links(val):
        if not val:
            return
        if isinstance(val, list):
            for x in val:
                if isinstance(x, str):
                    all_links.append(x)
        elif isinstance(val, str):
            all_links.append(val)

    if isinstance(exp, list):
        for item in exp:
            if isinstance(item, dict):
                push_links(item.get("links"))
    elif isinstance(exp, dict):
        push_links(exp.get("links"))

    return jsonify({"links": all_links})





# ==================================================================================================================================
# --- Learning Page: Kokoro TTS via fal.ai ---

@app.route('/api/learning-tts', methods=['POST'])
def learning_tts():
    """Synthesize speech for given text using fal.ai Kokoro and return a public audio URL."""
    try:
        data = request.get_json(force=True)
        text = (data or {}).get('text', '').strip()
        voice = (data or {}).get('voice', 'af_heart')  # example voice
        speed = float((data or {}).get('speed', 1.0))

        if not text:
            # Fallback to latest script text if none provided
            file_doc = files_collection.find_one({}, sort=[("uploadDate", -1)])
            if not file_doc or "explanation" not in file_doc:
                return jsonify({"error": "No text provided and no content available."}), 400
            text = " ".join(item.get("explanation", "") for item in file_doc["explanation"]).strip()

        fal_key = (FAL_API_KEY or os.getenv('FAL_API_KEY') or os.getenv('FAL_KEY'))
        if not fal_key:
            return jsonify({"error": "FAL_API_KEY not set in environment."}), 500

        # NOTE: Endpoint may vary; try multiple endpoints/headers for resilience
        candidates = [
            # Preferred working format (matches your Postman screenshot):
            #   URL: https://fal.run/fal-ai/kokoro/american-english
            #   Header: Authorization: Key <fal_id:secret>
            ('https://fal.run/fal-ai/kokoro/american-english', 'Key'),
            # Fallbacks
            (FAL_KOKORO_URL or os.getenv('FAL_KOKORO_URL', 'https://api.fal.ai/kokoro/tts'), 'Key'),
            ('https://fal.run/fal-ai/kokoro/tts', 'Key'),
            ('https://api.fal.ai/v1/kokoro/tts', 'Key'),
            ('https://api.fal.ai/kokoro/tts', 'Bearer'),
            ('https://fal.run/fal-ai/kokoro/tts', 'Bearer'),
        ]

        payload = {
            "input": {
                "text": text,
                "voice": voice,
                "speed": speed,
                # add other kokoro params as needed
            }
        }
        last_error = None
        resp = None
        for fal_url, auth_mode in candidates:
            try:
                if not fal_url:
                    continue
                headers = {
                    "Authorization": (f"Key {fal_key}" if auth_mode == 'Key' else f"Bearer {fal_key}"),
                    "Content-Type": "application/json"
                }
                print(f"[KOKORO-TTS] Trying {fal_url} with {auth_mode} auth…")
                resp = requests.post(fal_url, headers=headers, data=json.dumps(payload), timeout=60)
                if resp.status_code < 300:
                    break
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                continue

        if not resp or resp.status_code >= 300:
            return jsonify({"error": "fal.ai TTS request failed", "details": last_error}), 502

        print(f"[KOKORO-TTS] fal.ai response status: {resp.status_code}")
        print(f"[KOKORO-TTS] Response headers: {dict(resp.headers)}")

        # Expecting binary audio or a URL. Handle both cases.
        audio_bytes = None
        audio_url_from_fal = None
        try:
            # If fal returns JSON with a URL
            j = resp.json()
            print(f"[KOKORO-TTS] fal.ai JSON response: {j}")
            audio_url_from_fal = j.get('audio_url') or j.get('output', {}).get('audio_url')
        except Exception:
            print(f"[KOKORO-TTS] No JSON response, treating as binary audio (size: {len(resp.content)} bytes)")
            pass

        if not audio_url_from_fal:
            # Assume binary audio stream
            audio_bytes = resp.content

        # Save locally if we have bytes; otherwise proxy the remote URL
        public_url = None
        if audio_bytes:
            out_name = f"learning-tts-{int(datetime.utcnow().timestamp())}.mp3"
            out_path = os.path.join(TTS_OUTPUT_FOLDER, out_name)
            with open(out_path, 'wb') as f:
                f.write(audio_bytes)
            public_url = f"/static/collections/{out_name}"
            print(f"[KOKORO-TTS] Saved audio locally: {out_path}")
            print(f"[KOKORO-TTS] Public URL: {public_url}")
        else:
            public_url = audio_url_from_fal
            print(f"[KOKORO-TTS] Using remote URL from fal.ai: {public_url}")

        if not public_url:
            return jsonify({"error": "No audio produced by fal.ai"}), 502

        print(f"[KOKORO-TTS] Final audio URL returned: {public_url}")
        return jsonify({"audioUrl": public_url})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ==================================================================================================================================
# --- Q&A TTS via Groq Play.ai ---

@app.route('/api/qa-tts', methods=['POST'])
def qa_tts():
    try:
        data = request.get_json(force=True)
        text = (data or {}).get('text', '').strip()
        voice = (data or {}).get('voice', 'Aaliyah-PlayAI')
        fmt = (data or {}).get('format', 'wav')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Synthesize using Groq Play.ai model
        target_name = f"qa-tts-{int(datetime.utcnow().timestamp())}.{fmt}"
        out_path = os.path.join(TTS_OUTPUT_FOLDER, target_name)

        resp = groq_client.audio.speech.create(
            model="playai-tts",
            voice=voice,
            response_format=fmt,
            input=text,
        )
        # Stream to file
        with open(out_path, 'wb') as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)

        public_url = f"/static/collections/{target_name}"
        print(f"[QA-TTS] Generated TTS via Play.ai -> {public_url}")
        return jsonify({"audioUrl": public_url})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ==================================================================================================================================
# --- Voice Q&A (STT -> QA -> optional TTS) ---

@app.route('/api/qa-voice', methods=['POST'])
def qa_voice():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio provided'}), 400

        audio_file = request.files['audio']
        file_name_ctx = request.form.get('fileName', '')
        if not file_name_ctx:
            return jsonify({'error': 'No file name provided for context'}), 400

        # Validate size (avoid extremely large uploads)
        audio_file.seek(0, os.SEEK_END)
        size_bytes = audio_file.tell()
        audio_file.seek(0)
        if size_bytes > 25 * 1024 * 1024:
            return jsonify({'error': 'Audio file too large (max 25MB)'}), 400

        # Save temp audio (webm/opus from browser)
        temp_dir = os.path.join('static', 'uploads')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_path = os.path.join(temp_dir, secure_filename('qa-voice.webm'))
        audio_file.save(temp_path)

        # Transcode to 16kHz mono WAV using ffmpeg to avoid decoder issues
        wav_path = os.path.join(temp_dir, secure_filename('qa-voice-16k.wav'))
        try:
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', temp_path, '-ac', '1', '-ar', '16000', wav_path
            ]
            proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0 or (not os.path.exists(wav_path)):
                return jsonify({'error': f'Audio transcoding failed: {proc.stderr.decode("utf-8", errors="ignore")[:300]}'}), 400
        except FileNotFoundError:
            return jsonify({'error': 'ffmpeg not found. Please install FFmpeg and add it to PATH.'}), 500

        # STT with Groq hosted Whisper (whisper-large-v3-turbo)
        try:
            with open(wav_path, "rb") as f:
                groq_resp = groq_client.audio.transcriptions.create(
                    file=(os.path.basename(wav_path), f.read()),
                    model="whisper-large-v3-turbo",
                    temperature=0,
                    response_format="verbose_json",
                )
            transcript = getattr(groq_resp, 'text', '') or (groq_resp.get('text') if isinstance(groq_resp, dict) else '')
            detected_lang = getattr(groq_resp, 'language', None) or (groq_resp.get('language') if isinstance(groq_resp, dict) else None)
        except Exception as e:
            print(f"[QA-VOICE] Groq STT failed: {e}")
            transcript = ''
            detected_lang = None
        print(f"[QA-VOICE] Detected language: {detected_lang}")
        print(f"[QA-VOICE] Transcript: {transcript}")
        if not transcript:
            return jsonify({'error': 'Transcription failed'}), 500

        # Optionally translate transcript to English for retrieval/LLM if source isn't English
        translated_text = None
        if detected_lang and str(detected_lang).lower() != 'en':
            try:
                # Prefer local LibreTranslate if running
                libre_url = 'http://localhost:5000/translate'
                data = {"q": transcript, "source": detected_lang, "target": "en", "format": "text"}
                r = requests.post(libre_url, headers={"Content-Type": "application/json"}, data=json.dumps(data), timeout=10)
                r.raise_for_status()
                translated_text = r.json().get('translatedText')
            except Exception as _:
                # No local Whisper fallback (disabled)
                translated_text = None

        # Use translated text if available for the QA prompt
        effective_query = translated_text if translated_text else transcript
        if translated_text:
            print(f"[QA-VOICE] Transcript->EN: {translated_text}")

        # Route through existing QA pipeline with retrieval context
        collection_name = os.path.splitext(os.path.basename(file_name_ctx))[0].lower().replace(" ", "_")
        retriever = get_relevant_docs(effective_query, collection_name)
        qa = RetrievalQA.from_chain_type(
            llm=GroqLLM(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        template = '''You are an AI tutor designed to assist students by providing
        clear, concise, and easy-to-understand answers to their queries.

        Your task:
        1. Retrieve relevant information from the knowledge base.
        2. Provide an informative yet simple response.
        3. Use examples and analogies when necessary.
        4. Avoid unnecessary details and technical jargon.

        Now answer the following question:
        {query}'''
        prompt = PromptTemplate(template=template, input_variables=["query"]).format(query=effective_query)
        qa_resp = qa.invoke({"query": prompt})
        answer_text = textwrap.fill(extract_helpful_answer(qa_resp['result'].strip()), width=80)
        print(f"[QA-VOICE] Answer: {answer_text}")

        # Optional: TTS placeholder (Kokoro not wired yet)
        audio_url = None
        try:
            # Placeholder: create a small silent file to satisfy UI if needed
            # In real integration, replace with Kokoro TTS synthesis saving to TTS_OUTPUT_FOLDER
            audio_url = None
        except Exception:
            audio_url = None

        resp_payload = {
            'transcript': transcript,
            'response': answer_text,
            'audioUrl': audio_url,
            'translated': translated_text
        }

        # Best-effort cleanup of temp files
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

        return jsonify(resp_payload)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ===================================================================================================================================
# --- ASSESSMENT ---

@app.route('/api/assessment/generate', methods=['GET'])
def generate_question():
    try:
        # Get the latest document from MongoDB
        file_doc = files_collection.find_one({}, sort=[("uploadDate", -1)])
        if not file_doc or "explanation" not in file_doc:
            return jsonify({"error": "No explanation found in database"}), 404

        # Merge all explanation texts into one
        explanations = [item.get("explanation", "") for item in file_doc["explanation"]]
        combined_explanation = " ".join(explanations)

        template = f"""
        You are a tutor. Based on the following explanation, create ONE clear question
        that checks whether the student has understood the concept.

        Explanation: {combined_explanation}
        Question:
        """
        response = llm(template)

        start_index = response.find("Question:")
        if start_index != -1:
            final_response = response[start_index + len("Question:"):].strip()
        else:
            final_response = response.strip()

        return jsonify({"question": final_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/assessment/submit', methods=['POST'])
def submit_answer():
    try:
        # Get the latest document explanation again
        file_doc = files_collection.find_one({}, sort=[("uploadDate", -1)])
        if not file_doc or "explanation" not in file_doc:
            return jsonify({"error": "No explanation found in database"}), 404

        explanations = [item.get("explanation", "") for item in file_doc["explanation"]]
        combined_explanation = " ".join(explanations)

        # Read input from frontend
        data = request.json
        question = data.get("question", "")
        student_answer = data.get("answer", "")

        if not question or not student_answer:
            return jsonify({"error": "Both question and answer are required"}), 400

        prompt = f""" You are a tutor.

        Explanation: {combined_explanation}
        Question: {question}
        Student's Answer: {student_answer}

        Tasks:
        1. Check if the student’s answer is correct or not (be fair).
        2. Point out mistakes or missing parts if any.
        3. Give encouraging, constructive feedback.
        4. If wrong, provide the correct answer.

        Feedback:
        """

        evaluation = llm(prompt)

        start = evaluation.find("Feedback:")
        if start != -1:
            final_evaluation = evaluation[start + len("Feedback:"):].strip()
        else:
            final_evaluation = evaluation.strip()

        return jsonify({"feedback": final_evaluation})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================================================================================================================================


# ---------------- HTML Routing ----------------

# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve the React frontend for all routes"""
    # Path to the built React app
    react_build_path = os.path.join('..', 'NeuroFront', 'dist', 'spa')
    
    if path and os.path.exists(os.path.join(react_build_path, path)):
        # Serve static files (JS, CSS, images, etc.)
        return send_from_directory(react_build_path, path)
    else:
        # Serve index.html for all other routes (React Router will handle routing)
        return send_from_directory(react_build_path, 'index.html')

# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(debug=True)
