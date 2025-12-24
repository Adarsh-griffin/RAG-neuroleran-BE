# NeuroLearn: Multi-Modal RAG Tutoring Platform

NeuroLearn is an end-to-end AI platform that transforms static educational materials (PDFs/Textbooks) into an interactive, tutor-led learning experience. By leveraging Retrieval-Augmented Generation (RAG) and Generative Video, the system moves beyond simple summarization to provide active teaching and automated evaluation.

## System Architecture

The platform is built on a decoupled architecture designed for high throughput and low-latency user interactions.

### 1. Ingestion & Retrieval Pipeline
- **Recursive Character Splitting**: Documents are chunked to maintain semantic context.
- **Vector Embeddings**: Utilizes sentence-transformers for local embedding generation.
- **Contextual Injection**: A sliding-window RAG approach ensures that the LLM has high-fidelity access to source material without losing narrative flow.

### 2. Multi-Modal Generation
- **Inference**: Powered by Groq (LLaMA 3 / Mixtral) for sub-second response times.
- **Visual Synthesis**: Explanations are passed through a TTS/Video pipeline using Tavus API and Wav2Lip to generate human-like avatar lectures.
- **Grounding**: Uses Serpher AI to enrich AI responses with real-world images and web-verified references.

### 3. Evaluation Loop
- **Automated Assessment**: Generates MCQs and open-ended questions based on the specific document context.
- **Source-Grounded Grading**: The AI evaluates student answers by comparing them directly against the vector store, providing "improvement pointers" rather than just a score.

##  Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | React, TypeScript, TailwindCSS, TanStack Query, Framer Motion |
| **Backend** | Python (Flask), MongoDB, GridFS |
| **AI / Orchestration** | LangChain, Groq, Sentence-Transformers |
| **Audio/Video** | Whisper, PlayAI, Tavus, FFmpeg |
| **Data** | Serpher AI (Web Scraping/Semantic Search) |

##  Key Engineering Features

- **Asynchronous Processing**: Long-running tasks like video generation and PDF ingestion are handled via async pipelines to prevent UI blocking.
- **Hallucination Mitigation**: Custom prompt engineering and RAG grounding ensure that the AI "tutor" never invents facts outside the provided textbook.
- **Type-Safe UI**: Full TypeScript implementation on the frontend for scalable, maintainable component architecture.

##  System Flow

![System Flow Diagram](https://drive.google.com/file/d/19gPm8omYBqaXKIodcwkD_JivY9OToUEd/view?usp=sharing )

---

#  Startup Guide and Setup

Follow these steps to set up and run the NeuroLearn platform locally.

## Prerequisites

Ensure you have the following installed on your machine:
- **Node.js**: (v18+ recommended)
- **Python**: (v3.10+ recommended)
- **MongoDB**: (Running locally on port 27017)
- **Docker**: (Required for Qdrant vector database)

## 1. Environment Configuration

### Backend Keys
Ensure you have a `.env` file in the `NeuroBack` directory (or root) with the necessary API keys:
```env
GROQ_API_KEY=your_groq_api_key
FAL_API_KEY=your_fal_ai_key
# Add other keys as required (TAVUS_API_KEY, SERPHER_API_KEY, etc.)
```

## 2. Backend Setup (`NeuroBack`)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Adarsh-griffin/RAG-neuroleran-BE.git
    cd RAG-neuroleran-BE
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\Activate.ps1
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirement.txt
    ```

4.  **Start Qdrant Vector Database (Docker):**
    Open a terminal and run:
    ```bash
    docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant
    ```

5.  **Start the Flask Server:**
    ```bash
    python test_groq.py
    ```
    The backend server will start on `http://127.0.0.1:5000`.

## 3. Frontend Setup (`NeuroFront`)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Adarsh-griffin/RAG-neuroleran-FE.git
    cd RAG-neuroleran-FE
    ```

2.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```

3.  **Start the Development Server:**
    ```bash
    npm run dev
    ```
    The frontend will launch at `http://localhost:5173` (or similar).

## Usage

1.  Open your browser and navigate to the frontend URL (e.g., `http://localhost:5173`).
2.  Upload a PDF textbook/document.
3.  The system will ingest the document, generate embeddings, and prepare the interactive tutor.
4.  Engage with the generic AI tutor, or ask specific questions based on the uploaded content.
