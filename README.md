# SEMRAG — Semantic + Knowledge Graph RAG System

SEMRAG is a research-grade Retrieval-Augmented Generation (RAG) system built following the SEMRAG research paper architecture. It is designed to answer questions strictly grounded in a provided text corpus, with strong safeguards against hallucination.

The system processes a primary text corpus (PDF), builds a semantic index and knowledge graph, and answers queries using local + global retrieval with evidence-based synthesis.

---

## Key Features

- PDF-based corpus ingestion
- Semantic chunking with contextual continuity
- Canonicalized knowledge graph (entities + relationships)
- Community detection and summarization
- Local RAG (chunk-level semantic + graph-aware retrieval)
- Global RAG (community-level semantic retrieval)
- Weighted result ranking (local + global fusion)
- Hallucination-resistant answering
- Citation-backed answers

---

## Project Structure

```
SEMRAG/
├── data/
│   ├── Ambedkar_works.pdf          # Input corpus
│   └── processed/                   # Generated artifacts
│       # ChromaDB vector store
│
├── src/
│   ├── chunking/
│   │   ├── semantic_chunker.py     # Semantic text chunking
│   │   └── buffer_merger.py        # Chunk merging logic
│   │
│   ├── graph/
│   │   ├── entity_extractor.py     # NER + concept extraction
│   │   ├── relationship_extractor.py
│   │   ├── graph_builder.py        # NetworkX graph construction
│   │   ├── community_detector.py   # Leiden/Louvain clustering
│   │   └── summarizer.py           # LLM-based summaries
│   │
│   ├── retrieval/
│   │   ├── local_search.py         # Chunk-level retrieval
│   │   ├── global_search.py        # Community-level retrieval
│   │   └── ranker.py               # Result fusion
│   │
│   ├── llm/
│   │   ├── llm_client.py           # Ollama integration
│   │   ├── prompt_templates.py     # Prompt engineering
│   │   └── answer_generator.py     # Answer synthesis
│   │
│   ├── utils/
│   │   ├── data_loader.py          # PDF text extraction
│   │   ├── query_expander.py       # Query enhancement
│   │   └── vector_store.py         # ChromaDB wrapper
│   │
│   └── pipeline/
│       ├── index_builder.py        # Index construction pipeline
│       └── ambedkargpt.py          # Main orchestrator
│
├── app.py                          # Interactive QA interface
├── build_index.py                  # Index building script
├── config.yaml                     # System configuration
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Requirements

### System
- Python 3.10+
- Ollama (running locally)

### LLM
- Tested with: Mistral 7B and llama3.2

```bash
ollama pull mistral:7b
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Dino-1337/SEMRAG.git
cd SEMRAG
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy model
```bash
python -m spacy download en_core_web_lg
```

---

## Preparing the Corpus

Place your primary PDF inside the project:

```
data/
└── Ambedkar_works.pdf
```

## Building the Index (Pipeline 1)

This step:
- Loads the PDF
- Performs semantic chunking
- Extracts canonical entities
- Builds the knowledge graph
- Detects communities
- Generates summaries
- Saves all artifacts to disk

```bash
python build_index.py
```

<img width="1824" height="901" alt="image" src="https://github.com/user-attachments/assets/75a64fe8-1912-4b68-ae24-4007109029c4" />

Artifacts are stored in `data/processed/`

---

## Running the QA System (Pipeline 2)

Start the chat app:

```bash
python app.py
```

You can now ask questions in the terminal.

### Example Questions
```
What are the main concepts discussed in the document?
How does the author explain [specific concept]?
What arguments are presented regarding [topic]?
```

Type `/exit` to quit.

---

## Answer Format

Each response includes:
- Synthesized answer
- Top citations (chunks with similarity scores)
- Search metadata:
  - Local vs global matches
  - Entities involved
  - Communities used

This ensures transparency and traceability.

---

## Hallucination Control

The system is designed to:
- Never use external knowledge
- Clearly state when the corpus is insufficient
- Distinguish between:
  - Author's arguments
  - Theories the author explicitly rejects
