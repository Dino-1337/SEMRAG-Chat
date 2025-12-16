# SEMRAG — Semantic + Knowledge Graph RAG System

SEMRAG is a **research-grade Retrieval-Augmented Generation (RAG) system** built following the **SEMRAG research paper** architecture.  
It is designed to answer questions **strictly grounded** in a provided text corpus, with strong safeguards against hallucination.

The system processes a primary text corpus (PDF), builds a **semantic index and knowledge graph**, and answers queries using **local + global retrieval** with evidence-based synthesis.

---

## 🔍 Key Features

- 📄 **PDF-based corpus ingestion**
- 🧠 **Semantic chunking** with contextual continuity
- 🕸️ **Canonicalized knowledge graph** (entities + relationships)
- 🧩 **Community detection & summarization**
- 🔎 **Local RAG** (chunk-level semantic + graph-aware retrieval)
- 🌍 **Global RAG** (community-level semantic retrieval)
- ⚖️ **Weighted result ranking** (local + global fusion)
- 🛡️ **Hallucination-resistant answering**
- 📚 **Citation-backed answers**

---

## 🧱 Project Structure

```
src/
├── chunking/
│   ├── semantic_chunker.py
│   └── buffer_merger.py
│
├── graph/
│   ├── entity_extractor.py
│   ├── relationship_extractor.py
│   ├── graph_builder.py
│   ├── community_detector.py
│   └── summarizer.py
│
├── retrieval/
│   ├── local_search.py
│   ├── global_search.py
│   └── ranker.py
│
├── llm/
│   ├── llm_client.py
│   ├── prompt_templates.py
│   └── answer_generator.py
│
├── utils/
│   ├── data_loader.py
│   └── query_expander.py
│
└── pipeline/
    ├── index_builder.py
    └── ambedkargpt.py
```

---

## ⚙️ Requirements

### System
- Python **3.10+**
- **Ollama** (running locally)
- RAM: **8 GB minimum** (16 GB recommended)

### LLM
- Tested with: **Mistral 7B**

```bash
ollama pull mistral:7b
```

---

## 📦 Installation

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
python -m spacy download en_core_web_sm
```

---

## 📄 Preparing the Corpus

Place your primary PDF inside the project:

```
data/
└── corpus.pdf
```

⚠️ **Important**: This system is designed to be corpus-bounded. All answers are derived **only** from the provided PDF.

---

## 🏗️ Building the Index (Pipeline 1)

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

Artifacts are stored in:
```
data/processed/
├── chunks.json
├── chunk_embeddings.npy
├── entities.json
├── knowledge_graph.pkl
├── communities.json
├── community_summaries.json
└── metadata.json
```

---

## 💬 Running the QA System (Pipeline 2)

Start the interactive app:

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

## 📌 Answer Format

Each response includes:
- **Synthesized answer**
- **Top citations** (chunks)
- **Search metadata**:
  - local vs global matches
  - entities involved
  - communities used

This ensures **transparency and traceability**.

---

## 🛡️ Hallucination Control

The system is designed to:
- ✅ Never use external knowledge
- ✅ Clearly state when the corpus is insufficient
- ✅ Distinguish between:
  - Author's arguments
  - Theories the author explicitly rejects

---

## 🔬 Intended Use

- Academic research
- Digital humanities
- Political philosophy analysis
- Explainable AI demonstrations
- RAG system experimentation

---

## 🚧 Limitations

- Answers are limited to the provided corpus
- Not intended for general-purpose QA
- PDF quality affects extraction accuracy

---

## 📜 License

This project is intended for educational and research purposes.

---

## ✨ Acknowledgements

- **SEMRAG Research Paper** — architecture and methodology
- **SentenceTransformers**
- **spaCy**
- **NetworkX**
- **Ollama**
- **Mistral AI**

---

## 📬 Contact

For questions or collaboration, open an issue or reach out via GitHub.
