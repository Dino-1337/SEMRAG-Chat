# Step-by-Step Guide to Run SEMRAG RAG System

## Prerequisites Setup

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download spaCy English Model
```bash
python -m spacy download en_core_web_sm
```

### Step 3: Install and Setup Ollama
1. Download Ollama from https://ollama.ai
2. Install it on your system
3. Pull the Mistral model:
```bash
ollama pull mistral:7b
```
4. Verify Ollama is running:
```bash
ollama list
```
You should see `mistral:7b` in the list.

### Step 4: Place Your PDF
Place your `Ambedkar_works.pdf` file in the `data/` directory:
```
data/
  └── Ambedkar_works.pdf
```

---

## Running the System

### Option 1: Quick Check (Verify Graph Creation)

**Step 1:** Run the quick check script
```bash
python check_graph.py
```

This will:
- Load and process the PDF
- Show basic statistics
- Verify the knowledge graph was created

**Output:** You'll see:
- Number of chunks created
- Number of entities extracted
- Number of relationships
- Graph nodes and edges count
- Sample entities

---

### Option 2: Full Visualization

**Step 1:** Process PDF and create visualizations
```bash
python visualize_graph.py
```

This will:
- Load and process the PDF (if not already done)
- Create `knowledge_graph.png` (static visualization)
- Create `knowledge_graph.json` (graph data)
- Create `knowledge_graph.html` (interactive visualization, if plotly installed)

**Output Files:**
- `knowledge_graph.png` - Open with any image viewer
- `knowledge_graph.json` - Graph data in JSON format
- `knowledge_graph.html` - Open in web browser for interactive view

---

### Option 3: Use in Your Own Code

**Step 1:** Create a Python script (e.g., `my_script.py`)

```python
from src.pipeline.ambedkargpt import AmbedkarGPT

# Step 1: Initialize the system
print("Initializing AmbedkarGPT...")
gpt = AmbedkarGPT("config.yaml")

# Step 2: Load and process PDF
print("Processing PDF...")
gpt.load_and_process("data/Ambedkar_works.pdf")

# Step 3: Ask questions
print("Asking question...")
result = gpt.query("What were Dr. Ambedkar's views on social justice?")

# Step 4: Display results
print("\nAnswer:")
print(result['answer'])

print("\nCitations:")
for citation in result['citations']:
    print(f"  [{citation['rank']}] {citation['text'][:100]}...")
```

**Step 2:** Run your script
```bash
python my_script.py
```

---

## Step-by-Step Execution Flow

When you run the system, here's what happens internally:

### Phase 1: PDF Processing
1. **Load PDF** → Extracts text from `Ambedkar_works.pdf`
2. **Split Sentences** → Breaks text into individual sentences
3. **Semantic Chunking** → Groups sentences into semantic chunks using cosine similarity
4. **Token Limits** → Enforces chunk size limits and creates sub-chunks if needed

### Phase 2: Knowledge Graph Construction
5. **Extract Entities** → Uses spaCy NER to find entities (PERSON, ORG, GPE, etc.)
6. **Extract Relationships** → Uses dependency parsing to find entity relationships
7. **Build Graph** → Creates NetworkX graph with entities as nodes and relationships as edges
8. **Detect Communities** → Uses Leiden algorithm to group related entities

### Phase 3: Community Summarization
9. **Generate Summaries** → Uses LLM (Mistral) to summarize each community

### Phase 4: Query Processing (when you ask a question)
10. **Local RAG Search** → Finds relevant entities and their associated chunks
11. **Global RAG Search** → Finds relevant communities and their chunks
12. **Combine Results** → Merges local and global search results
13. **Generate Answer** → Uses LLM to generate answer from retrieved context

---

## Troubleshooting

### Issue: "spaCy model not found"
**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Issue: "Ollama connection error"
**Solution:**
1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```
2. Check if mistral:7b is installed:
   ```bash
   ollama list
   ```
3. If not installed:
   ```bash
   ollama pull mistral:7b
   ```

### Issue: "PDF not found"
**Solution:**
- Make sure `Ambedkar_works.pdf` is in the `data/` directory
- Check the path: `data/Ambedkar_works.pdf`

### Issue: "Import errors"
**Solution:**
```bash
pip install -r requirements.txt
```

---

## Expected Processing Times

- **PDF Loading**: 1-5 seconds
- **Sentence Splitting**: 1-2 seconds
- **Semantic Chunking**: 5-15 minutes (depends on PDF size)
- **Knowledge Graph**: 2-10 minutes (depends on number of entities)
- **Community Summarization**: 5-20 minutes (depends on number of communities and LLM speed)
- **Query Answering**: 10-30 seconds per question

**Total Initial Processing**: 15-50 minutes (one-time, depends on PDF size)

---

## Quick Start (Minimal Steps)

If you just want to test quickly:

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Setup Ollama
ollama pull mistral:7b

# 3. Place PDF in data/ directory

# 4. Run quick check
python check_graph.py
```

That's it! The system will process everything and show you the results.

