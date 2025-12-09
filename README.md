# SEMRAG RAG System

A fully functional RAG (Retrieval-Augmented Generation) system following the SEMRAG research paper's approach for answering questions about Dr. B.R. Ambedkar's works.

## Features

- **Semantic Chunking**: Cosine similarity-based sentence grouping with buffer merging
- **Knowledge Graph**: Entity extraction, relationship extraction, and community detection
- **Dual Retrieval**: Local RAG (entity-based) and Global RAG (community-based) search
- **LLM Integration**: Ollama integration with Mistral 7B for answer generation

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

3. Install and start Ollama (if not already running):
```bash
# Install Ollama from https://ollama.ai
# Pull the mistral model
ollama pull mistral:7b
```

4. Place your `Ambedkar_works.pdf` file in the `data/` directory

## Usage

### Basic Usage

```python
from src.pipeline.ambedkargpt import AmbedkarGPT

# Initialize system
gpt = AmbedkarGPT("config.yaml")

# Load and process PDF
gpt.load_and_process("data/Ambedkar_works.pdf")

# Ask questions
result = gpt.query("What were Dr. Ambedkar's views on social justice?")
print(result['answer'])
print(result['citations'])
```

### Run Application

```bash
python app.py
```

### Visualize Knowledge Graph

After processing the PDF, you can visualize the knowledge graph:

```bash
# Quick check if graph was created
python check_graph.py

# Create full visualizations (PNG, JSON, HTML)
python visualize_graph.py
```

This will generate:
- `knowledge_graph.png` - Static visualization
- `knowledge_graph.json` - Graph data in JSON format
- `knowledge_graph.html` - Interactive visualization (if plotly is installed)

## Project Structure

```
ambedkargpt/
├── data/
│   ├── Ambedkar_works.pdf
│   └── processed/
│       ├── chunks.json
│       └── knowledge_graph.pkl
├── src/
│   ├── chunking/
│   │   ├── semantic_chunker.py   # Algorithm 1
│   │   └── buffer_merger.py
│   ├── graph/
│   │   ├── entity_extractor.py
│   │   ├── relationship_extractor.py
│   │   ├── graph_builder.py
│   │   ├── community_detector.py
│   │   └── summarizer.py
│   ├── retrieval/
│   │   ├── local_search.py       # Equation 4
│   │   ├── global_search.py       # Equation 5
│   │   └── ranker.py
│   ├── llm/
│   │   ├── llm_client.py
│   │   ├── prompt_templates.py
│   │   └── answer_generator.py
│   ├── pipeline/
│   │   └── ambedkargpt.py        # Main pipeline
│   └── utils/
│       ├── data_loader.py
│       └── graph_visualizer.py
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_integration.py
├── config.yaml                   # Hyperparameters
├── requirements.txt
├── app.py                        # Main entry point
└── README.md
```

## Visualization

The system automatically prints detailed graph statistics after processing. You can also:

1. **Quick Check**: Run `python check_graph.py` to verify graph creation
2. **Full Visualization**: Run `python visualize_graph.py` to generate:
   - Static PNG image of the graph
   - JSON export of graph data
   - Interactive HTML visualization (requires plotly)

The graph visualization shows:
- Entities as nodes (colored by community or entity type)
- Relationships as edges
- Community groupings
- Connection statistics

## Configuration

Edit `config.yaml` to customize:
- Embedding model
- LLM settings
- Chunking parameters
- Retrieval thresholds
- Knowledge graph settings

## Architecture

1. **Semantic Chunking**: Groups sentences into semantically coherent chunks using cosine similarity
2. **Knowledge Graph**: Extracts entities and relationships, builds graph, detects communities
3. **Retrieval**: Local search (entity-based) and Global search (community-based)
4. **Generation**: LLM generates answers using retrieved context

## License

This project is for technical assignment purposes.

