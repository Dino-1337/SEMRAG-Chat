# LangChain vs Custom Implementation Breakdown

This document outlines which parts of the SEMRAG RAG system use LangChain imports and which parts use custom-built functions.

## 📦 LangChain Usage

### 1. **Data Loader** (`src/data_loader.py`)
**LangChain Imports:**
- `from langchain_community.document_loaders import PyPDFLoader`
- `from langchain.text_splitter import RecursiveCharacterTextSplitter`

**Usage:**
- ✅ **PyPDFLoader**: Used as **fallback** when `pdfplumber` fails (line 42-45)
- ⚠️ **RecursiveCharacterTextSplitter**: **Imported but NOT used** - custom regex-based sentence splitting is used instead (line 63-66)
- ✅ **Primary PDF loading**: Uses `pdfplumber` (custom), LangChain is only fallback

**Summary**: Minimal LangChain usage - only as fallback for PDF loading.

---

### 2. **LLM Integration** (`src/llm_integration.py`)
**LangChain Imports:**
- `from langchain_community.llms import Ollama`
- `from langchain.prompts import PromptTemplate`

**Usage:**
- ✅ **Ollama LLM**: Used for all LLM operations (lines 27-32)
  - Community summary generation (line 68)
  - Answer generation (line 138)
- ✅ **PromptTemplate**: Used for structured prompt creation (lines 54-63, 114-128)

**Summary**: **Heavy LangChain usage** - core LLM functionality relies on LangChain's Ollama integration and prompt templates.

---

## 🔧 Custom-Built Functions

### 1. **Semantic Chunking** (`src/semantic_chunking.py`)
**Custom Implementation:**
- ✅ **Algorithm 1**: Complete custom implementation of SEMRAG semantic chunking
  - `semantic_chunking()`: Cosine similarity-based sentence grouping (lines 67-130)
  - `apply_buffer_merging()`: Context preservation at boundaries (lines 132-178)
  - `enforce_token_limits()`: Token limit enforcement with sub-chunking (lines 180-195)
  - `_split_large_chunk()`: Custom sub-chunking with overlap (lines 197-244)

**External Libraries Used (Not LangChain):**
- `sentence-transformers`: For embeddings (not LangChain)
- `sklearn.metrics.pairwise`: For cosine similarity
- `tiktoken`: For token counting

**Summary**: **100% custom** - implements SEMRAG Algorithm 1 from scratch.

---

### 2. **Knowledge Graph** (`src/knowledge_graph.py`)
**Custom Implementation:**
- ✅ **Entity Extraction**: Custom implementation using spaCy NER (lines 62-95)
- ✅ **Relationship Extraction**: Custom dependency parsing logic (lines 97-150)
  - `_extract_relation()`: Custom relation type detection (lines 152-180)
- ✅ **Graph Construction**: Custom NetworkX graph building (lines 182-237)
- ✅ **Community Detection**: Custom implementation with Leiden/Louvain (lines 239-320)
  - `_detect_communities_leiden()`: Custom Leiden algorithm wrapper
  - `_detect_communities_louvain()`: Custom fallback using NetworkX

**External Libraries Used (Not LangChain):**
- `spacy`: For NER and dependency parsing
- `networkx`: For graph construction
- `leidenalg` / `igraph`: For community detection

**Summary**: **100% custom** - all graph construction logic is custom-built.

---

### 3. **Retrieval System** (`src/retrieval.py`)
**Custom Implementation:**
- ✅ **Local RAG Search (Equation 4)**: Complete custom implementation (lines 48-120)
  - Entity-based chunk retrieval
  - Cosine similarity ranking
  - Threshold filtering
- ✅ **Global RAG Search (Equation 5)**: Complete custom implementation (lines 122-200)
  - Community-based chunk retrieval
  - Weighted scoring
- ✅ **Result Combination**: Custom strategies (lines 202-270)
  - `combine_results()`: Weighted/union/intersection strategies

**External Libraries Used (Not LangChain):**
- `sentence-transformers`: For embeddings
- `sklearn.metrics.pairwise`: For cosine similarity
- `networkx`: For graph traversal

**Summary**: **100% custom** - implements SEMRAG Equations 4 and 5 from scratch.

---

### 4. **Data Loader** (`src/data_loader.py`)
**Custom Implementation:**
- ✅ **Primary PDF Loading**: Custom `pdfplumber` implementation (lines 29-39)
- ✅ **Sentence Splitting**: Custom regex-based splitting (lines 48-68)
- ✅ **Page Text Extraction**: Custom page-by-page extraction (lines 70-100)

**Summary**: **Mostly custom** - LangChain only used as fallback.

---

### 5. **Main RAG System** (`src/app.py`)
**Custom Implementation:**
- ✅ **Orchestration**: Complete custom orchestration logic
- ✅ **Pipeline Management**: Custom processing pipeline
- ✅ **Query Processing**: Custom end-to-end query handling

**Summary**: **100% custom** - orchestrates all components.

---

## 📊 Summary Table

| Module | LangChain Usage | Custom Implementation | Notes |
|--------|----------------|----------------------|-------|
| **Data Loader** | ⚠️ Minimal (fallback only) | ✅ Primary implementation | Uses `pdfplumber` primarily |
| **Semantic Chunking** | ❌ None | ✅ 100% custom | Algorithm 1 implementation |
| **Knowledge Graph** | ❌ None | ✅ 100% custom | Full graph construction |
| **Retrieval** | ❌ None | ✅ 100% custom | Equations 4 & 5 |
| **LLM Integration** | ✅ Heavy usage | ⚠️ Prompt engineering | Uses LangChain's Ollama |
| **RAG System** | ❌ None | ✅ 100% custom | Orchestration |

## 🎯 Key Takeaways

1. **LangChain is primarily used for:**
   - LLM integration (Ollama wrapper)
   - Prompt template management
   - PDF loading fallback (not primary)

2. **Custom implementations handle:**
   - ✅ SEMRAG Algorithm 1 (semantic chunking)
   - ✅ Knowledge graph construction
   - ✅ Entity and relationship extraction
   - ✅ Local RAG (Equation 4)
   - ✅ Global RAG (Equation 5)
   - ✅ Retrieval and ranking
   - ✅ System orchestration

3. **Why this approach?**
   - SEMRAG paper requires specific algorithms that aren't in LangChain
   - Custom implementation ensures exact compliance with paper specifications
   - LangChain provides convenient LLM integration without constraining the architecture

## 🔄 Could LangChain be used more?

**Potential additions (but not implemented):**
- ❌ LangChain's `RecursiveCharacterTextSplitter` - Not used because SEMRAG requires semantic similarity-based chunking, not character-based
- ❌ LangChain's `VectorStore` - Not used because we need graph-based retrieval (entities + communities)
- ❌ LangChain's `Retrievers` - Not used because we implement custom Local/Global RAG strategies
- ✅ LangChain's `Ollama` - Already used for LLM calls
- ✅ LangChain's `PromptTemplate` - Already used for prompt management

**Conclusion**: The current implementation uses LangChain appropriately - for LLM integration where it adds value, while keeping custom implementations for SEMRAG-specific algorithms.

