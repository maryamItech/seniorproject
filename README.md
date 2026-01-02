# RAG Project - LangChain + LangGraph Architecture

A production-grade RAG (Retrieval-Augmented Generation) system built with LangChain and LangGraph, featuring modular architecture and state machine orchestration.

## 🏗️ Architecture

### Technology Stack
- **LangChain**: RAG pipeline and chain orchestration
- **LangGraph**: State machine workflow orchestration
- **FAISS**: Vector store for similarity search
- **BGE Embeddings**: BAAI/bge-small-en for text embeddings
- **OpenRouter API**: LLM provider (DeepSeek Chat)
- **Streamlit**: Web interface

### Project Structure

```
ragproject/
├── config/              # Configuration settings
│   ├── __init__.py
│   └── settings.py      # API keys, model settings, paths
├── chains/              # LangChain chains
│   ├── __init__.py
│   ├── question_processing.py  # Correction, rephrasing, splitting chains
│   └── rag_chain.py            # RAG retrieval and answer generation
├── graph/               # LangGraph state machine
│   ├── __init__.py
│   └── rag_graph.py     # Workflow orchestration
├── retrieval/           # Vector store
│   ├── __init__.py
│   └── faiss_store.py   # FAISS wrapper for LangChain
├── utils/               # Utilities
│   ├── __init__.py
│   └── embeddings.py    # BGE embedding wrapper
├── rag/                 # Application
│   └── app.py           # Streamlit interface
├── embeddings/          # Pre-built embeddings
│   ├── build_embeddings.py
│   ├── index.faiss
│   ├── chunks.json
│   └── vectors.npy
├── data/                # Data files
│   └── science_qa.csv
└── requirements.txt
```

## 🔄 Workflow

The system uses LangGraph to orchestrate a 5-stage workflow:

1. **Question Correction** - Fixes spelling mistakes
2. **Question Rephrasing** - Improves clarity and structure
3. **Question Splitting** - Decomposes complex questions into parts
4. **Retrieval** - Finds relevant context from FAISS vector store
5. **Answer Generation** - Generates final answer using retrieved context

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Update `config/settings.py` with your OpenRouter API key, or set environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

### 3. Build Embeddings (if needed)

If embeddings are not already built:

```bash
cd embeddings
python build_embeddings.py
```

### 4. Run the Application

```bash
streamlit run rag/app.py
```

### 5. Download ScienceQA Images

Use the helper script to download all ScienceQA images (train/validation/test) locally:

```bash
python scripts/download_scienceqa_images.py
```

Images will be saved under `scienceqa_images/train`, `scienceqa_images/validation`, and `scienceqa_images/test` as PNG files.

### 6. Build ScienceQA Image Index

```bash
python scripts/build_image_index.py
```

Creates `embeddings/scienceqa_image_index.faiss` and metadata for image retrieval.

### 7. Graph Visualization (dev mode)

```bash
python graph/rag_graph.py  # or call rag_graph.print_graph_structure()
```

Textual graph structure is saved to `graph/graphs/graph_structure.txt`. Set `DEV_MODE=true` to show matplotlib visualization.

## 📦 Modules

### Configuration (`config/settings.py`)
Centralized configuration for:
- API keys and endpoints
- Model names and parameters
- File paths
- RAG parameters (retrieval K, temperatures)

### Question Processing Chains (`chains/question_processing.py`)
Three independent LangChain chains:
- `QuestionCorrectionChain`: Corrects spelling mistakes
- `QuestionRephrasingChain`: Rephrases questions professionally
- `QuestionSplittingChain`: Splits complex questions into parts
- `CombinedQuestionProcessingChain`: Orchestrates all three

### RAG Chain (`chains/rag_chain.py`)
- Retrieves relevant documents from FAISS
- Generates answers using retrieved context
- Uses LangChain's document chain for answer generation

### LangGraph State Machine (`graph/rag_graph.py`)
Orchestrates the complete workflow:
- State definition with TypedDict
- Node functions for each stage
- Edge definitions for workflow
- Error handling at each stage

### FAISS Vector Store (`retrieval/faiss_store.py`)
- Wraps FAISS index for LangChain compatibility
- Implements similarity search with scores
- Loads pre-built embeddings and chunks

### Embeddings (`utils/embeddings.py`)
- BGE embedding model wrapper
- Query and document embedding methods
- Handles BGE-specific prefixes

## 🎯 Features

### Production-Grade Architecture
- **Modular Design**: Each component is independent and testable
- **State Machine**: LangGraph provides clear workflow visualization
- **Error Handling**: Graceful degradation at each stage
- **Caching**: Streamlit caching for performance

### Independent Chains
Each processing step is a separate chain, allowing:
- Individual testing and debugging
- Easy modification of specific stages
- Parallel processing (future enhancement)

### Maintainability
- Clean separation of concerns
- Type hints throughout
- Comprehensive error handling
- Configuration management

## 🔧 Customization

### Adjust Retrieval Parameters
Edit `config/settings.py`:
```python
RETRIEVAL_K = 5  # Number of documents to retrieve
TEMPERATURE_GENERATION = 0.3  # LLM temperature
```

### Modify Prompts
Edit prompts in:
- `chains/question_processing.py` - Question processing prompts
- `chains/rag_chain.py` - Answer generation prompt

### Change LLM Model
Update `config/settings.py`:
```python
LLM_MODEL = "openai/gpt-4"  # Or any OpenRouter model
```

## 📊 Usage Example

```python
from retrieval.faiss_store import FAISSVectorStore
from graph.rag_graph import RAGGraph

# Load system
vector_store = FAISSVectorStore.load()
rag_graph = RAGGraph(vector_store)

# Process question
result = rag_graph.run("What is photosynthesis?")

# Access results
print(result["answer"])
print(result["rephrased_question"])
print(result["retrieved_docs"])
```

## 🐛 Troubleshooting

### Import Errors
Ensure project root is in Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### FAISS Index Not Found
Run `embeddings/build_embeddings.py` to generate the index.

### API Errors
Check your OpenRouter API key in `config/settings.py` or environment variables.

## 📝 License

This project is provided as-is for educational and development purposes.

