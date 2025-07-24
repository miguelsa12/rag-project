# RAG System – Technical Interview

This repository contains a fully functional and well-documented implementation of a **RAG** system. Designed to run locally on a CPU, this project showcases how to integrate a knowledge base, semantic search, and a lightweight generative model to answer user queries grounded in retrieved content.

## Solution Highlights
- **Local, low-cost deployment** with CPU-friendly models
- **Modular architecture**: Easily replace or improve components
- **Clear semantic search** using embeddings and vector similarity
- **Custom or tool-based retrieval system**
- **End-to-end notebook demo** for interactive exploration
- **Production-ready structure** with scaling ideas outlined

## Deliverables
- Jupyter notebook demo
- Architecture slide deck
- Design rationale, implementation steps, and challenges

## Requirements

- This project requires **Python 3.13.5** to run. Make sure it is properly installed in your environment.

## Run Instructions

- Clone this repository
- Install dependencies: `pip install -r requirements.txt`. Make sure you're inside the project folder (the one you cloned) when running this command in your terminal
- Include the PDF files (the ones I provided via email, only if they were not cloned automatically) in the local `data/` directory
- Launch the notebook: `jupyter notebook notebooks/rag_demo.ipynb`

## LLM and Embedding Model Setup

If the models were not cloned automatically when you cloned the repository, clone them manually:

- Clone the LLM:
  git clone https://huggingface.co/tiiuae/falcon-rw-1b

- Clone the embedding model:
  git clone https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2

After cloning, ensure both directories are placed inside the models/ folder. If not, move them manually:

move .\falcon-rw-1b .\models\
move .\paraphrase-MiniLM-L6-v2 .\models\

## App 

- Run `streamlit run app.py` and wait for it to open in your web browser
- Start testing in a more interactive way
