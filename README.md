# Knowledge-Based-Chatbot-with-LLM
A RAG (Retrieval-Augmented Generation) chatbot that answers questions based on your documents using Mistral-7B. Processes PDF/DOCX files and runs locally or deploy

# Document-Based Chatbot with Local LLM

A RAG (Retrieval-Augmented Generation) chatbot that answers questions based on your documents using Mistral-7B. Processes PDF/DOCX files and runs locally on your machine.

![Demo](demo-screenshot.png) <!-- Add actual screenshot later -->

## Features

- 📁 Process PDF/DOCX documents
- 🔍 Semantic search with FAISS
- 🧠 Mistral-7B LLM with 4-bit quantization
- 🚀 GPU acceleration (NVIDIA required)
- 🔒 Runs entirely locally

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU (RTX 20xx or newer recommended)
- CUDA 11.8
- 16GB+ RAM
- 20GB+ free disk space

### Step-by-Step Setup
0. **Create Virtual Environment**
1. **Clone Repository**
```bash
https://github.com/swapnauran/Knowledge-Based-Chatbot-with-Local-LLM.git

2. Install Requirements (Choose one)

For CUDA 11.8 (Recommended):
python -m pip install -r requirements_cuda.txt
