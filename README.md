# Knowledge-Based-Chatbot-with-LLM
A RAG (Retrieval-Augmented Generation) chatbot that answers questions based on your documents using Mistral-7B. Processes PDF/DOCX files and runs locally or deploy


![Demo](demo-screenshot.png) <!-- Add actual screenshot later -->

## Features

- 📁 Process PDF/DOCX documents
- 🔍 Semantic search with FAISS
- 🧠 Mistral-7B LLM with 4-bit quantization
- 🚀 GPU acceleration (NVIDIA required)
- 🔒 Runs locally or deploy any where 

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

Install Requirements (Choose one)

For CUDA 11.8 (Recommended):

bash
Copy
python -m pip install -r requirements_cuda.txt
For CPU-Only (Not Recommended):

bash
Copy
python -m pip install -r requirements.txt
Install FlashAttention (Windows Only)

bash
Copy
pip install flash-attn==2.5.8 --no-build-isolation
Usage
1. Prepare Documents
Place your files in company_docs/ folder:

Supported formats: PDF, DOCX, DOC

Max recommended file size: 50MB each

2. Train Model
bash
Copy
python train_model.py
