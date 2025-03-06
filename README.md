# Knowledge-Based-Chatbot-with-LLM
A RAG (Retrieval-Augmented Generation) chatbot that answers questions based on your documents using Mistral-7B. Processes PDF/DOCX files and runs locally or deploy


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
0. **Create-vertual-environment**
1. **Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/document-chatbot.git
cd document-chatbot
