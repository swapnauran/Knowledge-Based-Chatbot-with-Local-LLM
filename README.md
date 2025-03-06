# Document-Based Chatbot with LLM

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions based on your documents using **Mistral-7B**. Processes **PDF/DOCX** files and runs **locally** or **Deploy-any-where**.

&#x20;

## 🚀 Features

- 📁 Process **PDF/DOCX** documents
- 🔍 Semantic search with **FAISS**
- 🧠 **Mistral-7B** LLM with **4-bit quantization**
- ⚡ **GPU acceleration** (NVIDIA required)
- 🔒 Runs **locally or deploy any where**

## 🛠 Installation

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** 
- **CUDA 11.8**
- **16GB+ RAM**
- **20GB+ free disk space**

### Step-by-Step Setup

1. **Clone Repository**
   ```bash
   https://github.com/swapnauran/Knowledge-Based-Chatbot-with-Local-LLM.git
   ```
2. **Install Requirements**
   - **For CUDA 11.8 (Recommended):**
     ```bash
     python -m pip install -r requirements_cuda.txt
     ```
   - **For CPU-Only (Not Recommended):**
     ```bash
     python -m pip install -r requirements.txt
     ```
3. **Install FlashAttention (Windows Only)**
   ```bash
   pip install flash-attn==2.5.8 --no-build-isolation
   ```

## 🚀 Usage

### 1️⃣ Prepare Documents

- Place files in `company_docs/` 
- Supported formats: **PDF, DOCX, DOC**
- Max recommended file size: **50GB each**

### 2️⃣ Train Model

```bash
python train_model.py
```

### 3️⃣ Configure Chatbot

- Get **Hugging Face token**:
  - Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
  - Create token with **Write access**
- Edit `chatbot.py`:
  ```python
  login(token="YOUR_HF_TOKEN_HERE")  # Line 23
  ```

### 4️⃣ Start Chatbot

```bash
python chatbot.py
```

- Access at: **[http://localhost:5000](http://localhost:5000)**

## 📂 File Structure

```
.
├── company_docs/       # Your documents go here
├── faiss_index/        # Auto-created after training
├── templates/
│   └── index.html      # Web interface
├── train_model.py      # Document processing
├── chatbot.py          # Chatbot server
├── requirements.txt    # CPU-only dependencies
└── requirements_cuda.txt # GPU-optimized dependencies
```

## 🔧 Requirements Files

### **GPU-Optimized (********`requirements_cuda.txt`********):**

```text
numpy==1.26.4
langchain==0.3.20
langchain-community==0.3.1
flask==3.0.3
sentence-transformers==3.4.1
torch==2.2.0+cu118
torchvision==0.17.0+cu118
--extra-index-url https://download.pytorch.org/whl/cu118
bitsandbytes==0.43.1
accelerate==0.30.1
transformers==4.41.2
```

### **CPU-Only (********`requirements.txt`********):**

```text
numpy==1.26.4
langchain==0.3.20
langchain-community==0.3.1
flask==3.0.3
sentence-transformers==3.4.1
torch==2.2.0
torchvision==0.17.0
bitsandbytes==0.43.1
accelerate==0.30.1
transformers==4.41.2
```

## 🔍 Troubleshooting

### ❌ CUDA Errors

```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### 🔑 Hugging Face Authentication

Ensure you accepted **Mistral-7B** license:
[https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

### 🛑 Memory Errors

Reduce chunk size in `train_model.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Reduced from 1000
    chunk_overlap=100
)
```

## ⚡ Performance Tips

- **Use 4-bit quantization** for GPUs with **<8GB VRAM**
- **Restart chatbot** after changing documents

## 📜 License

**MIT License** 

