# chatbot.py
import os
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

# Critical NumPy initialization for FAISS
np._DummyFunc = lambda: None

# Initialize Flask
app = Flask(__name__)

class SmartChatbot:
    def __init__(self, 
                 model_name="mistralai/Mistral-7B-Instruct-v0.2",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        
        # Authenticate with Hugging Face (REPLACE WITH YOUR TOKEN)
        login(token="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")
        
        # Configure 4-bit quantization
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda'}
        )
        
        # Load knowledge base
        self.db = None
        if os.path.exists("faiss_index"):
            try:
                self.db = FAISS.load_local(
                    "faiss_index",
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Vector store error: {e}")
        
        # Load LLM with error handling and fallbacks
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Attempt 4-bit with FlashAttention2
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=self.bnb_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True
                )
                print(f"✅ Loaded {model_name} in 4-bit with FlashAttention2")
            
            # Fallback to standard 4-bit
            except Exception as e:
                print(f"FlashAttention2 failed: {e}. Falling back to standard attention.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=self.bnb_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
                print(f"✅ Loaded {model_name} in 4-bit without FlashAttention2")
        
        except Exception as e:
            print(f"Model load failed: {e}")
            self.model = None

    def retrieve_context(self, query, top_k=3):
        """Retrieve context from documents"""
        if not self.db:
            return ""
        
        try:
            docs = self.db.similarity_search(query, k=top_k)
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Retrieval error: {e}")
            return ""

    def generate_response(self, user_query):
        if not self.model:
            return "System error: Model not loaded"
        
        context = self.retrieve_context(user_query)
        
        # Enhanced prompt template
        prompt = f"""Answer truthfully using ONLY this context. If answer isn't in context, say "I don't know."

        CONTEXT:
        {context}

        QUESTION: {user_query}
        ANSWER:"""
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=4096,
                truncation=True
            ).to(self.model.device)
            
            # Generate response with GPU optimization
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Validate response
            if "don't know" in response.lower() or not context:
                return "I couldn't find relevant information in the documents."
                
            return response
        
        except Exception as e:
            print(f"Generation error: {e}")
            return "Error processing request."

# Initialize chatbot
chatbot = SmartChatbot() if os.path.exists("faiss_index") else None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def handle_chat():
    if not chatbot:
        return jsonify({"response": "System not initialized"}), 500
    
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Empty query"}), 400
    
    try:
        response = chatbot.generate_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"response": "Service unavailable"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)