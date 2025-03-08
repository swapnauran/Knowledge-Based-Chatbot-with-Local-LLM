# chatbot.py
import os
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ Using device: {device}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device}
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
                print("✅ Vector store loaded successfully")
            except Exception as e:
                print(f"⚠️ Vector store error: {e}")
        else:
            print("⚠️ No vector store found at 'faiss_index'")
        
        # Load LLM with error handling and fallbacks
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("✅ Set padding token equal to EOS token")
            
            # Check if FlashAttention is available
            use_flash_attn = False
            if torch.cuda.is_available():
                # Safely check if flash_attn is installed without import error
                try:
                    import importlib.util
                    flash_attn_spec = importlib.util.find_spec("flash_attn")
                    if flash_attn_spec is not None:
                        use_flash_attn = True
                        print("✅ FlashAttention2 is available")
                    else:
                        print("⚠️ FlashAttention2 not installed, using standard attention")
                except Exception:
                    print("⚠️ Error checking for FlashAttention2, using standard attention")
            
            # Load the model with appropriate settings
            load_kwargs = {
                "quantization_config": self.bnb_config,
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True
            }
            
            # Add FlashAttention only if available
            if use_flash_attn:
                load_kwargs["attn_implementation"] = "flash_attention_2"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
            print(f"✅ Loaded {model_name} successfully")
            
        except Exception as e:
            print(f"⚠️ Model load failed: {e}")
            self.model = None
            raise RuntimeError(f"Failed to initialize model: {e}")

    def retrieve_context(self, query, top_k=3):
        """Retrieve context from documents"""
        if not self.db:
            return ""
        
        try:
            docs = self.db.similarity_search(query, k=top_k)
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            return ""

    def generate_response(self, user_query):
        if not self.model:
            return "System error: Model not loaded"
        
        # For simple greeting patterns, return a direct response without retrieval
        greeting_patterns = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        if user_query.lower().strip() in greeting_patterns:
            return "Hello! How can I help you today?"
        
        # For regular queries, retrieve context
        context = self.retrieve_context(user_query)
        
        # Enhanced prompt template with clearer separation
        prompt = f"""Answer the question truthfully based only on the context provided below. If the answer isn't in the context, say "I don't have information about that."

        CONTEXT:
        {context or "No specific context available."}

        QUESTION: {user_query}
        ANSWER:"""
        
        try:
            # Create input with proper attention mask
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=4096,
                truncation=True,
                padding="max_length",
            ).to(self.model.device)
            
            # Generate response with GPU optimization
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Extract just the generated response (after the prompt)
            raw_response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Process the response to extract just the actual answer
            # This addresses the issue where model output might contain "QUESTION: ... ANSWER: ..."
            if "QUESTION:" in raw_response and "ANSWER:" in raw_response:
                # Extract just the last answer if there are multiple Q&A pairs
                parts = raw_response.split("ANSWER:")
                response = parts[-1].strip()
            else:
                response = raw_response
                
            # Check if the response indicates no information
            if "don't have information" in response.lower() or "i don't know" in response.lower():
                if not context:
                    return "I don't have any information about that in my knowledge base."
            
            return response
            
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            return "Error processing request."

# Initialize chatbot with proper error handling
chatbot = None
try:
    if os.path.exists("faiss_index"):
        chatbot = SmartChatbot()
        print("✅ Chatbot initialized successfully")
    else:
        print("⚠️ No vector store found. Please create a FAISS index before running.")
except Exception as e:
    print(f"⚠️ Failed to initialize chatbot: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/chat", methods=["POST"])
def handle_chat():
    if not chatbot:
        return jsonify({"response": "System not initialized. Please check server logs."}), 500
    
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Empty query"}), 400
    
    try:
        response = chatbot.generate_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        print(f"⚠️ Chat error: {e}")
        return jsonify({"response": "Service unavailable"}), 500

@app.route("/status")
def status():
    """Endpoint to check system status"""
    status_info = {
        "system": "online",
        "vector_store": "loaded" if chatbot and chatbot.db else "not_loaded",
        "model": "loaded" if chatbot and chatbot.model else "not_loaded",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    return jsonify(status_info)

if __name__ == "__main__":
    # Create static folder for favicon if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Create empty favicon if it doesn't exist (to avoid 404 errors)
    if not os.path.exists(os.path.join("static", "favicon.ico")):
        try:
            with open(os.path.join("static", "favicon.ico"), "w") as f:
                f.write("")
        except Exception:
            pass
            
    # Run the app
    print("✅ Starting server. Press CTRL+C to quit")
    app.run(host='0.0.0.0', port=5000)
