from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ModelBox - Public Model API")

# Model storage directory
MODEL_CACHE_DIR = Path("./model_cache")
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# Publicly available models (no authentication required)
SUPPORTED_MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "distilgpt2": "distilgpt2",
    "microsoft/DialoGPT-medium": "microsoft/DialoGPT-medium",
    "microsoft/DialoGPT-small": "microsoft/DialoGPT-small",
    "EleutherAI/gpt-neo-125M": "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B": "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-j-6B": "EleutherAI/gpt-j-6B",
    "facebook/opt-125m": "facebook/opt-125m",
    "facebook/opt-350m": "facebook/opt-350m",
    "facebook/opt-1.3b": "facebook/opt-1.3b",
    "bigscience/bloom-560m": "bigscience/bloom-560m",
    "bigscience/bloom-1b1": "bigscience/bloom-1b1",
    "bigscience/bloom-3b": "bigscience/bloom-3b",
    "google/flan-t5-small": "google/flan-t5-small",
    "google/flan-t5-base": "google/flan-t5-base",
    "google/flan-t5-large": "google/flan-t5-large",
    "microsoft/phi-1": "microsoft/phi-1",
    "microsoft/phi-1_5": "microsoft/phi-1_5",
    "microsoft/phi-2": "microsoft/phi-2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "stabilityai/stablelm-base-alpha-3b": "stabilityai/stablelm-base-alpha-3b",
    "stabilityai/stablelm-base-alpha-7b": "stabilityai/stablelm-base-alpha-7b"
}

# Store loaded models
model_store = {}

class PromptRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class ModelConfig(BaseModel):
    name: str
    model_id: str
    url: Optional[str] = None
    description: Optional[str] = None

class CustomModelRequest(BaseModel):
    name: str
    model_id_or_url: str
    description: Optional[str] = None

def download_model_from_url(url: str, model_name: str) -> str:
    """Download model files from a direct URL"""
    try:
        model_dir = MODEL_CACHE_DIR / model_name
        model_dir.mkdir(exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Determine file name from URL or use default
        filename = url.split('/')[-1] if '/' in url else 'model.bin'
        file_path = model_dir / filename
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded model to {file_path}")
        return str(model_dir)
    except Exception as e:
        logger.error(f"Failed to download model from {url}: {str(e)}")
        raise

def is_model_available(model_id: str) -> bool:
    """Check if a model is publicly available on Hugging Face"""
    try:
        # Try to access model info without authentication
        from huggingface_hub import model_info
        info = model_info(model_id, token=False)
        return True
    except Exception:
        return False

def load_model_safe(model_id: str):
    """Load model safely without authentication"""
    try:
        # Try loading without authentication first
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=False
        )
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            token=False
        )
        
        return tokenizer, model_obj
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {str(e)}")
        raise

@app.get("/")
def root():
    return {
        "message": "ModelBox API is live - Public Models Only",
        "version": "2.0",
        "features": [
            "Public model support",
            "Custom model URLs",
            "No authentication required",
            "Local model caching"
        ]
    }

@app.get("/models")
def list_models():
    """List all supported models"""
    return {
        "supported_models": SUPPORTED_MODELS,
        "loaded_models": list(model_store.keys()),
        "total_supported": len(SUPPORTED_MODELS)
    }

@app.post("/load_model/")
def load_model(model: str):
    """Load a model from the supported list"""
    if model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported model. Available models: {list(SUPPORTED_MODELS.keys())}"
        )

    try:
        model_id = SUPPORTED_MODELS[model]
        
        # Check if model is already loaded
        if model in model_store:
            return {"status": "already_loaded", "message": f"{model} is already loaded"}
        
        logger.info(f"Loading model: {model_id}")
        tokenizer, model_obj = load_model_safe(model_id)
        
        model_store[model] = {
            "tokenizer": tokenizer, 
            "model": model_obj,
            "model_id": model_id
        }
        
        return {
            "status": "success", 
            "message": f"{model} loaded successfully",
            "model_id": model_id
        }
    except Exception as e:
        logger.error(f"Model load failed for {model}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model load failed: {str(e)}")

@app.post("/load_custom_model/")
def load_custom_model(request: CustomModelRequest):
    """Load a custom model by Hugging Face ID or direct URL"""
    try:
        model_name = request.name
        
        # Check if model is already loaded
        if model_name in model_store:
            return {"status": "already_loaded", "message": f"{model_name} is already loaded"}
        
        # Handle direct URLs
        if request.model_id_or_url.startswith(('http://', 'https://')):
            # For direct URLs, you'd need to implement specific model format handling
            # This is a placeholder for URL-based model loading
            raise HTTPException(
                status_code=501, 
                detail="Direct URL model loading not yet implemented. Use Hugging Face model IDs."
            )
        
        # Handle Hugging Face model IDs
        model_id = request.model_id_or_url
        
        # Check if model is publicly available
        if not is_model_available(model_id):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_id} is not publicly available or requires authentication"
            )
        
        logger.info(f"Loading custom model: {model_id}")
        tokenizer, model_obj = load_model_safe(model_id)
        
        model_store[model_name] = {
            "tokenizer": tokenizer, 
            "model": model_obj,
            "model_id": model_id,
            "description": request.description
        }
        
        return {
            "status": "success", 
            "message": f"Custom model '{model_name}' loaded successfully",
            "model_id": model_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Custom model load failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Custom model load failed: {str(e)}")

@app.post("/generate/")
def generate(request: PromptRequest):
    """Generate text using a loaded model"""
    if request.model not in model_store:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model}' not loaded. Available models: {list(model_store.keys())}"
        )

    try:
        tokenizer = model_store[request.model]["tokenizer"]
        model = model_store[request.model]["model"]

        # Tokenize input
        inputs = tokenizer(
            request.prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the response
        if response.startswith(request.prompt):
            response = response[len(request.prompt):].strip()
        
        return {
            "response": response,
            "model": request.model,
            "model_id": model_store[request.model]["model_id"]
        }
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.delete("/unload_model/")
def unload_model(model: str):
    """Unload a model from memory"""
    if model not in model_store:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not loaded")
    
    try:
        del model_store[model]
        torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
        return {"status": "success", "message": f"Model '{model}' unloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to unload model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "loaded_models": len(model_store),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }