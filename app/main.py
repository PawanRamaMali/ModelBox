from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from huggingface_hub import login
import os

# Automatically login
login(token=os.getenv("HUGGINGFACE_TOKEN"))

app = FastAPI(title="ModelBox")

# Supported models (can add more)
SUPPORTED_MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma": "google/gemma-7b-it"
}

# Store loaded models
model_store = {}

class PromptRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 256

@app.get("/")
def root():
    return {"message": "ModelBox API is live."}

@app.post("/load_model/")
def load_model(model: str):
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model.")

    try:
        model_id = SUPPORTED_MODELS[model]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        model_store[model] = {"tokenizer": tokenizer, "model": model_obj}
        return {"status": "success", "message": f"{model} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {str(e)}")

@app.post("/generate/")
def generate(request: PromptRequest):
    if request.model not in model_store:
        raise HTTPException(status_code=400, detail="Model not loaded.")

    tokenizer = model_store[request.model]["tokenizer"]
    model = model_store[request.model]["model"]

    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            temperature=0.7
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
