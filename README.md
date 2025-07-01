# 🧠 ModelBox

**ModelBox** is a lightweight, self-hosted API for downloading and running open-source large language models (LLMs) like LLaMA3, Mistral, and Gemma — without relying on Ollama or cloud-based inference.

> ✅ 100% local  
> 🚀 FastAPI-powered  
> 🔐 No internet inference calls  
> 🧩 Extendable for quantized models or custom workflows

---

## 📦 Features

- Pulls and caches Hugging Face LLMs
- Runs models directly using `transformers` and `torch`
- FastAPI interface for loading models and generating responses
- Modular design for plug-and-play models
- Fully local — private by design

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/PawanRamaMali/modelbox.git
cd modelbox
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
uvicorn main:app --reload
```

---

## ⚙️ API Endpoints

### `POST /load_model/`

Downloads and loads an LLM into memory.

**Query param:**

* `model` (e.g., `llama3`, `mistral`, `gemma`)

```bash
curl -X POST 'http://localhost:8000/load_model/?model=llama3'
```

---

### `POST /generate/`

Sends a prompt to the selected model and gets a generated response.

**Body:**

```json
{
  "model": "llama3",
  "prompt": "Explain quantum mechanics simply.",
  "max_tokens": 200
}
```

---

## 📚 Supported Models

| Model Name | Hugging Face ID                       |
| ---------- | ------------------------------------- |
| `llama3`   | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `mistral`  | `mistralai/Mistral-7B-Instruct-v0.2`  |
| `gemma`    | `google/gemma-7b-it`                  |

> You can extend `SUPPORTED_MODELS` in `main.py` to add your own.

---

## 🧠 Tech Stack

* [FastAPI](https://fastapi.tiangolo.com/) for serving the API
* [Transformers](https://huggingface.co/docs/transformers) for model loading
* [Torch](https://pytorch.org/) for running the models

---

## 🔒 Privacy First

ModelBox keeps everything local:

* No third-party API calls
* No telemetry
* No external dependencies once models are downloaded

---

## 📌 Roadmap Ideas

* [ ] Add support for quantized models (e.g., GPTQ)
* [ ] Model unloading / memory management
* [ ] Token usage & latency metrics
* [ ] Streaming output support
* [ ] Local model UI

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## ✨ Credits

Inspired by the need for **fully local, open-source LLM APIs** that are production-ready but lightweight and easy to use.

---

> Built with ❤️ by [Pawan Rama Mali](https://github.com/PawanRamaMali)

