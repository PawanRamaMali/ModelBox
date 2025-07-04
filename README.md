# 🧠 ModelBox v2.0

**ModelBox** is a lightweight, self-hosted API for downloading and running open-source large language models (LLMs) from public repositories — completely independent of authentication or gated models.

> ✅ 100% local & public models only  
> 🚀 FastAPI-powered  
> 🔓 No authentication required  
> 🧩 Extendable with custom models  
> 🎯 Works with any publicly available model

---

## 📦 Features

- **Public Models Only**: Works exclusively with publicly available models
- **No Authentication**: No need for Hugging Face tokens or API keys
- **Custom Model Support**: Load any public model by Hugging Face ID
- **Memory Management**: Load/unload models as needed
- **Direct URL Support**: Framework for loading models from direct URLs (coming soon)
- **Health Monitoring**: Built-in health checks and monitoring
- **GPU Support**: Automatic GPU detection and utilization

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/PawanRamaMali/modelbox.git
cd modelbox
```

### 2. Install dependencies
```bash
pip install -r app/requirements.txt
```

### 3. Run the API
```bash
uvicorn app.main:app --reload
```

### 4. Test the API
```bash
# Check available models
curl http://localhost:8000/models

# Load a model
curl -X POST 'http://localhost:8000/load_model/?model=gpt2'

# Generate text
curl -X POST 'http://localhost:8000/generate/' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt2",
    "prompt": "The future of AI is",
    "max_tokens": 100
  }'
```

---

## 🐳 Using Docker

```bash
# Build the image
docker build -t modelbox .

# Run the container
docker run -p 8000:8000 modelbox

# Run with GPU support (if available)
docker run --gpus all -p 8000:8000 modelbox
```

---

## 📁 Project Structure

```
ModelBox/
├── Dockerfile
├── README.md
└── app/
    ├── main.py
    ├── requirements.txt
    └── model_cache/     # Local model storage
```

---

## ⚙️ API Endpoints

### `GET /`
Get API information and status

### `GET /models`
List all supported and loaded models

### `GET /health`
Health check endpoint

### `POST /load_model/`
Load a model from the supported list

**Query params:**
- `model`: Model name from supported list (e.g., `gpt2`, `distilgpt2`)

```bash
curl -X POST 'http://localhost:8000/load_model/?model=gpt2'
```

### `POST /load_custom_model/`
Load a custom model by Hugging Face ID

**Body:**
```json
{
  "name": "my-custom-model",
  "model_id_or_url": "microsoft/DialoGPT-large",
  "description": "Custom model for dialogue"
}
```

### `POST /generate/`
Generate text using a loaded model

**Body:**
```json
{
  "model": "gpt2",
  "prompt": "The future of artificial intelligence",
  "max_tokens": 200,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true
}
```

### `DELETE /unload_model/`
Unload a model from memory

**Query params:**
- `model`: Name of the model to unload

```bash
curl -X DELETE 'http://localhost:8000/unload_model/?model=gpt2'
```

---

## 📚 Supported Models

### Small Models (< 1B parameters)
- `gpt2` - OpenAI GPT-2 (117M)
- `distilgpt2` - Distilled GPT-2 (82M)
- `microsoft/DialoGPT-small` - Conversational AI (117M)
- `EleutherAI/gpt-neo-125M` - GPT-Neo (125M)
- `facebook/opt-125m` - OPT (125M)
- `google/flan-t5-small` - Flan-T5 (77M)

### Medium Models (1B-3B parameters)
- `gpt2-medium` - OpenAI GPT-2 (345M)
- `microsoft/DialoGPT-medium` - Conversational AI (345M)
- `EleutherAI/gpt-neo-1.3B` - GPT-Neo (1.3B)
- `facebook/opt-1.3b` - OPT (1.3B)
- `bigscience/bloom-1b1` - BLOOM (1.1B)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - TinyLlama (1.1B)

### Large Models (3B+ parameters)
- `gpt2-large` - OpenAI GPT-2 (774M)
- `EleutherAI/gpt-j-6B` - GPT-J (6B)
- `bigscience/bloom-3b` - BLOOM (3B)
- `microsoft/phi-2` - Phi-2 (2.7B)
- `google/flan-t5-large` - Flan-T5 (770M)

> **Note**: Large models require significant RAM/VRAM. Use smaller models for testing.

---

## 🔧 Configuration

### Environment Variables
- `MODEL_CACHE_DIR`: Directory for model storage (default: `./model_cache`)
- `MAX_MODELS`: Maximum number of models to keep in memory (default: unlimited)
- `CUDA_VISIBLE_DEVICES`: GPU devices to use

### Model Loading Options
- **Temperature**: Controls randomness (0.0-2.0)
- **Top-p**: Nucleus sampling parameter (0.0-1.0)
- **Max Tokens**: Maximum tokens to generate
- **Do Sample**: Enable/disable sampling

---

## 🚀 Performance Tips

1. **GPU Usage**: Models automatically use GPU if available
2. **Memory Management**: Unload unused models to free memory
3. **Model Size**: Start with smaller models for testing
4. **Batch Size**: Use smaller batch sizes for large models

---

## 🔒 Privacy & Security

- **100% Local**: No external API calls after model download
- **No Authentication**: No tokens or API keys required
- **Public Models Only**: Only works with publicly available models
- **Offline Capable**: Works offline after models are cached

---

## 📋 Troubleshooting

### Common Issues

1. **Model Not Loading**
   - Check if model is publicly available
   - Verify sufficient RAM/VRAM
   - Try smaller models first

2. **Generation Errors**
   - Ensure model is loaded before generating
   - Check input prompt length
   - Verify model compatibility

3. **Memory Issues**
   - Unload unused models
   - Use smaller models
   - Enable model offloading

### Debug Mode
```bash
# Run with debug logging
uvicorn app.main:app --reload --log-level debug
```

---

## 📌 Roadmap

- [x] Public model support
- [x] Custom model loading
- [x] Memory management
- [x] Health monitoring
- [ ] Direct URL model loading
- [ ] Model quantization support
- [ ] Streaming responses
- [ ] Model caching optimization
- [ ] Batch processing
- [ ] Model performance metrics

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## ✨ Credits

Built with ❤️ by [Pawan Rama Mali](https://github.com/PawanRamaMali)

**ModelBox v2.0** - Making AI accessible, private, and free for everyone.