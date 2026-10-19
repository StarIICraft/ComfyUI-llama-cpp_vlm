
# ComfyUI-llama-cpp

Run local LLM/VLM models and access OpenAI-compatible APIs in ComfyUI.

**[中文版](./README_zh.md)**

<img width="1976" height="1654" alt="image" src="https://github.com/user-attachments/assets/20d16bea-db9e-462d-98b4-ae817512d307" />


## Features

### Local Models
- Load GGUF models through llama.cpp; vision inference requires matching mmproj weights.
- Ten image inputs, each accepting a single image or an IMAGE batch.
- Group-by-group, combined-image, and sampled video-frame inference.
- Per-image megapixel limits, automatic model unloading, and conversation state.
- Bounding-box, mask, JSON utilities, and an experimental text encoder.

### API Models
- Two nodes: API LLM Loader and API LLM General Chain.
- Model-list refresh, combined-image inference, group-by-group inference, and string-list output.
- Tavily web search and SSE streaming.
- Streamed text appears in the backend console; nodes return complete results after generation.

## Installation

```Wheel
https://github.com/JamePeng/llama-cpp-python/releases

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/StarIICraft/ComfyUI-llama-cpp_vlm.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
```

CUDA 13 users can use `requirements_cu131.txt` instead. Install dependencies with your ComfyUI Python environment, restart ComfyUI, and refresh the browser.

## Configuration

### Local Inference
Place GGUF files in `ComfyUI/models/LLM`. For vision models, select the corresponding mmproj file.

### API Inference
Enter the base URL, API key, and model name in the API loader. Do not include `/chat/completions` in the base URL.

### Web Search
Create `search_config.json` in the plugin directory:

```json
{
  "tavily_api_key": "YOUR_SEARCH_KEY",
  "max_results": 5
}
```

Search settings are reloaded when search runs.

## Notes

- Each IMAGE batch counts as one group. Ten inputs do not mean a ten-image limit.
- Multi-image support and request limits depend on the API provider and model.
- Search sends keywords to Tavily; prompts and images are sent to your model provider.
- Remove API keys before sharing workflows. Never publish your search configuration.
- API nodes currently use Chinese interface labels.
- This combined plugin still requires its local-model dependencies.

## Credits

- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) — @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) — @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — @comfyanonymous
