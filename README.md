# ComfyUI-llama-cpp  

```markdown
# ComfyUI Local & API LLM Tools

Local llama.cpp inference and OpenAI-compatible API access for ComfyUI, with text generation, multi-image analysis, and prompt enhancement.

## Features

### Local Inference
- GGUF models and mmproj vision projectors.
- Combined-image, per-group, and sampled video-frame inference.
- Context, sampling, and image-resolution controls.
- Conversation state management and model unloading policies.
- JSON parsing, bounding-box visualization, MASK and SEGS conversion.
- Prompt presets and an experimental text encoder.

Model compatibility depends on your installed llama-cpp-python version and available chat handlers.

### API Inference
- OpenAI-compatible Chat Completions.
- Model-list refresh and selection.
- Text-only and multi-image requests.
- Combined-image or per-group processing.
- SSE streaming to the backend console; complete outputs are returned when generation finishes.
- Tavily web search with file-based credentials.

## Setup

Place the plugin in `ComfyUI/custom_nodes/`, install its required dependencies, and restart ComfyUI.

Store local models and projectors in:

`ComfyUI/models/LLM/`

Select a matching model, projector, and vision handler.

For web search, create `search_config.json` in the plugin directory:

```json
{"tavily_api_key": "YOUR_KEY", "max_results": 5}

**[[📃中文版](./README_zh.md)]** 

## Preview  
<img width="1887" height="1631" alt="image" src="https://github.com/user-attachments/assets/5729dde2-8336-4f87-8c4d-188351f7c734" />


## Installation  

#### Install the node:  
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
#For CUDA 13, please run:
#python -m pip install -r ComfyUI-llama-cpp/requirements_cu131.txt
```

#### Download models:  
- Place your model files in the `ComfyUI/models/LLM` folder.  

	> If you need a VLM model to process image input, don't forget to download the `mmproj` weights.

## Credits  
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
