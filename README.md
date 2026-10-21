
# ComfyUI-llama-cpp

Run LLM/VLM models natively in ComfyUI using llama.cpp, with additional API-based image generation/editing and multi-image resizing nodes.

This branch includes:

- Multi-image recognition for up to 10 images in the local LLM/VLM workflow.
- Improved automatic model unloading.
- A megapixel limit for input images.
- Universal API image generation and editing with up to 8 reference images.
- Multi-image resizing with 8 independent image inputs and outputs, image 1 mask processing, aspect-ratio padding, and resolution controls.


**[[📃中文版](./README_zh.md)]**

## Preview

<img width="1996" height="1624" alt="image" src="https://github.com/user-attachments/assets/caf29289-791c-433a-a78e-b5fd9e85ef7d" />
<img width="1880" height="1879" alt="api图像编辑-生成与多图统一缩放" src="https://github.com/user-attachments/assets/76af5597-4ad0-4b09-802f-f505be0bc98d" />


## Installation

### Install the nodes

Run the following commands using the Python environment used by ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
```

For CUDA 13, use the corresponding requirements file:

```bash
python -m pip install -r ComfyUI-llama-cpp/requirements_cu131.txt
```

Restart ComfyUI after installation. Refresh the browser to load frontend extensions.

> For ComfyUI Portable, use its bundled Python instead of an unrelated system Python environment.

### Local model files

Place `.gguf` models in:

```text
ComfyUI/models/LLM
```

For image recognition with a local VLM, also download and select the matching `mmproj` weights.

The API image node does not require downloading a local image generation model. A working API endpoint and API key are required instead.

## Universal API Image Generation and Editing

**Node:** `☁️API 图像生成与编辑`  
**Category:** `API大模型/图像生成编辑`  
**Internal ID:** `UniversalImageAPIEditor`

Use this node for text-to-image generation or reference-image editing through compatible remote services.

### Supported protocol formats

| Protocol | Default endpoint | Reference-image format |
|---|---|---|
| OpenAI Images | `/v1/images/generations` without references; `/v1/images/edits` with references | Multipart files for editing |
| Gemini native | `/v1beta/models/{model}:generateContent` | `inlineData` |
| Nano Banana OpenAI-compatible | `/v1/images/generations` | JSON `image` array containing data URLs |
| Chat Completions image | `/v1/chat/completions` | Multimodal `messages` |
| Generic JSON | `/v1/images/generations` | JSON `image` array containing data URLs |
| Generic multipart | `/v1/images/edits` | Multipart files; reference images required |

A custom endpoint can be a relative path or a complete URL.

> A protocol option describes the request format, not a guarantee that every model or provider supports it. Use the protocol documented by your API provider.

### Main settings

| Setting | Default | Description |
|---|---|---|
| Protocol | OpenAI Images | Request protocol |
| Request count | 1 | Number of sequential API requests, from 1 to 8 |
| Output width / height | 0 / 0 | Set both to positive integers for manual dimensions |
| Aspect ratio | Auto | Used when manual dimensions are not specified |
| Resolution level | None | Optional 1K / 2K / 4K setting |
| Limit output aspect ratio | Off | Adjusts extreme requested dimensions to the 1:3–3:1 range |
| Reference-image MP limit | 2.5 | Limits each uploaded reference independently; 0 disables the limit |
| Quality | medium | `auto`, `low`, `medium`, or `high` |
| Chunked transfer | On | Reads ordinary JSON responses incrementally; not SSE |
| Timeout | 100 seconds | Read inactivity timeout, not a strict total execution deadline |

Manual width and height can be converted to input sockets and connected to resolution nodes.

Both must be positive, or both must be zero. Manual dimensions override the aspect-ratio and resolution-level controls.

### Reference images and outputs

- Up to 8 reference images in total, including frames inside connected image batches.
- Oversized references are resized proportionally with Lanczos before upload.
- Outputs: an `IMAGE` batch and a text execution report.
- All recognized, distinct image sources are processed; there is no configurable output-count cap.
- The provider may return multiple images even when the request contains `n=1`.
- Different returned image sizes are centered on a shared white canvas to form a valid ComfyUI batch.
- Returned images are decoded to RGB; transparency is composited onto white.
- Each returned image has a 40 MB safety limit.
- Generation requests are not automatically retried to reduce the risk of duplicate requests or charges.

### Provider examples

#### GPT Image

```text
Protocol: OpenAI Images
Model: gpt-image-2
Output width: 1536
Output height: 1024
Quality: medium
Custom endpoint: empty
```

Without references, the node uses the generation endpoint. With references, it uses the multipart editing endpoint.

For the exact model name `gpt-image-2`, the node applies the configured provider constraints: 16-pixel alignment, maximum edge length, total pixel count, and aspect-ratio validation.

#### Gemini / Nano Banana

Select either Gemini native or Nano Banana OpenAI-compatible according to the provider documentation.

Gemini maps manual dimensions to supported aspect-ratio and resolution-level settings; exact pixel dimensions are not guaranteed.

Nano Banana compatibility mode uses `quality=1k/2k/4k` for its resolution level, not the GPT Image quality values.

#### Seedream

For the JSON editing route described in this branch's usage:

```text
Protocol: Generic JSON
Model: seedream-5.0
Custom endpoint: /images/edits
```

Use the provider's working base URL. If necessary, enter the complete editing URL as the custom endpoint.

Model names beginning with `seedream` omit the `quality` field outside the Nano Banana compatibility branch.

#### Qwen image through Chat Completions

```text
Protocol: Chat Completions image
Model: qwen-image-3.0
Custom endpoint: empty
```

This mode sends `messages`, `stream=false`, `temperature=0.7`, and `max_tokens=128`.

It does not send output size or quality parameters. Reference images use the standard multimodal message structure, but editing support depends on the provider.

### API limitations and security

- No SSE image streaming or intermediate generation previews.
- No asynchronous task polling.
- No dedicated edit-mask upload input.
- Generic JSON and multipart modes do not automatically adapt every provider-specific schema.
- Multipart image fields are fixed to `image` for one reference and `image[]` for multiple references.
- Cancellation checks occur between processing steps and received chunks; a blocked network read may wait until data arrives or the timeout expires.
- API keys entered in the node may be saved in workflows. Remove them before sharing.

Alternatively, leave the node key empty and set:

```text
UNIVERSAL_IMAGE_API_KEY
```

Only use API endpoints you trust: prompts and reference images are sent to that service.

## Multi-image Resize

**Node:** `多图统一缩放 / Multi-image Resize`  
**Category:** `孤海工具箱/图像`  
**Internal ID:** `GoohaiMultiImageResize`

A local image preprocessing node with Chinese/English interface labels and execution reports.

### Inputs and outputs

Inputs:

- Image 1: required.
- Images 2–8: optional.
- Image 1 mask: optional.

Each image input can contain a batch. Different input slots may have different resolutions.

Outputs:

- 8 corresponding image outputs.
- Image 1 mask.
- Image 1 padding mask.
- Image 1 final width and height.
- Execution report.

Unconnected optional images produce `None`; do not connect these empty outputs to downstream nodes requiring a valid image.

### Resize modes

| Mode | Behavior |
|---|---|
| Megapixel limit | Image 1 is reduced only above the MP limit. Other images are reduced only above `min(MP limit, image 1 final area × multiplier)`. No upscaling. |
| Force megapixel size | Every connected image is resized toward the target pixel area. Smaller images may be enlarged. |
| Image 1 manual size or edge | Image 1 uses custom dimensions or a target long/short edge. Other images shrink only when their area exceeds image 1's final area. |
| Pad to aspect ratio | Images are padded to a shared ratio, then reduced if the padded canvas exceeds the MP limit. |

The MP setting uses:

```text
1 MP = 1,000,000 pixels
```

The manual size/edge mode does not use the MP limit.

### Resolution and padding controls

- Default MP limit: **2.5**.
- Other-image area multiplier: **1.0**, available in megapixel-limit mode.
- Target edge length: **2048**, used for long-edge and short-edge resizing.
- Custom width/height mode supports stretch, center crop, and padding.
- The “Original” ratio preset uses **image 1's original aspect ratio as the common target**.
- Manual ratio is read only when the ratio preset is “Manual”.
- Seven preset fill colors plus a custom RGB option; default: **gray**.
- Custom colors accept `#000000`, `0,0,0`, or `rgb(0,0,0)`.
- Default interpolation: **Lanczos**. Bicubic, Bilinear, Nearest, and Box are also available.
- Image resizing uses the selected interpolation; masks use bilinear interpolation.
- Divisor defaults to **0**, meaning disabled.
- Divisor rounding can slightly change aspect ratios. Under-limit images in limit modes keep their dimensions unchanged.

### Mask behavior

For the current implementation:

- No input mask: image 1 mask output is white.
- With an input mask: it follows image 1's resize/crop/padding operations.
- The padding mask is white in newly padded regions and black in the original-image region.
- Without padding, the padding mask is black.
- A single input mask is broadcast across image 1's batch.

Outputs are RGB. Input alpha channels are composited onto the selected fill color.

### Suggested workflow

```text
Load images
    → Multi-image Resize
    → API Image Generation/Editing
    → Save Image
```

You can also connect image 1's width and height outputs to the API node's manual dimension inputs.

When preprocessing externally, account for the API node's own reference-image MP limit, which may resize uploaded references again.

## Updating and Troubleshooting

- Restart ComfyUI after changing Python files.
- Use `Ctrl+F5` to refresh frontend assets after changing JavaScript files.
- Dynamic resize controls require `web/multi_image_resize.js` and the plugin's `WEB_DIRECTORY` registration.
- If newly added controls restore incorrectly in an old workflow, recreate the affected node and reconnect it.
- Avoid loading an old standalone copy of the image API node alongside this plugin, as duplicate internal node IDs may conflict.
- For API failures, check the execution report, actual request path, model name, protocol, and provider documentation.

## Credits

- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
