
# ComfyUI-llama-cpp本地与 API 大模型工具

为 ComfyUI 提供 llama.cpp 本地推理与 OpenAI 兼容 API 调用，支持文本、多图分析和提示词增强。

## 主要功能

### 本地推理
- 加载 GGUF 模型及 mmproj 视觉投影文件。
- 多图合并、逐组分析、视频帧采样。
- 上下文长度、采样参数和图片像素限制。
- 对话状态保存、清理及模型卸载策略。
- JSON 解析、边界框绘制、MASK 与 SEGS 转换。
- 提示词增强预设及实验性文本编码器。

模型支持取决于所安装的 llama-cpp-python 版本和对应 Handler。

### API 推理
- OpenAI 兼容接口及模型列表刷新。
- 纯文本、多图合并、逐组推理。
- SSE 流式响应：实时正文显示在后端控制台。
- Tavily 联网搜索，密钥从本地配置读取。

## 安装与配置

将插件放入 `ComfyUI/custom_nodes/`，安装所需依赖后重启。

本地模型与 mmproj 放入 `ComfyUI/models/LLM/`，选择匹配的视觉 Handler。

联网搜索需在插件目录创建 `search_config.json`：

```json
{"tavily_api_key": "你的密钥", "max_results": 5}


**[[📃English](./README.md)]**   

## 预览
<img width="1887" height="1631" alt="image" src="https://github.com/user-attachments/assets/3aff4452-4bfb-432c-bcb6-dbee51a2abf8" />


## 安装步骤

#### 安装节点:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
#CUDA 13用户请执行:
#python -m pip install -r ComfyUI-llama-cpp/requirements_cu131.txt
```

### 模型路径:
- 请将下载的 `.gguf` 模型放置在 `ComfyUI/models/LLM` 目录中.  

	> 在使用VLM模型进行图像推理之前, 请确保已经下载并选择了主模型对应的`mmproj`权重文件.

## 致谢
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
