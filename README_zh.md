# ComfyUI-llama-cpp

在 ComfyUI 中运行本地 LLM/VLM，并调用 OpenAI 兼容 API。

**[English](./README.md)**
<img width="1980" height="1888" alt="image" src="https://github.com/user-attachments/assets/316cacab-98f9-4367-80d6-67640c2106bf" />

## 功能

### 本地模型
- 基于 llama.cpp 加载 GGUF 模型，视觉推理需匹配的 mmproj。
- 提供 10 个图片输入接口，每个支持单图或图片批次。
- 支持逐组、多图合并及视频帧采样推理。
- 支持图片像素限制、模型自动卸载和文本对话状态。
- 提供边界框、遮罩、JSON 处理及实验性文本编码器。

### API 模型
- 提供 API 加载器与通用链路两个节点。
- 支持模型列表刷新、多图合并、逐组推理及回复列表。
- 支持 Tavily 联网搜索和 SSE 流式响应。
- 流式正文显示在后端控制台，节点结束后输出完整结果。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
```

CUDA 13 用户可改用 `requirements_cu131.txt`。请使用 ComfyUI 对应的 Python 环境安装依赖，完成后重启并刷新浏览器。

## 配置

本地模型放入 `ComfyUI/models/LLM`，视觉模型需选择对应 mmproj。

API 加载器填写基础地址、密钥和模型名称；地址不要包含 `/chat/completions`。

联网搜索启用前，在插件目录创建 `search_config.json`：

```json
{
  "tavily_api_key": "你的搜索密钥",
  "max_results": 5
}
```

## 注意

同一图片批次视为一组，10 个接口不代表最多 10 张。API 多图能力取决于服务及模型。搜索会向 Tavily 发送关键词，图片与提示词会发送给模型服务。分享工作流前清除密钥，勿上传搜索配置。

## 致谢

[llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) · [原项目](https://github.com/kijai/ComfyUI-llama-cpp) · [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
