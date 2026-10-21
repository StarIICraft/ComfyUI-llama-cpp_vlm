# ComfyUI-llama-cpp

在 ComfyUI 中基于 llama.cpp 原生运行 LLM / VLM 模型，并提供 API 图像生成编辑与多图统一缩放节点。

本分支增加：

- 本地 LLM / VLM 工作流支持最多10张图像的多图识别。
- 改进的模型自动卸载。
- 输入图像百万像素限制。
- 支持最多8张参考图的通用 API 图像生成与编辑。
- 支持8路图像输入输出、图1遮罩处理、统一比例填充和尺寸控制的多图缩放节点。

**[[📃English](./README.md)]**

## 预览

<img width="1996" height="1624" alt="image" src="https://github.com/user-attachments/assets/caf29289-791c-433a-a78e-b5fd9e85ef7d" />
<img width="1880" height="1879" alt="api图像编辑-生成与多图统一缩放" src="https://github.com/user-attachments/assets/76af5597-4ad0-4b09-802f-f505be0bc98d" />

## 安装步骤

### 安装节点

请使用运行 ComfyUI 的 Python 环境执行：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
```

CUDA 13 用户使用对应依赖文件：

```bash
python -m pip install -r ComfyUI-llama-cpp/requirements_cu131.txt
```

安装后重启 ComfyUI，并刷新浏览器以加载前端扩展。

> 使用 ComfyUI 便携版时，请使用其自带 Python，而不是其他系统 Python 环境。

### 本地模型路径

请将 `.gguf` 模型放置在：

```text
ComfyUI/models/LLM
```

使用本地 VLM 进行图像识别前，还需要下载并选择与主模型匹配的 `mmproj` 权重。

API 图像节点不需要下载本地图像生成模型，但需要可用的 API 服务地址和密钥。

## 通用 API 图像生成与编辑

**节点名称：** `☁️API 图像生成与编辑`  
**节点分类：** `API大模型/图像生成编辑`  
**内部标识：** `UniversalImageAPIEditor`

通过兼容的远程服务进行文生图、参考图编辑或多图参考生成。

### 支持的接口格式

| 接口协议 | 默认端点 | 参考图上传方式 |
|---|---|---|
| OpenAI Images | 无参考图使用 `/v1/images/generations`；有参考图使用 `/v1/images/edits` | 编辑时使用 multipart 文件上传 |
| Gemini 原生 | `/v1beta/models/{model}:generateContent` | `inlineData` |
| Nano Banana OpenAI兼容 | `/v1/images/generations` | JSON `image` 数组，内容为图片 data URL |
| Chat Completions 图像 | `/v1/chat/completions` | 多模态 `messages` |
| 通用 JSON | `/v1/images/generations` | JSON `image` 数组，内容为图片 data URL |
| 通用 multipart | `/v1/images/edits` | multipart 文件上传，必须连接参考图 |

支持自定义端点路径或完整 URL。

> 接口协议表示请求格式，不代表所有模型、所有服务商都支持该格式。请根据服务商文档选择，不能仅通过更换模型名称自动适配所有接口。

### 主要设置

| 参数 | 默认值 | 作用 |
|---|---|---|
| 接口协议 | OpenAI Images | 选择请求格式 |
| 请求次数 | 1 | 串行执行1～8次请求 |
| 输出宽度 / 高度 | 0 / 0 | 两者均大于0时使用手动尺寸 |
| 图像比例 | 自动 | 未指定手动宽高时使用 |
| 分辨率档位 | 无 | 可选1K、2K、4K |
| 限制输出宽高比 | 关闭 | 将过宽或过高的请求尺寸调整到1:3～3:1范围 |
| 参考图百万像素上限 | 2.5 | 每张上传参考图独立限幅，0表示不限制 |
| 生成质量 | medium | 支持auto、low、medium、high |
| 分块传输 | 开启 | 分块读取普通JSON，不是SSE |
| 请求超时 | 100秒 | 读取响应的空闲超时，不是整个任务的严格总时限 |

输出宽度和高度可通过右键菜单转换为输入端口，连接其他分辨率节点。

宽高必须同时大于0，或同时为0。指定手动尺寸时，忽略界面中的图像比例与分辨率档位。

### 参考图与输出

- 最多上传8张参考图，输入端口内的图像批次也计入总数。
- 超过参考图像素上限时，上传前使用Lanczos等比例缩小。
- 输出为图像批次和执行报告文本。
- 处理所有可识别的不同图片来源，不提供返回数量截断选项。
- 即使请求发送`n=1`，服务端仍可能返回多张图片。
- 返回图片尺寸不一致时，居中填充到统一白色画布，组成ComfyUI图像批次。
- 输出统一为RGB，透明区域合成到白色背景。
- 单张返回图片有40MB安全限制。
- 生成请求不自动重试，减少重复提交和重复计费风险。

### 常用接口配置

#### GPT Image

```text
接口协议：OpenAI Images
模型名称：gpt-image-2
输出宽度：1536
输出高度：1024
生成质量：medium
自定义端点：留空
```

不连接参考图时使用文生图接口；连接参考图后使用multipart编辑接口。

对于精确模型名`gpt-image-2`，节点按已配置的服务商约束处理16倍数、最大边长、总像素数和宽高比。实际发送尺寸可在执行报告中查看。

#### Gemini / Nano Banana

根据服务商文档选择“Gemini 原生”或“Nano Banana OpenAI兼容”。

Gemini会将手动尺寸近似映射为支持的比例和分辨率档位，不保证精确输出指定像素尺寸。

Nano Banana兼容模式中的分辨率使用`quality=1k/2k/4k`，与GPT Image的质量档位不同。

#### Seedream

对于本分支使用过程中验证的JSON编辑接口：

```text
接口协议：通用 JSON
模型名称：seedream-5.0
自定义端点：/images/edits
```

基础地址使用服务商提供的有效地址。为避免路径拼接歧义，也可以在自定义端点直接填写完整编辑URL。

除Nano Banana兼容分支外，模型名以`seedream`开头时不会发送`quality`参数。

#### Qwen Chat Completions 图像接口

```text
接口协议：Chat Completions 图像
模型名称：qwen-image-3.0
自定义端点：留空
```

该模式发送`messages`、`stream=false`、`temperature=0.7`和`max_tokens=128`。

不发送输出尺寸或生成质量参数。参考图使用标准多模态消息格式，但该路由是否支持图像编辑，由服务商决定。

### 限制与安全说明

- 不支持SSE图像流或生成中的中间预览。
- 不支持提交异步任务后自动轮询。
- 不提供独立的编辑遮罩上传输入。
- 通用JSON和multipart不会自动适配所有服务商的专用参数结构。
- multipart图片字段固定为：单图`image`，多图`image[]`。
- 中断检查在处理步骤和接收数据块之间进行；网络读取阻塞时可能需要等待数据或超时。
- 节点内填写的密钥可能保存到工作流，分享前请清除。

也可以将节点密钥留空，使用环境变量：

```text
UNIVERSAL_IMAGE_API_KEY
```

请只使用可信任的接口地址：提示词和参考图会发送给对应服务。

## 多图统一缩放

**节点名称：** `多图统一缩放 / Multi-image Resize`  
**节点分类：** `孤海工具箱/图像`  
**内部标识：** `GoohaiMultiImageResize`

用于API调用前的本地图像预处理，支持中英文界面标签及执行报告。

### 输入与输出

输入：

- 图1：必需。
- 图2～图8：可选。
- 图1遮罩：可选。

每个图像输入都支持图像批次，不同输入端口可以使用不同分辨率。

输出：

- 图1～图8对应图像。
- 图1遮罩。
- 图1拓展遮罩。
- 图1最终宽度、高度。
- 执行报告。

未连接的可选图片输出为`None`，不要将其连接到必须接收有效图像的下游节点。

### 四种缩放模式

| 模式 | 处理方式 |
|---|---|
| 统一百万像素限制 | 图1超出百万像素上限才缩小；其他图超过`min(百万像素上限, 图1最终面积×乘数)`才缩小，不放大 |
| 所有图像强制缩放到百万像素 | 所有已连接图片都向目标像素面积缩放，小图允许放大 |
| 图1手动尺寸/边模式 | 图1使用手动宽高或目标长短边；其他图仅在面积超过图1最终面积时缩小 |
| 统一比例填充 | 将所有图像填充到共同目标比例，再按百万像素上限缩小 |

百万像素按以下方式计算：

```text
1 MP = 1,000,000像素
```

手动尺寸/边模式不使用百万像素上限。

### 尺寸与填充设置

- 百万像素上限默认 **2.5**。
- 其余图像根据图1面积乘数默认 **1.0**，仅统一百万像素限制模式使用。
- “将边缩放到”默认 **2048**，用于按长边或短边保持比例。
- 自定义宽高支持拉伸、居中裁剪、填充。
- 比例预设“原图”表示：**以图1原始比例作为所有图像的共同目标比例**。
- 只有选择“手动”比例时，手动比例输入才生效。
- 填充色提供七种预设及自定义选项，默认 **灰色**。
- 自定义颜色支持`#000000`、`0,0,0`、`rgb(0,0,0)`。
- 图像插值默认 **Lanczos**，另有双三次、双线性、邻近、Box。
- 所有图像缩放使用所选插值；遮罩固定使用双线性插值。
- 整除数默认 **0**，表示关闭。
- 整除及整数取整可能带来少量比例偏差；上限模式中未超限的图片保持尺寸不变。

### 遮罩行为

当前代码的行为：

- 未连接图1遮罩：图1遮罩输出为纯白。
- 已连接图1遮罩：遮罩跟随图1执行缩放、裁剪或填充。
- 图1拓展遮罩：新增填充区域为白色，原图区域为黑色。
- 没有发生填充时，拓展遮罩为纯黑。
- 单张遮罩可自动复用于图1的整个批次。

输出图像统一为RGB。输入带Alpha通道时，透明区域合成到所选填充色。

### 推荐工作流

```text
加载图像
    → 多图统一缩放
    → API图像生成与编辑
    → 保存图像
```

也可以将图1宽度、高度输出连接到API节点的手动尺寸输入。

如果已在外部缩放参考图，请同时检查API节点自身的参考图百万像素上限，避免上传前再次缩小。

## 更新与排错

- 修改Python文件后，需要重启ComfyUI。
- 修改JavaScript文件后，可使用`Ctrl+F5`刷新浏览器。
- 动态选项需要`web/multi_image_resize.js`，并在插件入口保留`WEB_DIRECTORY`注册。
- 旧工作流恢复新控件时若出现参数错位，请重新添加对应节点并接线。
- 不要同时加载同名旧版独立API图像插件，避免内部节点标识重复。
- API报错时，优先检查执行报告、实际请求路径、模型名称、协议和服务商文档。

## 致谢

- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
