import os
import io
import gc
import re
import json
import base64
import random
import torch
import inspect
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from .support.cqdm import cqdm
from .support.gguf_layers import get_layer_count
from .support.prompt_enhancer_preset import *

import folder_paths
import comfy.model_management as mm
import comfy.utils

import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
    NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler
)

# 基础 Chat Handlers 定义
chat_handlers = ["None", "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA", "llama3-Vision-Alpha", "MiniCPM-v2.6"]

try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
    chat_handlers += ["DeepSeek-OCR"]
    _MTMD = True
except Exception:
    _MTMD = False

try:
    from llama_cpp.llama_chat_format import Gemma3ChatHandler
    chat_handlers += ["Gemma3"]
except Exception:
    Gemma3ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
    chat_handlers += ["Gemma4"]
except Exception:
    Gemma4ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    chat_handlers += ["Qwen2.5-VL", "MinerU2.5-Pro"]
except Exception:
    Qwen25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
    chat_handlers += ["Qwen3-VL", "Qwen3-VL-Thinking"]
except Exception:
    Qwen3VLChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
    chat_handlers += ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3.8", "Qwen3.8-Thinking"]
except Exception:
    Qwen35ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import (GLM46VChatHandler, LFM2VLChatHandler, GLM41VChatHandler)
    chat_handlers += ["GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking", "LFM2-VL"]
except Exception:
    GLM46VChatHandler = None
    LFM2VLChatHandler = None
    GLM41VChatHandler = None

try:
    from llama_cpp.llama_chat_format import LFM25VLChatHandler
    chat_handlers += ["LFM2.5-VL"]
except Exception:
    LFM25VLChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
    chat_handlers += ["Granite-Docling"]
except Exception:
    GraniteDoclingChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import MiniCPMv45ChatHandler
    chat_handlers += ["MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"]
except Exception:
    MiniCPMv45ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import MiniCPMV46ChatHandler
    chat_handlers += ["MiniCPM-v4.6", "MiniCPM-v4.6-Thinking"]
except Exception:
    MiniCPMV46ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import PaddleOCRChatHandler
    chat_handlers += ["PaddleOCR-VL-1.5"]
except Exception:
    PaddleOCRChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Qwen3ASRChatHandler
    chat_handlers += ["Qwen3-ASR"]
except Exception:
    Qwen3ASRChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Step3VLChatHandler
    chat_handlers += ["Step3-VL"]
except Exception:
    Step3VLChatHandler = None

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

UNLOAD_AUTO = "自动"
UNLOAD_KEEP = "保持加载"
UNLOAD_AFTER = "运行后自动卸载"
UNLOAD_BEFORE_AFTER = "运行前后自动卸载"

UNLOAD_MODES = [
    UNLOAD_AUTO,
    UNLOAD_KEEP,
    UNLOAD_AFTER,
    UNLOAD_BEFORE_AFTER,
]

DEFAULT_MAX_MEGAPIXELS = 0.5

class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None
    current_embedding_mode = False
    messages = {}
    sys_prompts = {}
    
    @classmethod
    def ensure_embedding_mode(cls, enable_embeddings: bool):
        """
        根据底层 LlamaContext 源码精准适配的毫秒级热切换
        """
        if not cls.llm or not hasattr(cls.llm, "_ctx") or cls.llm._ctx is None:
            return
        
        # 如果当前模式已经符合要求，无需切换
        if getattr(cls, "current_embedding_mode", None) == enable_embeddings:
            return
        
        print(f"[llama-cpp_vlm] 正在无缝热切换上下文模式 (embeddings={enable_embeddings})...")
        
        import llama_cpp
        ctx_class = type(cls.llm._ctx)
        
        # 1. 获取原生的 llama_context_params 结构体
        if hasattr(cls.llm._ctx, "params") and cls.llm._ctx.params is not None:
            params = cls.llm._ctx.params
        else:
            params = llama_cpp.llama_context_default_params()
            
        # 2. 修改结构体中的 n_ctx 与 embeddings 状态
        if hasattr(params, "n_ctx"):
            params.n_ctx = cls.current_config["n_ctx"]
            
        if hasattr(params, "embeddings"):
            params.embeddings = enable_embeddings
            
        # 3. 关闭上一个旧 Context (保留 4GB _model 权重)
        try:
            cls.llm._ctx.close()
        except Exception:
            pass
            
        # 4. 精准依据类定义进行实例化 (必须使用关键字参数传参)
        cls.llm._ctx = ctx_class(
            model=cls.llm._model,
            params=params,
            verbose=False
        )
        cls.current_embedding_mode = enable_embeddings

    @classmethod
    def clean_state(cls, id=-1):
        if id == -1:
            cls.messages.clear()
            cls.sys_prompts.clear()
        else:
            cls.messages.pop(f"{id}", None)
            cls.sys_prompts.pop(f"{id}", None)
        
    @classmethod
    def clean(cls, all=False):
        """
        释放当前 llama.cpp 模型及视觉 Handler。

        all=False：
            保留 Python 中保存的文本对话记录。

        all=True：
            同时清除文本对话记录。
        """
        llm = cls.llm
        handler = cls.chat_handler

        # 先取消全局引用，避免后续误用已经关闭的对象。
        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        cls.current_embedding_mode = False

        try:
            if llm is not None:
                close = getattr(llm, "close", None)

                if callable(close):
                    close()
        except Exception as error:
            print(
                "[llama-cpp_vlm] "
                f"关闭模型时出现提示：{error}"
            )

        # 先关闭视觉 Handler，避免 Handler 仍然引用底层 Context。
        if handler is not None:
            try:
                close = getattr(
                    handler,
                    "close",
                    None,
                )

                if callable(close):
                    close()
                else:
                    exit_stack = getattr(
                        handler,
                        "_exit_stack",
                        None,
                    )

                    if exit_stack is not None:
                        exit_stack.close()

            except Exception as error:
                print(
                    "[llama-cpp_vlm] "
                    f"关闭视觉 Handler 时出现提示：{error}"
                )

        # 再关闭 llama.cpp 模型。
        if llm is not None:
            try:
                close = getattr(
                    llm,
                    "close",
                    None,
                )

                if callable(close):
                    close()

            except Exception as error:
                print(
                    "[llama-cpp_vlm] "
                    f"关闭模型时出现提示：{error}"
                )

        if all:
            cls.clean_state()

        del llm
        del handler

        gc.collect()

        try:
            mm.soft_empty_cache()
        except Exception:
            pass

    @staticmethod
    def release_cuda_cache():
        """
        与参考运行时一致：
        释放 PyTorch 未使用的 CUDA 缓存。

        注意：这不能代替 llama.cpp 模型本身的 close()。
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

                ipc_collect = getattr(
                    torch.cuda,
                    "ipc_collect",
                    None,
                )

                if callable(ipc_collect):
                    ipc_collect()
        except Exception:
            pass

    @staticmethod
    def memory_snapshot():
        """
        返回 ComfyUI 当前计算设备的：
            (可用内存字节数, 总内存字节数)

        GPU 模式通常对应显存；
        CPU 模式可能对应系统内存。
        """
        try:
            device = mm.get_torch_device()

            free = int(
                mm.get_free_memory(device)
            )

            total = int(
                mm.get_total_memory(device)
            )

            return free, total

        except Exception:
            return 0, 0

    @staticmethod
    def estimated_model_memory(config):
        """
        按参考运行时：
        (主模型文件大小 + mmproj 文件大小) × 1.25。

        这只是粗略估算，不包含精确的 KV Cache 和运行时开销。
        """
        total = 0

        for key in ("model", "mmproj"):
            name = config.get(key)

            if not name or name == "None":
                continue

            path = os.path.join(
                folder_paths.models_dir,
                "LLM",
                name,
            )

            try:
                total += os.path.getsize(path)
            except OSError:
                continue

        return int(total * 1.25)

    @staticmethod
    def unload_other_comfy_models():
        """
        释放 ComfyUI 管理的其他模型。

        不调用 unload_all_models()，
        避免触发本文件已有的全局卸载 Hook，
        导致当前 llama.cpp 模型也被一起清除。
        """
        loaded_models = getattr(
            mm,
            "current_loaded_models",
            None,
        )

        if loaded_models is not None and not loaded_models:
            return

        try:
            devices = mm.get_all_torch_devices()
        except Exception:
            try:
                devices = [mm.get_torch_device()]
            except Exception:
                return

        for device in devices:
            try:
                mm.free_memory(
                    1e30,
                    device,
                )
            except Exception as error:
                print(
                    "[llama-cpp_vlm] "
                    f"释放 ComfyUI 模型时出现提示：{error}"
                )

    @classmethod
    def prepare_unload_mode(cls, unload_mode, config):
        """
        对齐参考运行时的运行前策略：

        1. 释放未使用的 CUDA 缓存。
        2. 当前配置已经加载时直接复用。
        3. 否则满足下面任一条件时，尝试释放其他 ComfyUI 模型：
           - 运行前后自动卸载；
           - 预计所需内存大于当前可用内存。

        注意：
        即使是“保持加载”，参考运行时也会在加载前内存不足时
        尝试释放其他 ComfyUI 模型。
        """
        cls.release_cuda_cache()

        if (
            cls.llm is not None
            and cls.current_config == config
        ):
            return

        free, _ = cls.memory_snapshot()

        required = cls.estimated_model_memory(
            config
        )

        should_release = (
            unload_mode == UNLOAD_BEFORE_AFTER
        )

        should_release = should_release or (
            required > 0
            and free > 0
            and free < required
        )

        if should_release:
            print(
                "[llama-cpp_vlm] "
                "加载前释放其他 ComfyUI 模型..."
            )

            cls.unload_other_comfy_models()
            cls.release_cuda_cache()

    @classmethod
    def clear_inference_cache(cls):
        """
        清除本次推理的上下文缓存，不删除 Python 文本对话记录。
        """
        llm = cls.llm

        if llm is None:
            return

        try:
            if hasattr(llm, "n_tokens"):
                llm.n_tokens = 0
        except Exception:
            pass

        ctx = getattr(llm, "_ctx", None)

        if ctx is not None:
            for name, args in (
                ("kv_cache_clear", ()),
                ("memory_clear", (True,)),
            ):
                method = getattr(ctx, name, None)

                if not callable(method):
                    continue

                try:
                    method(*args)
                    break
                except Exception:
                    continue

        # 保留你原来对 Qwen 混合架构缓存的额外处理。
        hybrid_cache = getattr(
            llm,
            "_hybrid_cache_mgr",
            None,
        )

        if (
            getattr(llm, "is_hybrid", False)
            and hybrid_cache is not None
        ):
            try:
                hybrid_cache.clear()
            except Exception:
                pass

    @classmethod
    def finish_unload_mode(cls, unload_mode):
        """
        对齐参考运行时的运行后策略。

        自动：
            可用内存 < 总内存 50% 时卸载。

        保持加载：
            只清理上下文缓存。

        运行后自动卸载 / 运行前后自动卸载：
            卸载当前 llama.cpp 模型。
        """
        cls.clear_inference_cache()

        should_unload = unload_mode in {
            UNLOAD_AFTER,
            UNLOAD_BEFORE_AFTER,
        }

        if unload_mode == UNLOAD_AUTO:
            free, total = cls.memory_snapshot()

            should_unload = bool(
                total > 0
                and free * 2 < total
            )

            if total > 0:
                print(
                    "[llama-cpp_vlm] "
                    "自动卸载判断："
                    f"可用 {free / (1024 ** 3):.2f} GiB / "
                    f"总计 {total / (1024 ** 3):.2f} GiB，"
                    f"卸载={should_unload}"
                )

        if should_unload:
            print(
                "[llama-cpp_vlm] "
                f"运行后卸载模型，策略：{unload_mode}"
            )

            cls.clean(all=False)
    
    @classmethod
    def load_model(cls, config):
        def get_chat_handler(chat_handler):
            match chat_handler:
                case "Qwen3.5"|"Qwen3.5-Thinking"|"Qwen3.6"|"Qwen3.6-Thinking"|"Qwen3.8"|"Qwen3.8-Thinking":
                    return Qwen35ChatHandler
                case "Qwen3-VL"|"Qwen3-VL-Thinking":
                    return Qwen3VLChatHandler
                case "Qwen3-ASR":
                    return Qwen3ASRChatHandler
                case "Qwen2.5-VL"|"MinerU2.5-Pro":
                    return Qwen25VLChatHandler
                case "LLaVA-1.5":
                    return Llava15ChatHandler
                case "LLaVA-1.6":
                    return Llava16ChatHandler
                case "Moondream2":
                    return MoondreamChatHandler
                case "nanoLLaVA":
                    return NanoLlavaChatHandler
                case "llama3-Vision-Alpha":
                    return Llama3VisionAlphaChatHandler
                case "MiniCPM-v2.6":
                    return MiniCPMv26ChatHandler
                case "MiniCPM-v4.5"|"MiniCPM-v4.5-Thinking":
                    return MiniCPMv45ChatHandler
                case "MiniCPM-v4.6"|"MiniCPM-v4.6-Thinking":
                    return MiniCPMV46ChatHandler
                case "Gemma3":
                    return Gemma3ChatHandler
                case "Gemma4":
                    return Gemma4ChatHandler
                case "GLM-4.6V"|"GLM-4.6V-Thinking":
                    return GLM46VChatHandler
                case "GLM-4.1V-Thinking":
                    return GLM41VChatHandler
                case "LFM2-VL":
                    return LFM2VLChatHandler
                case "LFM2.5-VL":
                    return LFM25VLChatHandler
                case "Granite-Docling":
                    return GraniteDoclingChatHandler
                case "DeepSeek-OCR":
                    return MTMDChatHandler
                case "PaddleOCR-VL-1.5":
                    return PaddleOCRChatHandler
                case "Step3-VL":
                    return Step3VLChatHandler
                case "None":
                    return None
                case _:
                    raise ValueError(f'Unknown model type: "{chat_handler}"')
        
        # Loader 节点可能先于 Instruct 节点加载模型。
        # 因此这里也需要执行“预计内存不足则释放其他模型”的逻辑。
        cls.prepare_unload_mode(
            UNLOAD_KEEP,
            config,
        )

        previous_config = cls.current_config

        # 卸载后重新加载相同配置时，保留已保存的文本对话。
        # 真正从另一套已加载配置切换时，则清除旧对话。
        clear_history = (
            previous_config is not None
            and previous_config != config
        )

        cls.clean(all=clear_history)

        # 只有模型成功加载后才正式写入 current_config。
        cls.current_config = None
        model = config["model"]
        mmproj = config["mmproj"]
        chat_handler = config["chat_handler"]
        n_ctx = config["n_ctx"]
        vram_limit = config["vram_limit"]
        image_max_tokens = config["image_max_tokens"]
        image_min_tokens = config["image_min_tokens"]
        load_mtp = config["load_mtp"]
        n_gpu_layers = -1
        
        model_path = os.path.join(folder_paths.models_dir, 'LLM', model)
        handler_cls = get_chat_handler(chat_handler)
        
        if vram_limit != -1:
            gguf_layers = get_layer_count(model_path) or 32
            gguf_size = os.path.getsize(model_path) * 1.55 / (1024 ** 3)
            gguf_layer_size = gguf_size / gguf_layers
        
        if mmproj and mmproj != "None":
            mmproj_path = os.path.join(folder_paths.models_dir, 'LLM', mmproj)
            if chat_handler == "None":
                raise ValueError('"chat_handler" cannot be None when mmproj is used!')
            
            if handler_cls is None:
                raise RuntimeError(f"Chat handler '{chat_handler}' is not available or failed to import.")

            if vram_limit != -1:
                mmproj_size = os.path.getsize(mmproj_path) * 1.55 / (1024 ** 3)
                n_gpu_layers = max(1, int((vram_limit - mmproj_size) / gguf_layer_size))
            
            print(f"[llama-cpp_vlm] Loading clip: {mmproj}")
            
            think_mode = "Thinking" in chat_handler
            kwargs = {"clip_model_path": mmproj_path, "verbose": False}
            if chat_handler in ["Qwen3-VL", "Qwen3-VL-Thinking"]:
                kwargs["force_reasoning"] = think_mode
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
            elif any(x in chat_handler for x in ["GLM-4.6V", "MiniCPM-v4.5", "MiniCPM-v4.6", "Qwen3.5", "Qwen3.6", "Qwen3.8"]):
                kwargs["enable_thinking"] = think_mode

            if _MTMD:
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens

            try:
                cls.chat_handler = handler_cls(**kwargs)
            except Exception as e:
                raise RuntimeError(f"{e}\nPlease update llama-cpp-python from 'https://github.com/JamePeng/llama-cpp-python/releases'")

        else:
            if vram_limit != -1:
                n_gpu_layers = max(1, int(vram_limit / gguf_layer_size))
            if handler_cls is not None:
                cls.chat_handler = handler_cls(verbose=False)
            else:
                cls.chat_handler = None
        
        print(f"[llama-cpp_vlm] Loading model: {model}")
        print(f"[llama-cpp_vlm] n_gpu_layers = {n_gpu_layers}")
        kwargs = {
            "model_path": model_path,
            "chat_handler": cls.chat_handler,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "verbose": False,
        }
        if "load_mtp" in inspect.signature(Llama.__init__).parameters:
            kwargs["load_mtp"] = load_mtp
        else:
            if load_mtp:
                raise RuntimeError('"load_mtp" is unavailable! Please upgrade your llama-cpp-python.')
            
        try:
            new_llm = Llama(**kwargs)

        except Exception:
            # 加载失败时清理可能已经创建的 Handler。
            cls.clean(all=False)
            raise

        cls.llm = new_llm
        cls.current_config = config.copy()
        cls.current_embedding_mode = False

any_type = AnyType("*")

if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        LLAMA_CPP_STORAGE.clean(all=True)
        result = mm.unload_all_models_backup(*args, **kwargs)
        return result
    mm.unload_all_models = patched_unload_all_models
    print("[llama-cpp_vlm] Model cleanup hook applied!")

llm_extensions = ['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.gguf']
folder_paths.folder_names_and_paths["LLM"] = ([os.path.join(folder_paths.models_dir, "LLM")], llm_extensions)
preset_prompts = {
    "Empty - Nothing": "",
    "Normal - Describe": "Describe this @.",
    "Prompt Style - Tags": "Your task is to generate a clean list of comma-separated tags for a text-to-@ AI, based *only* on the visual information in the @. Limit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags.",
    "Prompt Style - Simple": "Analyze the @ and generate a simple, single-sentence text-to-@ prompt. Describe the main subject and the setting concisely.",
    "Prompt Style - Detailed": "Generate a detailed, artistic text-to-@ prompt based on the @. Combine the subject, their actions, the environment, lighting, and overall mood into a single, cohesive paragraph of about 2-3 sentences. Focus on key visual details.",
    "Prompt Style - Extreme Detailed": "Generate an extremely detailed and descriptive text-to-@ prompt from the @. Create a rich paragraph that elaborates on the subject's appearance, textures of clothing, specific background elements, the quality and color of light, shadows, and the overall atmosphere. Aim for a highly descriptive and immersive prompt.",
    "Prompt Style - Cinematic": "Act as a master prompt engineer. Create a highly detailed and evocative prompt for an @ generation AI. Describe the subject, their pose, the environment, the lighting, the mood, and the artistic style (e.g., photorealistic, cinematic, painterly). Weave all elements into a single, natural language paragraph, focusing on visual impact.",
    "Creative - Detailed Analysis": "Describe this @ in detail, breaking down the subject, attire, accessories, background, and composition into separate sections.",
    "Creative - Summarize Video": "Summarize the key events and narrative points in this video.",
    "Creative - Short Story": "Write a short, imaginative story inspired by this @ or video.",
    "Creative - Refine & Expand Prompt": "Refine and enhance the following user prompt for creative text-to-@ generation. Keep the meaning and keywords, make it more expressive and visually rich. Output **only the improved prompt text itself**, without any reasoning steps, thinking process, or additional commentary.",
    "Vision - *Bounding Box": 'Locate every instance that belongs to the following categories: "#". Report bbox coordinates in {"bbox_2d": [x1, y1, x2, y2], "label": "string"} JSON format as a List.'
}
preset_tags = list(preset_prompts.keys())

def image2base64(image):
    """
    将 NumPy 图片编码为 JPEG Base64。
    """
    img = Image.fromarray(image)

    if img.mode != "RGB":
        img = img.convert("RGB")

    buffered = io.BytesIO()

    img.save(
        buffered,
        format="JPEG",
        quality=85,
        optimize=False,
    )

    return base64.b64encode(
        buffered.getvalue()
    ).decode("ascii")

def parse_json(json_str):
    json_output = json_str.strip()
    if json_output.startswith("```json"):
        json_output = json_output[7:]
    if json_output.startswith("```"):
        json_output = json_output[3:]
    if json_output.endswith("```"):
        json_output = json_output[:-3]
    json_output = json_output.strip()
    try:
        parsed = json.loads(json_output)
    except Exception as e:
        raise ValueError(f"Unable to load JSON data!\n{e}")
    return parsed

def scale_image(
    image: torch.Tensor,
    max_megapixels: float = DEFAULT_MAX_MEGAPIXELS,
):
    """
    按百万像素上限处理单张图片。

    支持：
        [H, W, C]
        [1, H, W, C]

    例如：
        0.5 = 500,000 像素
        1.0 = 1,000,000 像素

    未超限不缩放，超限后使用 Lanczos 等比例缩小。
    不放大图片。
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            "图像必须为 torch.Tensor。"
        )

    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(
                "scale_image() 只能处理单张图像，"
                f"收到：{tuple(image.shape)}"
            )

        image = image[0]

    if image.ndim != 3:
        raise ValueError(
            "图像需要为 [H, W, C]，"
            f"收到：{tuple(image.shape)}"
        )

    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"不支持的图像通道数：{image.shape[-1]}"
        )

    max_megapixels = float(max_megapixels)

    if (
        not math.isfinite(max_megapixels)
        or max_megapixels <= 0
    ):
        raise ValueError(
            "百万像素限制必须是大于 0 的有限数值。"
        )

    array = (
        image.detach()
        .cpu()
        .float()
        .clamp(0.0, 1.0)
        .numpy()
    )

    # 只保留 RGB，避免 RGBA 直接转 JPEG 时出错。
    array = (
        array[..., :3] * 255.0 + 0.5
    ).astype(np.uint8)

    pil = Image.fromarray(
        array,
        mode="RGB",
    )

    width, height = pil.size

    if width <= 0 or height <= 0:
        raise ValueError(
            "图像宽高必须大于 0。"
        )

    max_pixels = max(
        1,
        int(max_megapixels * 1_000_000),
    )

    current_pixels = width * height

    # 未超过限制时，不调用 resize。
    if current_pixels <= max_pixels:
        return np.asarray(pil)

    scale = math.sqrt(
        max_pixels / current_pixels
    )

    new_width = max(
        1,
        min(
            width,
            round(width * scale),
        ),
    )

    new_height = max(
        1,
        min(
            height,
            round(height * scale),
        ),
    )

    # 由于 round() 可能导致少量超出，再做一次修正。
    while new_width * new_height > max_pixels:
        if new_width >= new_height:
            new_width = max(
                1,
                new_width - 1,
            )
        else:
            new_height = max(
                1,
                new_height - 1,
            )

    if (
        new_width != width
        or new_height != height
    ):
        pil = pil.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

    return np.asarray(pil)

def qwen3bbox(image, json_data):
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    bboxes = []
    if isinstance(json_data, dict):
        json_data = [json_data]
    for item in json_data:
        if not isinstance(item, dict) or "bbox_2d" not in item:
            continue
        x0, y0, x1, y1 = item["bbox_2d"]
        size = 1000.0
        x0 = x0 / size * img.width
        y0 = y0 / size * img.height
        x1 = x1 / size * img.width
        y1 = y1 / size * img.height
        bboxes.append((x0, y0, x1, y1))
    return bboxes

def draw_bbox(image, json_data, mode):
    label_colors = {}
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    
    if isinstance(json_data, dict):
        json_data = [json_data]

    for item in json_data:
        if not isinstance(item, dict):
            continue
        label = item.get("label", item.get("text_content", "bbox"))
        if "bbox_2d" not in item:
            continue
            
        x0, y0, x1, y1 = item["bbox_2d"]
        if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
            size = 1000.0
            x0 = x0 / size * img.width
            y0 = y0 / size * img.height
            x1 = x1 / size * img.width
            y1 = y1 / size * img.height
        bbox = (x0, y0, x1, y1)
        
        if label not in label_colors:
            label_colors[label] = tuple(random.randint(80, 180) for _ in range(3))
        color = label_colors[label]
        draw.rectangle(bbox, outline=color, width=4)
        text_y = max(0, y0 - 10)
        text_size = draw.textbbox((x0, text_y), str(label))
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), str(label), fill=(255,255,255))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        all_llms = folder_paths.get_filename_list("LLM")
        model_list = [f for f in all_llms if "mmproj" not in f.lower()]
        mmproj_list = ["None"] + [f for f in all_llms if "mmproj" in f.lower()]
            
        return {"required": {
            "model": (model_list,),
            "mmproj": (mmproj_list, {"default": "None"}),
            "chat_handler": (chat_handlers, {"default": "None"}),
            "n_ctx": ("INT", {
                "default": 8192,
                "min": 1024, "max": 327680, "step": 128,
                "tooltip": "Context length limit."
            }),
            "vram_limit": ("INT", {
                "default": -1,
                "min": -1, "max": 1024, "step": 1,
                "tooltip": "VRAM usage limit in GB (-1 = no limit)\nReference range; actual usage may slightly exceed."
            }),
            "image_min_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            "load_mtp": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vlm"

    def loadmodel(self, model, mmproj, chat_handler, n_ctx, vram_limit, image_min_tokens, image_max_tokens, load_mtp):
        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler": chat_handler,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "load_mtp": load_mtp
        }
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != custom_config:
            print("[llama-cpp_vlm] Loading model...")
            LLAMA_CPP_STORAGE.load_model(custom_config)
        return (custom_config,)

def collect_instruct_images(images=None, extra_inputs=None):
    """
    按输入接口顺序收集图片。

    支持：
        - 单张 IMAGE：[H, W, C]
        - IMAGE Batch：[B, H, W, C]
        - INPUT_IS_LIST=True 产生的列表包装
        - 上游节点产生的嵌套列表

    返回值中的每个元素保持原始 Tensor：
        [H, W, C] 或 [B, H, W, C]

    不在此处拆分 batch，以便后续根据 inference_mode
    决定 batch 是多图、视频帧，还是独立输入组。
    """
    collected = []
    extra_inputs = extra_inputs or {}

    def visit(value, input_name):
        if value is None:
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item, input_name)
            return

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"图像输入 {input_name} 收到不支持的类型："
                f"{type(value).__name__}。"
                "需要 ComfyUI IMAGE Tensor。"
            )

        if value.ndim not in (3, 4):
            raise ValueError(
                f"图像输入 {input_name} 的形状无效："
                f"{tuple(value.shape)}。"
                "需要 [H, W, C] 或 [B, H, W, C]。"
            )

        if value.ndim == 4 and value.shape[0] == 0:
            return

        if value.shape[-3] <= 0 or value.shape[-2] <= 0:
            raise ValueError(
                f"图像输入 {input_name} 的宽高无效："
                f"{tuple(value.shape)}"
            )

        collected.append(value)

    # 原有接口，必须保留以兼容旧工作流。
    visit(images, "images")

    # 新增接口，按照 images_02 到 images_10 的顺序收集。
    for index in range(2, 11):
        input_name = f"images_{index:02d}"
        visit(extra_inputs.get(input_name), input_name)

    return collected

def frame_to_data_url(
    frame,
    max_megapixels=DEFAULT_MAX_MEGAPIXELS,
):
    """
    按每张图片的百万像素上限缩小，再编码为 JPEG data URL。
    """
    frame_np = scale_image(
        image=frame,
        max_megapixels=max_megapixels,
    )

    encoded = image2base64(frame_np)

    return f"data:image/jpeg;base64,{encoded}"
    
class llama_cpp_instruct_adv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),

                "preset_prompt": (
                    preset_tags,
                    {
                        "default": preset_tags[1],
                    },
                ),

                "custom_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": (
                            "用户提示词\n\n"
                            "如果预设提示词中包含 '*', "
                            "这里的内容将用于填充占位符 "
                            "(例如目标检测中的对象名称。).\n"
                            "其他情况下，这里的内容会覆盖预设提示词。."
                        ),
                    },
                ),

                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "可选的系统提示词 "
                            "留空时不添加额外系统提示词."
                        ),
                    },
                ),

                "inference_mode": (
                    [
                        "one by one",
                        "images",
                        "video",
                    ],
                    {
                        "default": "one by one",
                        "tooltip": (
                            "逐个处理：每个连接的图像组单独生成一次结果。"
                            "同一个 IMAGE Batch 会被视为一个图像组 。\n"
                            "图像合并：将所有连接的图像合并到同一次生成中。\n"
                            "视频：将每个图像组视为一个独立视频，并按最大帧数均匀采样。"
                        ),
                    },
                ),

                "max_frames": (
                    "INT",
                    {
                        "default": 24,
                        "min": 2,
                        "max": 1024,
                        "step": 1,
                        "tooltip": (
                            "视频模式下每个视频组最多使用的帧数 "
                            "如果原始帧数超过该值，将进行均匀采样."
                        ),
                    },
                ),

                "max_megapixels": (
                    "FLOAT",
                    {
                        "default": DEFAULT_MAX_MEGAPIXELS,
                        "min": 0.01,
                        "max": 256.0,
                        "step": 0.01,
                        "tooltip": (
                            "每张图片允许的最大百万像素数。\n"
                            "默认 0.5 = 500,000 像素。\n"
                            "每张图片独立计算，未超限不缩放，"
                            "超限使用 Lanczos 等比例缩小。"
                        ),
                    },
                ),

                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                        "tooltip": (
                            "随机种子 "
                            "相同模型、提示词和参数下，尽量保持生成结果一致."
                        ),                        
                    },
                ),

                "unload_mode": (
                    UNLOAD_MODES,
                    {
                        "default": UNLOAD_AUTO,
                        "tooltip": (
                            "自动：运行后可用内存不足总量 50% 时卸载。\n"
                            "保持加载：运行后保留模型。\n"
                            "运行后自动卸载：结束后卸载当前模型。\n"
                            "运行前后自动卸载：需要加载模型时先释放"
                            "其他 ComfyUI 模型，结束后卸载当前模型。\n"
                            "所有模式在加载前预计内存不足时，"
                            "都可能释放其他 ComfyUI 模型。"
                        ),
                    },
                ),

                "save_states": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "保存文本对话状态到内存中."
                            "下一次使用相同状态 ID 时可以继续对话。\n"
                            "注意：为节省内存，保存状态时不会保留真实图片，"
                            "图片内容会被替换为占位图。"
                        ),
                    },
                ),
            },

            "hidden": {
                "unique_id": "UNIQUE_ID",
            },

            "optional": {
                "parameters": ("LLAMACPPARAMS",),

                # 保留原来的 images 接口，兼容旧工作流。
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "图像输入 1 "
                            "支持单张 IMAGE 或 IMAGE Batch。"
                        ),
                    },
                ),

                # 新增 images_02 到 images_10。
                **{
                    f"images_{index:02d}": (
                        "IMAGE",
                        {
                            "tooltip": (
                                f"Image input {index}. Supports a single "
                                "image or an IMAGE batch."
                            ),
                        },
                    )
                    for index in range(2, 11)
                },

                "queue_handler": (
                    any_type,
                    {
                        "tooltip": (
                            "用于控制多个 Instruct 节点的执行顺序。"
                            "关于指令节点."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("output", "output_list", "state_uid")
    OUTPUT_IS_LIST = (False, True, False)

    # 必须保留，否则多个 IMAGE 输入不会按照列表形式传入。
    INPUT_IS_LIST = True

    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"

    @staticmethod
    def _first(value, default=None):
        """
        INPUT_IS_LIST=True 时，普通参数通常会被包装为列表。
        图像参数不使用这个函数，避免破坏图像输入结构。
        """
        if isinstance(value, list):
            return value[0] if value else default
        return value if value is not None else default

    @staticmethod
    def _remove_thinking(text):
        # Normalize response content.
        text = str(text or "")
        text = text.removeprefix(": ").lstrip()

        # Construct tags without writing them literally.
        open_tag = "<" + "think" + ">"
        close_tag = "<" + "/think" + ">"

        pattern = (
            re.escape(open_tag)
            + r".*?(?:"
            + re.escape(close_tag)
            + r"|$)"
        )

        return re.sub(
            pattern,
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    def sanitize_messages(self, messages):
        """
        保存状态时替换图片 Base64，避免大量图片数据长期占用 RAM。

        注意：
        这意味着 save_states=True 只保存文本对话结构，
        不会保存可供下一轮重新视觉识别的真实图片。
        """
        clean_messages = json.loads(
            json.dumps(messages)
        )

        placeholder = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
            "AAAACXBIWXMAAAsTAAALEwEAmpwYAAAADElEQVQImWP4"
            "//8/AAX+Av5Y8msOAAAAAElFTkSuQmCC"
        )

        for message in clean_messages:
            if not isinstance(message, dict):
                continue

            content = message.get("content")

            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") != "image_url":
                    continue

                image_url = item.get("image_url")

                if isinstance(image_url, dict):
                    image_url["url"] = placeholder

        return clean_messages

    def _append_frame(self, content, frame):
        """
        每张图片独立执行百万像素限制。
        """
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": frame_to_data_url(
                        frame=frame,
                        max_megapixels=getattr(
                            self,
                            "_max_megapixels",
                            DEFAULT_MAX_MEGAPIXELS,
                        ),
                    ),
                },
            }
        )

    def _create_completion(
        self,
        messages,
        seed,
        parameters,
    ):
        """
        统一调用 llama.cpp，便于后续维护。
        """
        return LLAMA_CPP_STORAGE.llm.create_chat_completion(
            messages=messages,
            seed=seed,
            **parameters,
        )

    def process(
        self,
        llama_model,
        preset_prompt,
        custom_prompt,
        system_prompt,
        inference_mode,
        max_frames,
        seed,
        save_states,
        unique_id=-1,
        parameters=None,
        images=None,
        queue_handler=None,
        unload_mode=None,
        max_megapixels=DEFAULT_MAX_MEGAPIXELS,
        force_offload=None,
        **kwargs,
    ):
        """
        外层执行入口。

        负责：
            - 解包新增配置；
            - 运行前内存准备；
            - 调用原推理逻辑；
            - 无论成功、失败、中断，都执行运行后策略。

        force_offload：
            保留给旧 API 参数使用，新界面使用 unload_mode。
        """
        config = self._first(
            llama_model,
            None,
        )

        if not isinstance(config, dict):
            raise ValueError(
                "llama_model 配置无效，请连接 Model Loader。"
            )

        selected_unload_mode = self._first(
            unload_mode,
            None,
        )

        # 兼容旧 API 中的 force_offload。
        if selected_unload_mode is None:
            legacy_force = bool(
                self._first(
                    force_offload,
                    False,
                )
            )

            selected_unload_mode = (
                UNLOAD_AFTER
                if legacy_force
                else UNLOAD_KEEP
            )

        # 兼容旧布尔值。
        if isinstance(selected_unload_mode, bool):
            selected_unload_mode = (
                UNLOAD_AFTER
                if selected_unload_mode
                else UNLOAD_KEEP
            )

        if selected_unload_mode not in UNLOAD_MODES:
            raise ValueError(
                f"未知模型卸载模式：{selected_unload_mode}"
            )

        megapixels = float(
            self._first(
                max_megapixels,
                DEFAULT_MAX_MEGAPIXELS,
            )
        )

        if (
            not math.isfinite(megapixels)
            or megapixels <= 0
        ):
            raise ValueError(
                "百万像素限制必须是大于 0 的有限数值。"
            )

        self._max_megapixels = megapixels

        try:
            LLAMA_CPP_STORAGE.prepare_unload_mode(
                selected_unload_mode,
                config,
            )

            return self._process_impl(
                llama_model=llama_model,
                preset_prompt=preset_prompt,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                inference_mode=inference_mode,
                max_frames=max_frames,
                seed=seed,
                save_states=save_states,
                unique_id=unique_id,
                parameters=parameters,
                images=images,
                queue_handler=queue_handler,
                **kwargs,
            )

        finally:
            try:
                LLAMA_CPP_STORAGE.finish_unload_mode(
                    selected_unload_mode,
                )
            except Exception as error:
                print(
                    "[llama-cpp_vlm] "
                    f"运行后清理模型时出现提示：{error}"
                )

            gc.collect()

    def _process_impl(
        self,
        llama_model,
        preset_prompt,
        custom_prompt,
        system_prompt,
        inference_mode,
        max_frames,
        seed,
        save_states,
        unique_id=-1,
        parameters=None,
        images=None,
        queue_handler=None,
        **kwargs,
    ):
        # ---------------------------------------------------------
        # 1. 解包 INPUT_IS_LIST=True 产生的普通参数
        # ---------------------------------------------------------
        llama_model = self._first(llama_model)
        preset_prompt = self._first(
            preset_prompt,
            preset_tags[1],
        )
        custom_prompt = self._first(
            custom_prompt,
            "",
        )
        system_prompt = self._first(
            system_prompt,
            "",
        )
        inference_mode = self._first(
            inference_mode,
            "one by one",
        )
        max_frames = int(
            self._first(
                max_frames,
                24,
            )
        )
        seed = int(
            self._first(
                seed,
                0,
            )
        )
        save_states = bool(
            self._first(
                save_states,
                False,
            )
        )
        unique_id = self._first(
            unique_id,
            -1,
        )

        if isinstance(parameters, list):
            parameters = parameters[0] if parameters else None

        if parameters is None:
            parameters = {}

        parameters = dict(parameters)

        # state_uid 不是 llama.cpp 的推理参数。
        state_uid = parameters.pop(
            "state_uid",
            None,
        )

        if _MTMD:
            # MTMD Handler 不接受 present_penalty。
            parameters.pop(
                "present_penalty",
                None,
            )

        # 限制 seed 范围。
        seed = max(
            0,
            min(
                0xFFFFFFFFFFFFFFFF,
                seed,
            ),
        )

        # ---------------------------------------------------------
        # 2. 确保当前加载的模型与输入模型一致
        # ---------------------------------------------------------
        if (
            not LLAMA_CPP_STORAGE.llm
            or LLAMA_CPP_STORAGE.current_config != llama_model
        ):
            LLAMA_CPP_STORAGE.load_model(llama_model)

        # Instruct 模式需要关闭 embeddings。
        LLAMA_CPP_STORAGE.ensure_embedding_mode(False)

        # ---------------------------------------------------------
        # 3. 建立对话状态 ID
        # ---------------------------------------------------------
        if state_uid in (None, -1):
            uid = str(unique_id).rpartition(".")[-1]
        else:
            uid = str(state_uid)

        last_system_prompt = LLAMA_CPP_STORAGE.sys_prompts.get(
            uid,
            None,
        )

        video_input = inference_mode == "video"

        if video_input:
            effective_system_prompt = (
                "请将输入的图片序列当做视频，而不是静态帧序列。"
                + str(system_prompt or "")
            )
        else:
            effective_system_prompt = str(
                system_prompt or ""
            )

        if last_system_prompt != effective_system_prompt:
            messages = []

            LLAMA_CPP_STORAGE.clean_state(uid)

            LLAMA_CPP_STORAGE.sys_prompts[uid] = (
                effective_system_prompt
            )

            if effective_system_prompt.strip():
                messages.append(
                    {
                        "role": "system",
                        "content": effective_system_prompt,
                    }
                )

        elif save_states:
            print(
                f"[llama-cpp_vlm] Loading state and history id={uid}..."
            )

            saved_messages = LLAMA_CPP_STORAGE.messages.get(
                uid,
                [],
            )

            # 深拷贝，避免本轮修改全局状态。
            messages = json.loads(
                json.dumps(saved_messages)
            )

        else:
            messages = []

        # 保存本轮实际产生的 user/assistant 消息。
        history_messages = []

        output_list = []

        # ---------------------------------------------------------
        # 4. 构建基础文本提示词
        # ---------------------------------------------------------
        if (
            str(custom_prompt or "").strip()
            and "*" not in str(preset_prompt)
        ):
            prompt_text = str(custom_prompt)

        else:
            if preset_prompt not in preset_prompts:
                raise ValueError(
                    f'Unknown preset prompt: "{preset_prompt}"'
                )

            prompt_text = preset_prompts[preset_prompt]

            prompt_text = prompt_text.replace(
                "#",
                str(custom_prompt or "").strip(),
            )

            prompt_text = prompt_text.replace(
                "@",
                "video" if video_input else "image",
            )

        base_user_content = [
            {
                "type": "text",
                "text": prompt_text,
            }
        ]

        # ---------------------------------------------------------
        # 5. 收集多图输入
        # ---------------------------------------------------------
        #
        # images 是第一个接口。
        # images_02 到 images_10 位于 kwargs 中。
        #
        # collect_instruct_images() 会保留每个输入 Tensor 的边界：
        #
        # 单图：
        #     [H, W, C] -> 一个图像组
        #
        # Batch：
        #     [B, H, W, C] -> 一个包含 B 帧的图像组
        #
        raw_image_inputs = collect_instruct_images(
            images=images,
            extra_inputs=kwargs,
        )

        image_groups = []

        for image_input in raw_image_inputs:
            if image_input.ndim == 3:
                image_groups.append(
                    [image_input]
                )

            elif image_input.ndim == 4:
                frames = [
                    image_input[index]
                    for index in range(image_input.shape[0])
                ]

                if frames:
                    image_groups.append(frames)

        # ---------------------------------------------------------
        # 6. 图像推理
        # ---------------------------------------------------------
        if image_groups:
            current_config = (
                LLAMA_CPP_STORAGE.current_config
                or {}
            )

            current_mmproj = current_config.get(
                "mmproj"
            )

            if (
                LLAMA_CPP_STORAGE.chat_handler is None
                or current_mmproj in (
                    None,
                    "",
                    "None",
                )
            ):
                raise ValueError(
                    "检测到图像输入，但当前模型没有配置 mmproj。"
                )

            # =====================================================
            # one by one
            # =====================================================
            if inference_mode == "one by one":
                print(
                    "[llama-cpp_vlm] "
                    f"[One-by-One Mode] Processing "
                    f"{len(image_groups)} image groups..."
                )

                result_parts = []

                for group_index, frames in enumerate(
                    cqdm(image_groups)
                ):
                    if mm.processing_interrupted():
                        raise mm.InterruptProcessingException()

                    current_user_content = json.loads(
                        json.dumps(base_user_content)
                    )

                    for frame in frames:
                        self._append_frame(
                            current_user_content,
                            frame,
                        )

                    current_messages = messages + [
                        {
                            "role": "user",
                            "content": current_user_content,
                        }
                    ]

                    result = self._create_completion(
                        current_messages,
                        seed,
                        parameters,
                    )

                    raw_text = (
                        result["choices"][0]["message"].get(
                            "content",
                            "",
                        )
                    )

                    text = self._remove_thinking(
                        raw_text
                    )

                    output_list.append(text)

                    if len(image_groups) > 1:
                        result_parts.append(
                            f"====== Item {group_index + 1} ======"
                        )

                    result_parts.append(text)

                    if save_states:
                        history_messages.extend(
                            [
                                {
                                    "role": "user",
                                    "content": current_user_content,
                                },
                                {
                                    "role": "assistant",
                                    "content": text,
                                },
                            ]
                        )

                output = "\n\n".join(
                    result_parts
                )

            # =====================================================
            # images
            # =====================================================
            elif inference_mode == "images":
                all_frames = [
                    frame
                    for group in image_groups
                    for frame in group
                ]

                if not all_frames:
                    raise ValueError(
                        "images 模式没有收到有效图片。"
                    )

                print(
                    "[llama-cpp_vlm] "
                    f"[Images Mode] Packing "
                    f"{len(all_frames)} images into one completion..."
                )

                combined_user_content = json.loads(
                    json.dumps(base_user_content)
                )

                for frame in all_frames:
                    self._append_frame(
                        combined_user_content,
                        frame,
                    )

                current_messages = messages + [
                    {
                        "role": "user",
                        "content": combined_user_content,
                    }
                ]

                result = self._create_completion(
                    current_messages,
                    seed,
                    parameters,
                )

                raw_text = (
                    result["choices"][0]["message"].get(
                        "content",
                        "",
                    )
                )

                output = self._remove_thinking(
                    raw_text
                )

                output_list = [output]

                if save_states:
                    history_messages.extend(
                        [
                            {
                                "role": "user",
                                "content": combined_user_content,
                            },
                            {
                                "role": "assistant",
                                "content": output,
                            },
                        ]
                    )

            # =====================================================
            # video
            # =====================================================
            elif inference_mode == "video":
                print(
                    "[llama-cpp_vlm] "
                    f"[Video Mode] Processing "
                    f"{len(image_groups)} video clips..."
                )

                result_parts = []

                for video_index, frames in enumerate(
                    cqdm(image_groups)
                ):
                    if mm.processing_interrupted():
                        raise mm.InterruptProcessingException()

                    if not frames:
                        continue

                    frame_count = min(
                        len(frames),
                        max_frames,
                    )

                    indices = np.linspace(
                        0,
                        len(frames) - 1,
                        frame_count,
                        dtype=int,
                    )

                    sampled_frames = [
                        frames[index]
                        for index in indices
                    ]

                    current_user_content = json.loads(
                        json.dumps(base_user_content)
                    )

                    for frame in sampled_frames:
                        self._append_frame(
                            current_user_content,
                            frame,
                        )

                    current_messages = messages + [
                        {
                            "role": "user",
                            "content": current_user_content,
                        }
                    ]

                    result = self._create_completion(
                        current_messages,
                        seed,
                        parameters,
                    )

                    raw_text = (
                        result["choices"][0]["message"].get(
                            "content",
                            "",
                        )
                    )

                    text = self._remove_thinking(
                        raw_text
                    )

                    output_list.append(text)

                    if len(image_groups) > 1:
                        result_parts.append(
                            f"====== Video Clip {video_index + 1} ======"
                        )

                    result_parts.append(text)

                    if save_states:
                        history_messages.extend(
                            [
                                {
                                    "role": "user",
                                    "content": current_user_content,
                                },
                                {
                                    "role": "assistant",
                                    "content": text,
                                },
                            ]
                        )

                output = "\n\n".join(
                    result_parts
                )

            else:
                raise ValueError(
                    f"Unknown inference mode: {inference_mode}"
                )

        # ---------------------------------------------------------
        # 7. 纯文本推理
        # ---------------------------------------------------------
        else:
            text_string = "".join(
                item["text"]
                for item in base_user_content
                if item.get("type") == "text"
            )

            current_messages = messages + [
                {
                    "role": "user",
                    "content": text_string,
                }
            ]

            result = self._create_completion(
                current_messages,
                seed,
                parameters,
            )

            raw_text = (
                result["choices"][0]["message"].get(
                    "content",
                    "",
                )
            )

            output = self._remove_thinking(
                raw_text
            )

            output_list = [output]

            if save_states:
                history_messages.extend(
                    [
                        {
                            "role": "user",
                            "content": text_string,
                        },
                        {
                            "role": "assistant",
                            "content": output,
                        },
                    ]
                )

        # ---------------------------------------------------------
        # 8. 保存或清理状态
        # ---------------------------------------------------------
        if save_states:
            print(
                f"[llama-cpp_vlm] Saving state id={uid}..."
            )

            messages.extend(history_messages)

            LLAMA_CPP_STORAGE.messages[uid] = (
                self.sanitize_messages(messages)
            )
        else:
            LLAMA_CPP_STORAGE.clean_state(uid)

        # 模型缓存清理和卸载统一由外层 process() 的 finally 执行。
        return (
            output,
            output_list,
            int(uid),
        )
        
class llama_cpp_parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 1, "tooltip": "最大生成 Token 数。"}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 1000, "step": 1, "tooltip": "Top-K 采样范围。0 表示不限制。"}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "核采样概率。数值越低，输出越保守。"}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,  "tooltip": "最小概率采样阈值。"}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,  "tooltip": "Typical-P 采样参数。"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01,  "tooltip": "温度参数 越高越随机，越低越稳定。"}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,  "tooltip": "重复惩罚系数。"}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,  "tooltip": "频率惩罚，抑制重复出现的词元。"}),
                "present_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,  "tooltip": "存在惩罚，鼓励模型使用新的词元。"}),
                "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1,  "tooltip": "Mirostat 模式 0=关闭，1/2=启用不同版本。"}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,  "tooltip": "Mirostat 学习率。"}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01,  "tooltip": "Mirostat 目标熵。"}),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "对话状态 ID  -1：使用当前节点的唯一 ID  指定其他数值：多个节点可以共享同一个对话状态"
                }),
            }
        }
    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    def process(self, **kwargs):
        return (kwargs,)
    
class llama_cpp_clean_states:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "清除指定的文本对话状态 (-1：清除全部状态)"
                }),
            },
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any, state_uid):
        print(f"[llama-cpp_vlm] Cleaning up saved states {state_uid}...")
        LLAMA_CPP_STORAGE.clean_state(state_uid)
        return (any,)

class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type,)}}
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any):
        print("[llama-cpp_vlm] Unloading llama model...")
        LLAMA_CPP_STORAGE.clean()
        return (any,)

class json_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (["simple", "Qwen3-VL", "Qwen2.5-VL"], {"default": "simple"}),
                "label": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "只输出指定标签对应的边界框 留空时输出全部边界框."
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("BBOX", "IMAGE")
    RETURN_NAMES = ("bboxes", "image_list")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, json, mode, label, image=None):
        mode_val = mode[0] if isinstance(mode, list) and mode else "simple"
        label_val = label[0] if isinstance(label, list) and label else ""

        flat_images_list = []
        original_structure = []
    
        if image is not None:
            for img_batch in image:
                if img_batch is None:
                    continue
                if img_batch.ndim == 3:
                    flat_images_list.append(img_batch.unsqueeze(0))
                    original_structure.append(1)
                elif img_batch.ndim == 4:
                    count = img_batch.shape[0]
                    original_structure.append(count)
                    for n in range(count):
                        flat_images_list.append(img_batch[n:n+1])
        
        total_images = len(flat_images_list)
        output_bboxes = []
        processed_flat_results = []
        
        for i, j in enumerate(json):
            bboxes = parse_json(j)
            if isinstance(bboxes, dict):
                bboxes = [bboxes]
            
            if label_val != "":
                bboxes = [
                    item for item in bboxes 
                    if isinstance(item, dict) and (item.get("label") == label_val or item.get("text_content") == label_val)
                ]

            if total_images > 0:
                curr_idx = i if i < total_images else (total_images - 1)
                curr_img = flat_images_list[curr_idx]
                
                try:
                    res_img = draw_bbox(curr_img[0], bboxes, mode_val)
                    if res_img.ndim == 3:
                        res_img = res_img.unsqueeze(0)
                    elif res_img.ndim == 4 and res_img.shape[0] > 1:
                        res_img = res_img[0:1]
                        
                    processed_flat_results.append(res_img)
                except Exception as e:
                    print(f"Error drawing on image {curr_idx}: {e}")
                    processed_flat_results.append(curr_img)
                    
            if mode_val in ["Qwen3-VL", "Qwen2.5-VL"]:
                if total_images == 0:
                    raise ValueError("Image required for Qwen mode")
                curr_idx = i if i < total_images else (total_images - 1)
                bbox = qwen3bbox(flat_images_list[curr_idx][0], bboxes)
            else:
                bbox = [tuple(item["bbox_2d"]) for item in bboxes if isinstance(item, dict) and "bbox_2d" in item]
                
            output_bboxes.append(bbox)
            
        restructured_images_list = []
        cursor = 0
        for count in original_structure:
            chunk = processed_flat_results[cursor : cursor + count]
            if chunk:
                restructured_images_list.append(torch.cat(chunk, dim=0))
            cursor += count
            
        return (output_bboxes, restructured_images_list)

class SEG:
    def __init__(self, cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper=None):
        self.cropped_image = cropped_image
        self.cropped_mask = cropped_mask
        self.confidence = confidence
        self.crop_region = crop_region
        self.bbox = bbox
        self.label = label
        self.control_net_wrapper = control_net_wrapper
        
    def __repr__(self):
        return (f"SEG(cropped_image={self.cropped_image}, cropped_mask=shape{self.cropped_mask.shape}, confidence={self.confidence}, bbox={self.bbox}, label='{self.label}'), control_net_wrapper={self.control_net_wrapper}")

class bbox_to_segs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        
        seg_list = []
        image_for_cropping = image[0] 
        
        # 展平嵌套的 bbox
        flat_bboxes = []
        for item in bboxes:
            if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (list, tuple)):
                flat_bboxes.extend(item)
            else:
                flat_bboxes.append(item)
        
        for bbox in flat_bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            
            crop_region = [x1_exp, y1_exp, x2_exp, y2_exp]
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                print(f"Warning: Skipping bbox with invalid expanded size: {crop_region}")
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = min(crop_w, local_x1 + (x2 - x1))
            local_y2 = min(crop_h, local_y1 + (y2 - y1))
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
            cropped_mask_np = local_mask_np
            cropped_img_padded = torch.zeros((crop_h, crop_w, 3), dtype=image.dtype, device=image.device)
            
            src_x_start = max(0, x1_exp)
            src_y_start = max(0, y1_exp)
            src_x_end = min(width, x2_exp)
            src_y_end = min(height, y2_exp)
            
            dst_x_start = src_x_start - x1_exp
            dst_y_start = src_y_start - y1_exp
            dst_x_end = src_x_end - x1_exp
            dst_y_end = src_y_end - y1_exp
            
            if src_x_end > src_x_start and src_y_end > src_y_start:
                source_crop = image_for_cropping[src_y_start:src_y_end, src_x_start:src_x_end, :]
                cropped_img_padded[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = source_crop
                
            cropped_image_tensor = cropped_img_padded.permute(2, 0, 1).unsqueeze(0)
            
            seg = SEG(
                cropped_image=cropped_image_tensor,
                cropped_mask=cropped_mask_np,
                confidence=np.array([0.9], dtype=np.float32),
                crop_region=crop_region,
                bbox=np.array(bbox, dtype=np.float32),
                label="bbox"
            )
            
            seg_list.append(seg)
            
        segs = (mask_shape, seg_list)
        return (segs,)
    
class bbox_to_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image, dilation, feather):
        masks = []
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        combined_full_mask = torch.zeros(mask_shape, dtype=torch.float32, device=image.device)
        
        flat_bboxes = []
        for item in bboxes:
            if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (list, tuple)):
                flat_bboxes.extend(item)
            else:
                flat_bboxes.append(item)

        for bbox in flat_bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = min(crop_w, local_x1 + (x2 - x1))
            local_y2 = min(crop_h, local_y1 + (y2 - y1))
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
            current_full_mask_np = np.zeros(mask_shape, dtype=np.float32)
            x1_c, y1_c = max(0, x1_exp), max(0, y1_exp)
            x2_c, y2_c = min(width, x2_exp), min(height, y2_exp)
            
            if x2_c > x1_c and y2_c > y1_c:
                src_x1, src_y1 = max(0, -x1_exp), max(0, -y1_exp)
                src_x2 = src_x1 + (x2_c - x1_c)
                src_y2 = src_y1 + (y2_c - y1_c)
                current_full_mask_np[y1_c:y2_c, x1_c:x2_c] = local_mask_np[src_y1:src_y2, src_x1:src_x2]
                
            current_full_mask_tensor = torch.from_numpy(current_full_mask_np).to(image.device)
            combined_full_mask = torch.maximum(combined_full_mask, current_full_mask_tensor)
            
        masks.append(combined_full_mask.unsqueeze(0))
        return (torch.cat(masks, dim=0),)

class bboxes_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "bbox_index": ("INT", {
                    "default": 0,
                    "min": -998,
                    "max": 999,
                    "step": 1,
                    "tooltip": "图像中的 BBox 索引。设置为 999 以获取所有 BBox。."
                }),
            }
        }
    
    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image_index, bbox_index):
        if not bboxes:
            return ([],)
            
        # 兼容 nested list 与 flat list
        if isinstance(bboxes[0], (list, tuple)) and len(bboxes[0]) > 0 and isinstance(bboxes[0][0], (list, tuple)):
            img_idx = min(max(0, image_index), len(bboxes) - 1)
            target_bboxes = bboxes[img_idx]
        else:
            target_bboxes = bboxes

        if not target_bboxes:
            return ([],)

        if bbox_index == 999:
            return (target_bboxes,)
        
        b_idx = min(max(0, bbox_index), len(target_bboxes) - 1)
        return ([target_bboxes[b_idx]],)

def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split('.')
    for key in keys:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return default
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

class parse_json_node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "key": ("STRING", {"default": ""}),
                "default": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = (any_type, "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("any", "string", "int", "float", "boolean")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, input, key="", default=""):
        if isinstance(input, str):
            input_list = [input]
        else:
            input_list = input
            
        res_any, res_str, res_int, res_float, res_bool = [], [], [], [], []
        
        for json_str in input_list:
            if not key:
                val = json_str
            else:
                parsed_json = json_str.strip()
                if parsed_json.startswith("```json"):
                    parsed_json = parsed_json[7:]
                if parsed_json.startswith("```"):
                    parsed_json = parsed_json[3:]
                if parsed_json.endswith("```"):
                    parsed_json = parsed_json[:-3]
                val = get_nested_value(parsed_json.strip(), key, default)
            
            res_any.append(val)
            res_str.append(str(val) if val is not None else "")
            
            try:
                res_int.append(int(val))
            except Exception:
                res_int.append(0)
                
            try:
                res_float.append(float(val))
            except Exception:
                res_float.append(0.0)
                
            try:
                if isinstance(val, bool):
                    res_bool.append(val)
                else:
                    res_bool.append(str(val).lower() == "true")
            except Exception:
                res_bool.append(False)
                
        if len(res_any) == 1:
            return (res_any[0], res_str[0], res_int[0], res_float[0], res_bool[0])
        
        return (res_any, res_str, res_int, res_float, res_bool)

class remove_code_block:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "label": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, input, label=""):
        if isinstance(input, list):
            input_str = "\n".join(input)
        else:
            input_str = input
            
        val = input_str.strip()
        if label and val.startswith(f"```{label}"):
            val = val[len(f"```{label}"):]
        elif val.startswith("```"):
            lines = val.split("\n", 1)
            if len(lines) > 1:
                val = lines[1]
            else:
                val = lines[0].removeprefix("```")
                
        val = val.strip()
        if val.endswith("```"):
            val = val[:-3].strip()
            
        return (val,)

class PromptEnhancerPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (["Qwen-Image [EN]", "Qwen-Image [ZH]", "Qwen-Image 2512 [EN]", "Qwen-Image 2512 [ZH]", "Qwen-Image-Edit", "Qwen-Image-Edit 2509", "Qwen-Image-Edit 2511", "Z-Image Turbo", "Flux.2 T2I", "Flux.2 I2I", "Wan T2V [EN]", "Wan T2V [ZH]", "Wan I2V [EN]", "Wan I2V [ZH]", "Wan I2V Full-Auto [EN]", "Wan I2V Full-Auto [ZH]", "Wan FLF2V [EN]", "Wan FLF2V [ZH]"], )
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "main"
    CATEGORY = "llama-cpp-vlm"
    
    def main(self, preset):
        match preset:
            case "Qwen-Image [EN]":
                return (QWEN_IMAGE_EN,)
            case "Qwen-Image [ZH]":
                return (QWEN_IMAGE_ZH,)
            case "Qwen-Image 2512 [EN]":
                return (QWEN_IMAGE_2512_EN,)
            case "Qwen-Image 2512 [ZH]":
                return (QWEN_IMAGE_2512_ZH,)
            case "Qwen-Image-Edit":
                return (QWEN_IMAGE_EDIT,)
            case "Qwen-Image-Edit 2509":
                return (QWEN_IMAGE_EDIT_2509,)
            case "Qwen-Image-Edit 2511":
                return (QWEN_IMAGE_EDIT_2511,)
            case "Z-Image Turbo":
                return (ZIMAGE_TURBO,)
            case "Flux.2 T2I":
                return (FLUX2_T2I,)
            case "Flux.2 I2I":
                return (FLUX2_I2I,)
            case "Wan T2V [EN]":
                return (WAN_T2V_EN,)
            case "Wan T2V [ZH]":
                return (WAN_T2V_ZH,)
            case "Wan I2V [EN]":
                return (WAN_I2V_EN,)
            case "Wan I2V [ZH]":
                return (WAN_I2V_ZH,)
            case "Wan I2V Full-Auto [EN]":
                return (WAN_I2V_EMPTY_EN,)
            case "Wan I2V Full-Auto [ZH]":
                return (WAN_I2V_EMPTY_ZH,)
            case "Wan FLF2V [EN]":
                return (WAN_FLF2V_EN,)
            case "Wan FLF2V [ZH]":
                return (WAN_FLF2V_ZH,)
            case _:
                raise ValueError(f'Unknown preset: "{preset}"')

class llama_cpp_text_encoder:
    """
    统一的 Llama-cpp Text Encoder 节点。
    支持替代 Z-Image, Lumina2, Qwen-Image, MiniMax T2V, Boogu-Image, JoyImage, Mage-Flow 等原生文本编码器。
    """
    PRESETS = {
        "Boogu-Image": {
            "template": "<|im_start|>system\nYou are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n",
            "layer_idx": -1,
            "trim_template": False,
        },
        "JoyImage": {
            "template": "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "layer_idx": -1,
            "trim_template": True,
        },
        "Lumina2": {
            "template": "{}",
            "layer_idx": -2,
            "trim_template": False,
        },
        "Mage-Flow": {
            "template": "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "layer_idx": -1,
            "trim_template": True,
        },
        "MiniMax-T2V": {
            "template": "{}",
            "layer_idx": 49,
            "trim_template": False,
            "is_minimax_t2v": True,
        },
        "Qwen-Image": {
            "template": "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "layer_idx": -1,
            "trim_template": True,
        },
        "Z-Image": {
            "template": "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "layer_idx": -2,
            "trim_template": False,
        },
    }
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "type": (list(s.PRESETS.keys()), {"default": "Z-Image"}),
                "force_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the model after inference."
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "llama-cpp-vlm"
    
    def _extract_native_embeddings(self, llm, prompt_text, target_layer, tokens):
        import ctypes
        
        # 1. 目标层换算 (负数层转正数)
        model = getattr(llm, "_model", None) or getattr(llm, "model", None) or llm
        n_layers = getattr(model, "n_layer", None) or getattr(llm, "n_layer", None)
        if callable(n_layers):
            n_layers = n_layers()
            
        pos_layer = (n_layers + target_layer) if (n_layers is not None and target_layer < 0) else target_layer
        
        # 尝试通过原生接口设置目标层
        if hasattr(model, "set_target_layer_ids") and callable(model.set_target_layer_ids):
            try: model.set_target_layer_ids([pos_layer])
            except Exception: pass
        elif hasattr(model, "target_layer_ids"):
            if callable(model.target_layer_ids):
                try: 
                    t_ids = model.target_layer_ids()
                    if isinstance(t_ids, list):
                        t_ids.clear()
                        t_ids.append(pos_layer)
                except Exception: pass
                
        num_tokens = len(tokens)
        
        # 2. 安全清理状态
        if hasattr(llm, "n_tokens"):
            llm.n_tokens = 0
        if hasattr(llm, "_ctx"):
            if hasattr(llm._ctx, "kv_cache_clear"):
                llm._ctx.kv_cache_clear()
            elif hasattr(llm._ctx, "memory_clear"):
                llm._ctx.memory_clear(True)
                
        # 3. 执行推理计算
        llm.eval(tokens)
        
        # 4. 从上下文中提取深层特征
        ctx_ptr = getattr(llm, "ctx", None) or getattr(getattr(llm, "_ctx", None), "ctx", None)
        n_embd = llm.n_embd() if callable(getattr(llm, "n_embd", None)) else getattr(llm, "n_embd", 4096)
        
        emb_np = None
        
        # 优先提取指定层激活状态
        for fn_name in ["llama_get_layer_state", "llama_get_layer_embeddings", "llama_get_layer_output"]:
            if hasattr(llama_cpp, fn_name):
                fn = getattr(llama_cpp, fn_name)
                try:
                    ptr = fn(ctx_ptr, ctypes.c_int(pos_layer))
                    if ptr:
                        emb_np = np.ctypeslib.as_array(ptr, shape=(num_tokens, n_embd)).copy()
                        break
                except Exception: pass
                
        # 备选提取
        if emb_np is None:
            if hasattr(llama_cpp, "llama_get_embeddings_ith"):
                token_embeds = []
                for i in range(num_tokens):
                    ptr = llama_cpp.llama_get_embeddings_ith(ctx_ptr, ctypes.c_int(i))
                    if ptr:
                        token_embeds.append(np.ctypeslib.as_array(ptr, shape=(n_embd,)).copy())
                if len(token_embeds) == num_tokens:
                    emb_np = np.stack(token_embeds, axis=0)
                    
            if emb_np is None and hasattr(llama_cpp, "llama_get_embeddings"):
                ptr = llama_cpp.llama_get_embeddings(ctx_ptr)
                if ptr:
                    emb_np = np.ctypeslib.as_array(ptr, shape=(num_tokens, n_embd)).copy()
                    
        if emb_np is None:
            raise RuntimeError(f"提取 Layer {pos_layer} 失败！请确保模型加载时启用了 'embeddings=True'。")
            
        # 5. 转为 Tensor 并规范维度为 [1, seq_len, hidden_dim]
        emb = torch.tensor(emb_np, dtype=torch.float32)
        if emb.ndim == 2:
            emb = emb.unsqueeze(0)
        elif emb.ndim == 1:
            emb = emb.unsqueeze(0).unsqueeze(0)
            
        return emb
    
    def encode(self, llama_model, prompt, type, force_offload):
        if not LLAMA_CPP_STORAGE.llm:
            LLAMA_CPP_STORAGE.load_model(llama_model)
        
        LLAMA_CPP_STORAGE.ensure_embedding_mode(True)
            
        llm = LLAMA_CPP_STORAGE.llm
        cfg = self.PRESETS[type]
        
        target_layer = cfg["layer_idx"]
        prompt_text = cfg["template"].format(prompt) if "{}" in cfg["template"] else prompt
        
        # 【关键修复 1】：必须加 special=True，否则 <|im_start|> 无法被解析为 151644
        tokens = llm.tokenize(prompt_text.encode("utf-8"), add_bos=False, special=True)
        if len(tokens) == 0:
            tokens = [151643]
        
        # MiniMax 严格要求纯净文本，不能有任何 BOS (151643 / 151644 / 1)
        if cfg.get("is_minimax_t2v", False) and len(tokens) > 1:
            if tokens[0] in [151643, 151644, 1]:
                tokens = tokens[1:]  # 剥离开头的 BOS Token
        
        # 提取特征
        hidden_tensor = self._extract_native_embeddings(llm, prompt_text, target_layer, tokens)
        
        # 【关键修复 2】：更鲁棒的裁切验证
        if cfg.get("trim_template", False) and len(tokens) > 3:
            trim_idx = 0
            count_im_start = 0
            for i in range(len(tokens) - 2):
                if tokens[i] == 151644:  # <|im_start|>
                    count_im_start += 1
                    if count_im_start == 2: 
                        # 找到了 user block 的开始
                        if tokens[i+1] == 872 and tokens[i+2] == 198: # user \n
                            trim_idx = i + 3
                        else:
                            trim_idx = i + 1
                        break
                
            # 执行特征截断
            if trim_idx > 0 and hidden_tensor.shape[1] > trim_idx:
                hidden_tensor = hidden_tensor[:, trim_idx:, :]
            else:
                print(f"\n[llama-cpp_vlm] Warning: Trim failed! Token IDs: {tokens[:10]}...")
                
        seq_len = hidden_tensor.shape[1]
        pooled = torch.zeros((hidden_tensor.shape[0], hidden_tensor.shape[-1]), dtype=torch.float32)
        
        # 构造 Conditioning 字典
        cond_dict = {"pooled_output": pooled}
        
        # MiniMax T2V 专属标记
        if cfg.get("is_minimax_t2v", False):
            cond_dict["minimax_token_tags"] = torch.ones(seq_len, dtype=torch.long)
        
        print(f"\n[MiniMax Diagnostic] Tensor Shape: {hidden_tensor.shape}, Mean: {hidden_tensor.mean().item():.4f}, Std: {hidden_tensor.std().item():.4f}\n")
        conditioning = [[hidden_tensor, cond_dict]]
        
        if force_offload:
            LLAMA_CPP_STORAGE.clean()
        
        return (conditioning,)

NODE_CLASS_MAPPINGS = {
    "llama_cpp_model_loader": llama_cpp_model_loader,
    "llama_cpp_instruct_adv": llama_cpp_instruct_adv,
    "llama_cpp_parameters": llama_cpp_parameters,
    "llama_cpp_unload_model": llama_cpp_unload_model,
    "llama_cpp_clean_states": llama_cpp_clean_states,
    "parse_json_node": parse_json_node,
    "json_to_bbox": json_to_bbox,
    "bbox_to_segs": bbox_to_segs,
    "bbox_to_mask": bbox_to_mask,
    "bboxes_to_bbox": bboxes_to_bbox,
    "remove_code_block": remove_code_block,
    "PromptEnhancerPreset": PromptEnhancerPreset,
    "llama_cpp_text_encoder": llama_cpp_text_encoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_model_loader": "Llama-cpp 模型加载器",
    "llama_cpp_instruct_adv": "Llama-cpp 图像/文本指令",
    "llama_cpp_parameters": "Llama-cpp 推理参数",
    "llama_cpp_unload_model": "Llama-cpp 卸载模型",
    "llama_cpp_clean_states": "Llama-cpp 清理对话状态",
    "parse_json_node": "解析 JSON",
    "json_to_bbox": "JSON 转边界框",
    "bbox_to_segs": "边界框转 SEGS",
    "bbox_to_mask": "边界框转 MASK",
    "bboxes_to_bbox": "选择边界框",
    "remove_code_block": "移除代码块标记",
    "PromptEnhancerPreset": "提示词增强预设",
    "llama_cpp_text_encoder": "Llama-cpp 文本编码器（测试版）",
}