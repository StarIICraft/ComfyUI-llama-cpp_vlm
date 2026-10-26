from __future__ import annotations

import base64
import json
import math
import os
import re
import warnings
from io import BytesIO
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np
import requests
import torch
from PIL import Image, ImageOps

import comfy.model_management as mm
import comfy.utils


PROTOCOLS = [
    "OpenAI Images",
    "Gemini 原生",
    "Nano Banana OpenAI兼容",
    "Chat Completions 图像",
    "通用 JSON",
    "通用 multipart",
]

RATIOS = [
    "自动", "1:1", "2:3", "3:2", "3:4", "4:3",
    "4:5", "5:4", "9:16", "16:9", "21:9",
    "1:4", "4:1", "1:8", "8:1",
]

MAX_INPUT_IMAGES = 8
MAX_IMAGE_BYTES = 40 * 1024 * 1024


def check_interrupt():
    mm.throw_exception_if_processing_interrupted()


def valid_url(value):
    value = str(value or "").strip()
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in ("http", "https")
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.fragment
        )
    except ValueError:
        valid = False

    if not valid:
        raise ValueError(
            "接口地址必须是有效的 HTTP/HTTPS 地址，"
            "不能含账号密码或 fragment。"
        )
    return value


def endpoint_url(base, endpoint, default):
    """
    自定义 endpoint 优先。
    基础地址允许包含 /v1 等服务商前缀。
    """
    endpoint = str(endpoint or "").strip()
    if endpoint.startswith(("http://", "https://")):
        return valid_url(endpoint)

    base = valid_url(base).rstrip("/")
    parsed = urlsplit(base)
    if parsed.query:
        raise ValueError(
            "基础地址不能包含查询参数；"
            "需要查询参数时请在自定义端点填写完整 URL。"
        )

    known = (
        "/images/edits", "/images/generations",
        "/chat/completions", ":generateContent",
        ":streamGenerateContent",
    )

    if not endpoint and parsed.path.endswith(known):
        return base

    path = "/" + (endpoint or default).lstrip("/")
    if parsed.path.endswith(path):
        return base

    base_path = parsed.path.rstrip("/")
    if endpoint:
        for suffix in (
            "/images/edits", "/images/generations",
            "/chat/completions",
        ):
            if base_path.endswith(suffix):
                base_path = base_path[:-len(suffix)]
                break

        if (
            ":generateContent" in base_path
            or ":streamGenerateContent" in base_path
        ):
            marker = base_path.find("/models/")
            if marker >= 0:
                base_path = base_path[:marker]

    for prefix in ("/v1beta", "/v1"):
        if base_path.endswith(prefix) and path.startswith(prefix + "/"):
            path = path[len(prefix):]
            break

    return urlunsplit((
        parsed.scheme, parsed.netloc, base_path + path, "", ""
    ))


def ratio_value(text):
    if text in ("自动", "Auto", "", None):
        return None
    a, b = str(text).split(":", 1)
    a, b = float(a), float(b)
    if a <= 0 or b <= 0:
        raise ValueError("图像比例必须大于 0。")
    return a / b


def nearest_ratio(ratio):
    return min(
        RATIOS[1:],
        key=lambda x: abs(math.log(ratio_value(x) / ratio)),
    )


def prepare_dimensions(width, height, aspect, level, limit_ratio, model):
    """
    返回：像素尺寸或 None、是否手动指定、诊断信息。
    手动宽高优先；不将输出图片本身强行缩放。
    """
    width, height = int(width), int(height)
    if width < 0 or height < 0:
        raise ValueError("输出宽高不能小于 0。")
    if bool(width) != bool(height):
        raise ValueError(
            "输出宽度和高度必须同时大于 0，或同时设为 0。"
        )

    manual = width > 0 and height > 0
    notes = []

    if manual:
        notes.append(
            f"手动尺寸：{width}x{height}，忽略比例和分辨率档位。"
        )
    else:
        r = ratio_value(aspect)
        if r is None:
            return None, False, notes
        longest = {
            "无": 1024, "1K": 1024, "2K": 2048, "4K": 3840
        }[level]
        width = longest if r >= 1 else longest * r
        height = longest / r if r >= 1 else longest
        width = max(16, round(width / 16) * 16)
        height = max(16, round(height / 16) * 16)

    original = (int(width), int(height))
    area = width * height
    ratio = width / height

    if limit_ratio and (ratio > 3 or ratio < 1 / 3):
        # 3*k 与 k 均为16的倍数，保持严格3:1。
        short = max(16, round(math.sqrt(area / 3) / 16) * 16)
        if ratio > 3:
            width, height = short * 3, short
        else:
            width, height = short, short * 3
        notes.append(
            f"宽高比限制：{original[0]}x{original[1]}"
            f" → {width}x{height}。"
        )

    # 根据用户提供的服务文档，对精确模型名应用约束。
    name = str(model).strip().lower().rsplit("/", 1)[-1]
    if name == "gpt-image-2":
        ratio = width / height
        if not 1 / 3 <= ratio <= 3:
            raise ValueError(
                "gpt-image-2 超过 1:3～3:1，"
                "请开启宽高比限制或修改尺寸。"
            )

        target_area = min(8_294_400, max(655_360, width * height))
        target_area = min(
            target_area, 3840 ** 2 / max(ratio, 1 / ratio)
        )
        target_w = math.sqrt(target_area * ratio)
        target_h = math.sqrt(target_area / ratio)

        best = None
        best_score = float("inf")
        for w in range(16, 3841, 16):
            center = w / ratio
            candidates = {
                int(math.floor(center / 16) * 16),
                int(math.ceil(center / 16) * 16),
                int(round(target_area / w / 16) * 16),
            }
            for h in candidates:
                if not 16 <= h <= 3840:
                    continue
                pixels = w * h
                if not 655_360 <= pixels <= 8_294_400:
                    continue
                if max(w, h) > 3 * min(w, h):
                    continue
                score = (
                    8 * abs(math.log((w / h) / ratio))
                    + abs(math.log(pixels / target_area))
                    + 0.001 * (
                        abs(w - target_w) + abs(h - target_h)
                    ) / 3840
                )
                if score < best_score:
                    best, best_score = (w, h), score

        if best is None:
            raise ValueError("无法找到满足 gpt-image-2 约束的尺寸。")
        if best != (width, height):
            notes.append(
                f"模型尺寸约束调整：{width}x{height}"
                f" → {best[0]}x{best[1]}。"
            )
        width, height = best

    return (int(width), int(height)), manual, notes


def collect_images(inputs, max_mp):
    max_mp = float(max_mp)
    if not math.isfinite(max_mp) or max_mp < 0:
        raise ValueError(
            "参考图百万像素上限必须是大于等于 0 的有限数值。"
        )
    max_pixels = max(1, int(max_mp * 1_000_000)) if max_mp else 0
    outputs, notes = [], []

    for index in range(1, MAX_INPUT_IMAGES + 1):
        tensor = inputs.get(f"参考图_{index}")
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.ndim not in (3, 4):
            raise ValueError(
                f"参考图_{index} 必须是 [H,W,C] 或 [B,H,W,C] 图像张量。"
            )
        frames = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
        for frame in frames:
            check_interrupt()
            if len(outputs) >= MAX_INPUT_IMAGES:
                raise ValueError(
                    "参考图总数最多8张，每个端口内的批次也计入总数。"
                )
            if frame.shape[-1] not in (1, 3, 4):
                raise ValueError("参考图只支持1、3、4通道。")
            if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                raise ValueError("参考图宽高必须大于0。")

            array = frame.detach().cpu().float().numpy()
            array = np.nan_to_num(array, nan=0, posinf=1, neginf=0)
            array = (np.clip(array, 0, 1) * 255 + 0.5).astype(np.uint8)
            if array.shape[-1] == 1:
                image = Image.fromarray(array[..., 0]).convert("RGB")
            else:
                image = Image.fromarray(array)
            old = image.size

            if max_pixels and image.width * image.height > max_pixels:
                scale = math.sqrt(
                    max_pixels / (image.width * image.height)
                )
                w = max(1, int(image.width * scale))
                h = max(1, int(image.height * scale))
                if w * h > max_pixels:
                    if w >= h:
                        w = max(1, max_pixels // h)
                    else:
                        h = max(1, max_pixels // w)
                image = image.resize(
                    (w, h), Image.Resampling.LANCZOS
                )

            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                outputs.append(buffer.getvalue())
            notes.append(
                f"参考图{len(outputs)}：{old[0]}x{old[1]}"
                f" → {image.width}x{image.height}"
            )
    return outputs, notes


def data_url(raw):
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def error_summary(response, api_key):
    message = "请求失败，请检查接口地址、密钥和参数。"
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or message)
            elif isinstance(error, str):
                message = error
            else:
                message = str(
                    body.get("message") or body.get("detail") or message
                )
    except ValueError:
        pass
    if api_key:
        message = message.replace(api_key, "[密钥已隐藏]")
    return message[:800]


def post_json(session, url, key, body, files, timeout, chunked):
    check_interrupt()
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    arguments = {"json": body}
    if files:
        arguments = {
            "data": {
                k: json.dumps(v, ensure_ascii=False)
                if isinstance(v, (dict, list, bool)) else str(v)
                for k, v in body.items() if v is not None
            },
            "files": files,
        }
    try:
        with session.post(
            url, headers=headers, timeout=(15, timeout),
            stream=bool(chunked), allow_redirects=False, **arguments,
        ) as response:
            if 300 <= response.status_code < 400:
                raise RuntimeError(
                    f"HTTP {response.status_code}："
                    "请直接填写最终接口地址，不自动跟随重定向。"
                )
            if not response.ok:
                raise RuntimeError(
                    f"HTTP {response.status_code}："
                    f"{error_summary(response, key)}"
                )

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/event-stream" in content_type:
                raise RuntimeError(
                    "服务端返回了 SSE，但当前节点只支持普通 JSON 分块传输。"
                )

            if chunked:
                chunks = []
                received = 0
                next_log = 4 * 1024 * 1024
                for chunk in response.iter_content(64 * 1024):
                    check_interrupt()
                    if chunk:
                        chunks.append(chunk)
                        received += len(chunk)
                        if received >= next_log:
                            print(
                                f"[图像API] 已接收 "
                                f"{received / 1024 / 1024:.1f} MiB",
                                flush=True,
                            )
                            next_log = received + 4 * 1024 * 1024
                raw = b"".join(chunks)
                try:
                    result = json.loads(raw.decode("utf-8-sig"))
                except (ValueError, UnicodeError):
                    raise RuntimeError(
                        "接口没有返回有效的普通 JSON。"
                    ) from None
            else:
                check_interrupt()
                try:
                    result = response.json()
                except ValueError:
                    raise RuntimeError("接口没有返回有效 JSON。") from None

    except requests.Timeout:
        raise RuntimeError(
            "API连接或读取超时，服务端可能仍在生成；"
            "为避免重复计费，不自动重试。"
        ) from None
    except requests.RequestException:
        raise RuntimeError(
            "API网络请求失败，请检查代理、证书和连接。"
        ) from None

    check_interrupt()
    if not isinstance(result, dict):
        raise RuntimeError("接口返回格式无效，需要 JSON 对象。")
    if result.get("error"):
        raise RuntimeError(
            "接口返回业务错误，请检查服务端记录和请求参数。"
        )
    return result


def extract_content(body):
    sources, texts = [], []
    seen = set()

    def add(source):
        if not isinstance(source, str):
            return
        source = source.strip()
        if source and source not in seen:
            seen.add(source)
            sources.append(source)

    def text(value):
        if not isinstance(value, str) or not value.strip():
            return
        value = value.strip()
        if value.startswith("data:image/") and not re.search(r"\s", value):
            add(value)
            return
        for match in re.finditer(
            r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+", value
        ):
            add(match.group(0))
        for match in re.finditer(
            r"!\[[^\]]*\]\(\s*(https?://[^\s)]+)"
            r"(?:\s+\"[^\"]*\")?\s*\)", value
        ):
            add(match.group(1))
        for match in re.finditer(r"https?://[^\s<>\"']+", value):
            url = match.group(0).rstrip(").,;，。；")
            if re.search(
                r"\.(png|jpe?g|webp|gif|bmp)(?:[?#]|$)", url, re.I
            ):
                add(url)
        if (
            value.startswith(("http://", "https://"))
            and not re.search(r"\s", value)
        ):
            add(value)
        safe = re.sub(
            r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
            "[图片数据已省略]", value,
        )
        if safe not in texts:
            texts.append(safe[:3000])

    def visit(value, depth=0):
        if depth > 20:
            return
        if isinstance(value, list):
            for item in value:
                visit(item, depth + 1)
        elif isinstance(value, str):
            text(value)
        elif isinstance(value, dict):
            inline = value.get("inlineData") or value.get("inline_data")
            if isinstance(inline, dict):
                mime = inline.get("mimeType") or inline.get("mime_type") or ""
                if not mime or str(mime).startswith("image/"):
                    add(inline.get("data"))

            for key in ("b64_json", "base64", "image_base64", "imageBase64"):
                add(value.get(key))

            for key in ("url", "image_url", "imageUrl"):
                item = value.get(key)
                if isinstance(item, dict):
                    item = item.get("url")
                if isinstance(item, str) and item.startswith((
                    "http://", "https://", "data:image/"
                )):
                    add(item)

            file_data = value.get("fileData") or value.get("file_data")
            if isinstance(file_data, dict):
                uri = (
                    file_data.get("fileUri") or file_data.get("file_uri")
                    or file_data.get("uri") or file_data.get("url")
                )
                if isinstance(uri, str) and uri.startswith((
                    "http://", "https://"
                )):
                    add(uri)

            for key in ("text", "revised_prompt", "output_text"):
                text(value.get(key))

            for key in (
                "candidates", "content", "parts", "data", "images", "image",
                "output", "result", "choices", "message",
            ):
                if key in value:
                    visit(value[key], depth + 1)

    visit(body)
    return sources, "\n".join(texts)


def decode_image(source, session, timeout):
    check_interrupt()
    if source.startswith(("http://", "https://")):
        valid_url(source)
        try:
            # Session不设置默认Authorization，下载图床不携带API Key。
            with session.get(
                source, headers={"Accept": "image/*"},
                timeout=(15, timeout), stream=True,
            ) as response:
                response.raise_for_status()
                chunks, total = [], 0
                for chunk in response.iter_content(64 * 1024):
                    check_interrupt()
                    if chunk:
                        total += len(chunk)
                        if total > MAX_IMAGE_BYTES:
                            raise RuntimeError("单张图片超过40MB安全限制。")
                        chunks.append(chunk)
                raw = b"".join(chunks)
        except requests.RequestException:
            raise RuntimeError(
                "返回图片下载失败，请检查图床连接。"
            ) from None
    else:
        if source.startswith("data:"):
            if "," not in source:
                raise RuntimeError("图片data URL格式无效。")
            encoded = source.split(",", 1)[1]
        else:
            encoded = source
        encoded = re.sub(r"\s+", "", encoded)
        if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
            raise RuntimeError("单张图片超过40MB安全限制。")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            raise RuntimeError("返回图片的base64无效。") from None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                image = ImageOps.exif_transpose(image)
                image.load()
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                return background.convert("RGB")
    except Exception as error:
        raise RuntimeError(
            f"图片解码失败：{type(error).__name__}"
        ) from None


def image_batch(images):
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    batch = torch.empty(
        (len(images), height, width, 3), dtype=torch.float32
    )
    for index, image in enumerate(images):
        check_interrupt()
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(
            image,
            ((width - image.width) // 2, (height - image.height) // 2),
        )
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        batch[index].copy_(torch.from_numpy(array))
    return batch


class UniversalImageAPINode:
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("生成图像", "执行报告")
    FUNCTION = "generate"
    CATEGORY = "API大模型/图像生成编辑"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "接口协议": (PROTOCOLS, {"default": "OpenAI Images"}),
                "接口地址": ("STRING", {
                    "default": "https://api.example.com",
                }),
                "接口密钥": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "留空读取UNIVERSAL_IMAGE_API_KEY；"
                        "填写的密钥可能保存到工作流，分享前请清除。"
                    ),
                }),
                "模型名称": ("STRING", {"default": "gpt-image-2"}),
                "提示词": ("STRING", {"default": "", "multiline": True}),
                "请求次数": ("INT", {
                    "default": 1, "min": 1, "max": 8,
                    "tooltip": "串行请求次数，不是最终图片数量；保留全部返回图片。",
                }),
                "输出宽度": ("INT", {
                    "default": 0, "min": 0, "max": 32768,
                    "tooltip": (
                        "0表示自动；可右键转换为输入端口连接其他节点。"
                        "手动宽高必须同时填写。"
                    ),
                }),
                "输出高度": ("INT", {
                    "default": 0, "min": 0, "max": 32768,
                    "tooltip": "与宽度同时大于0时，忽略图像比例和分辨率档位。",
                }),
                "图像比例": (RATIOS, {"default": "自动"}),
                "分辨率档位": (
                    ["无", "1K", "2K", "4K"], {"default": "无"}
                ),
                "限制输出宽高比": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "将超出1:3～3:1的请求尺寸调整到边界比例，"
                        "尽量保持像素总量；不裁切参考图。"
                    ),
                }),
                "参考图百万像素上限": ("FLOAT", {
                    "default": 2.5, "min": 0.0, "max": 256.0, "step": 0.1,
                    "tooltip": (
                        "每张参考图独立等比例缩小，"
                        "0表示不限制；不会放大图片。"
                    ),
                }),
                "生成质量": (
                    ["auto", "low", "medium", "high"],
                    {"default": "medium"},
                ),
                "分块传输": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "HTTP分块读取普通JSON；不是SSE，"
                        "不发送stream=true，不提供生成中间预览。"
                    ),
                }),
                "请求超时": ("INT", {
                    "default": 100, "min": 30, "max": 3600,
                    "tooltip": (
                        "读取响应的空闲超时，单位秒；"
                        "不是整个任务的严格总时限。"
                    ),
                }),
                "绕过代理": ("BOOLEAN", {"default": False}),
                "失败后继续": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "自定义端点": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "例如/images/edits、/v1/chat/completions，"
                        "也可填写完整URL。"
                    ),
                }),
                **{
                    f"参考图_{index}": ("IMAGE", {
                        "tooltip": "所有端口与批次合计最多8张参考图。",
                    })
                    for index in range(1, MAX_INPUT_IMAGES + 1)
                },
            },
        }

    def generate(
        self, 接口协议, 接口地址, 接口密钥, 模型名称, 提示词,
        请求次数, 输出宽度, 输出高度, 图像比例, 分辨率档位,
        限制输出宽高比, 参考图百万像素上限, 生成质量,
        分块传输, 请求超时, 绕过代理, 失败后继续,
        自定义端点="", **kwargs,
    ):
        check_interrupt()
        key = (
            str(接口密钥 or "").strip()
            or os.environ.get("UNIVERSAL_IMAGE_API_KEY", "").strip()
        )
        model = str(模型名称 or "").strip()
        prompt = str(提示词 or "").strip()
        if not key or not model or not prompt:
            raise ValueError("请填写有效的接口密钥、模型名称和提示词。")
        if 接口协议 not in PROTOCOLS:
            raise ValueError("未知接口协议。")
        if 生成质量 not in ("auto", "low", "medium", "high"):
            raise ValueError("生成质量必须是auto、low、medium或high。")
        if 分辨率档位 not in ("无", "1K", "2K", "4K"):
            raise ValueError("未知分辨率档位。")

        count = int(请求次数)
        if not 1 <= count <= 8:
            raise ValueError("请求次数必须为1～8。")
        timeout = int(请求超时)
        if timeout <= 0:
            raise ValueError("请求超时必须大于0。")

        dims, manual, notes = prepare_dimensions(
            输出宽度, 输出高度, 图像比例, 分辨率档位,
            限制输出宽高比, model,
        )
        refs, ref_notes = collect_images(kwargs, 参考图百万像素上限)
        notes.extend(ref_notes)
        size = f"{dims[0]}x{dims[1]}" if dims else "auto"
        name = model.lower().rsplit("/", 1)[-1]
        body, files = {}, None
        final_quality = "未发送"

        if 接口协议 == "Gemini 原生":
            normalized = model
            for prefix in ("v1beta/", "models/"):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]

            default = (
                "/v1beta/models/"
                + quote(normalized, safe="-._~")
                + ":generateContent"
            )
            url = endpoint_url(接口地址, 自定义端点, default)
            if ":streamGenerateContent" in urlsplit(url).path:
                raise ValueError(
                    "不使用SSE时，请使用generateContent端点。"
                )

            parts = [{"text": prompt}]
            parts.extend({"inlineData": {
                "mimeType": "image/png",
                "data": base64.b64encode(raw).decode("ascii"),
            }} for raw in refs)

            body = {"contents": [{"role": "user", "parts": parts}]}
            generation = {"responseModalities": ["IMAGE"]}
            config = {}

            if dims:
                config["aspectRatio"] = nearest_ratio(dims[0] / dims[1])
                if manual:
                    longest = max(dims)
                    config["imageSize"] = min(
                        ("1K", "2K", "4K"),
                        key=lambda x: abs(math.log(
                            longest / {
                                "1K": 1024, "2K": 2048, "4K": 4096
                            }[x]
                        )),
                    )
                elif 分辨率档位 != "无":
                    config["imageSize"] = 分辨率档位
            elif 分辨率档位 != "无":
                config["imageSize"] = 分辨率档位

            if config:
                generation["imageConfig"] = config
            body["generationConfig"] = generation
            notes.append(
                "Gemini不承诺精确像素输出，"
                "尺寸按支持的比例和档位近似映射。"
            )
            notes.append(
                "Gemini imageConfig："
                + json.dumps(config, ensure_ascii=False)
            )

        elif 接口协议 == "Chat Completions 图像":
            url = endpoint_url(
                接口地址, 自定义端点, "/v1/chat/completions"
            )
            content = prompt
            if refs:
                content = [{"type": "text", "text": prompt}]
                content.extend({
                    "type": "image_url",
                    "image_url": {"url": data_url(raw)},
                } for raw in refs)
                notes.append(
                    "聊天参考图使用多模态messages；"
                    "是否支持编辑由服务商决定。"
                )

            body = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 128,
            }
            if dims:
                notes.append(
                    "聊天接口不发送尺寸参数；"
                    "当前接口示例未定义尺寸字段，本地不会强制缩放输出。"
                )
            notes.append("聊天接口不发送生成质量参数。")

        else:
            if 接口协议 in ("Nano Banana OpenAI兼容", "通用 JSON"):
                # 保留原通用JSON路由，编辑端点由用户指定。
                default = "/v1/images/generations"
            else:
                default = (
                    "/v1/images/edits"
                    if refs else "/v1/images/generations"
                )
            url = endpoint_url(接口地址, 自定义端点, default)
            body = {
                "model": model, "prompt": prompt,
                "n": 1, "size": size,
            }

            if 接口协议 == "Nano Banana OpenAI兼容":
                body["response_format"] = "b64_json"
                if not manual and 分辨率档位 != "无":
                    body["quality"] = 分辨率档位.lower()
                if manual:
                    notes.append(
                        "香蕉兼容手动尺寸模式只发送size，"
                        "不使用分辨率档位生成quality。"
                    )
            elif name.startswith("seedream"):
                # 已提供的Seedream请求示例没有quality。
                notes.append(
                    "Seedream兼容处理：不发送quality参数。"
                )
            else:
                body["quality"] = 生成质量

            if 接口协议 == "OpenAI Images":
                if name.startswith("gpt-image-"):
                    body.pop("response_format", None)
                    if not refs:
                        body["output_format"] = "png"

                elif name in ("dall-e-2", "dall-e-3"):
                    body["response_format"] = "b64_json"
                    if size == "auto":
                        body["size"] = "1024x1024"

                    if name == "dall-e-3":
                        q = "hd" if 生成质量 == "high" else "standard"
                        body["quality"] = q
                        notes.append(
                            f"DALL·E 3质量兼容映射：{生成质量} → {q}。"
                        )
                        supported = {
                            "1024x1024", "1792x1024", "1024x1792"
                        }
                    else:
                        body.pop("quality", None)
                        supported = {
                            "256x256", "512x512", "1024x1024"
                        }

                    if body["size"] not in supported:
                        raise ValueError(
                            "该DALL·E模型支持的尺寸："
                            + "、".join(sorted(supported))
                        )

            multipart = (
                接口协议 == "通用 multipart"
                or (接口协议 == "OpenAI Images" and bool(refs))
            )
            if multipart:
                if not refs:
                    raise ValueError(
                        "通用multipart至少需要一张参考图。"
                    )
                upload_field = (
                    "image" if len(refs) == 1 else "image[]"
                )
                files = [
                    (
                        upload_field,
                        (f"reference_{i+1}.png", raw, "image/png"),
                    )
                    for i, raw in enumerate(refs)
                ]
            elif refs:
                body["image"] = [data_url(raw) for raw in refs]

            final_quality = str(body.get("quality", "未发送"))

        notes.insert(
            0,
            f"协议：{接口协议}；模型：{model}；参考图：{len(refs)}张",
        )
        # 不记录带查询参数的完整URL，避免输出签名信息。
        notes.append("请求路径：" + urlsplit(url).path)
        notes.append(
            f"目标尺寸：{size}；质量：{final_quality}；"
            f"HTTP分块读取：{'开启' if 分块传输 else '关闭'}"
        )
        print(
            "[图像API] " + "\n[图像API] ".join(notes),
            flush=True,
        )

        images, reports = [], list(notes)
        progress = comfy.utils.ProgressBar(count)
        with requests.Session() as session:
            session.trust_env = not bool(绕过代理)
            for index in range(count):
                check_interrupt()
                try:
                    result = post_json(
                        session, url, key, body, files,
                        timeout, 分块传输,
                    )
                    sources, text = extract_content(result)
                    if not sources:
                        detail = "接口未返回可识别的图片。"
                        if result.get("task_id") or result.get("taskId"):
                            detail += (
                                "返回了任务ID，"
                                "此版本不支持异步任务轮询。"
                            )
                        if text:
                            detail += "\n返回文本：" + text[:1000]
                        raise RuntimeError(detail)

                    succeeded, failures = 0, []
                    for source_index, source in enumerate(sources):
                        check_interrupt()
                        try:
                            images.append(
                                decode_image(source, session, timeout)
                            )
                            succeeded += 1
                        except mm.InterruptProcessingException:
                            raise
                        except Exception as error:
                            failures.append(
                                f"图片{source_index+1}：{error}"
                            )

                    if not succeeded:
                        raise RuntimeError(
                            "全部图片解码失败：" + "；".join(failures)
                        )

                    report = (
                        f"[{index+1}/{count}] "
                        f"识别{len(sources)}个图片来源，"
                        f"成功输出{succeeded}张。"
                    )
                    if text:
                        report += "\n" + text[:3000]
                    if failures:
                        report += "\n部分失败：" + "；".join(failures)
                    reports.append(report)

                except mm.InterruptProcessingException:
                    raise
                except Exception as error:
                    safe_error = str(error).replace(
                        key, "[密钥已隐藏]"
                    )
                    reports.append(
                        f"[{index+1}/{count}] 失败：{safe_error}"
                    )
                    if not 失败后继续:
                        raise RuntimeError(
                            "\n".join(reports)
                        ) from None
                finally:
                    progress.update(1)

        if not images:
            raise RuntimeError(
                "未生成任何图片\n" + "\n".join(reports)
            )
        if len({image.size for image in images}) > 1:
            reports.append(
                "返回图片尺寸不同，"
                "已居中填充白边组成IMAGE批次，不拉伸、不裁切。"
            )
        reports.append(
            f"总输出：{len(images)}张；未设置返回图片数量上限。"
        )
        return image_batch(images), "\n".join(reports)


NODE_CLASS_MAPPINGS = {
    "UniversalImageAPIEditor": UniversalImageAPINode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalImageAPIEditor": "☁️API 图像生成与编辑",
}
