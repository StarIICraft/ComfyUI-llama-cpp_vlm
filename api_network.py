import asyncio
import json
from urllib.parse import urlsplit

import requests
from aiohttp import web
from server import PromptServer

import comfy.model_management as mm


def validate_base_url(value):
    value = str(value or "").strip().rstrip("/")
    parsed = urlsplit(value)

    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Base URL 必须是有效的 HTTP/HTTPS 基础地址，"
            "不能包含用户名、密码、查询参数或 fragment。"
        )

    if value.endswith(("/chat/completions", "/models")):
        raise ValueError(
            "请填写基础地址，例如 https://api.openai.com/v1，"
            "不要包含 /chat/completions 或 /models。"
        )

    return value


def response_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "\n".join(
            item["text"]
            for item in value
            if isinstance(item, dict)
            and isinstance(item.get("text"), str)
        )

    return str(value)


def parse_json_response(response, service_name):
    if not response.ok:
        # 不输出响应体，防止服务端回显敏感信息。
        raise RuntimeError(
            f"{service_name} 请求失败，HTTP {response.status_code}。"
            "请检查接口地址、密钥和请求参数。"
        )

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            f"{service_name} 没有返回有效 JSON。"
        ) from None

    if not isinstance(data, dict):
        raise RuntimeError(
            f"{service_name} 返回格式无效，需要 JSON 对象。"
        )

    if data.get("error"):
        raise RuntimeError(
            f"{service_name} 返回了错误，请检查服务端日志或配置。"
        )

    return data


def call_chat_api(config, messages, parameters, stream=False):
    """
    调用 OpenAI 兼容 Chat Completions。

    stream=True:
        接收 SSE，实时打印到后端控制台，
        最后返回完整字符串。
    """
    base_url = validate_base_url(config["base_url"])

    headers = {
        "Content-Type": "application/json",
    }

    if config.get("api_key"):
        headers["Authorization"] = (
            f"Bearer {config['api_key']}"
        )

    payload = dict(parameters)
    payload.update({
        "model": config["model_name"],
        "messages": messages,
        "stream": bool(stream),
    })

    try:
        with requests.post(
            base_url + "/chat/completions",
            headers=headers,
            json=payload,
            stream=bool(stream),
            timeout=(15, int(config.get("timeout", 180))),
        ) as response:
            if not response.ok:
                raise RuntimeError(
                    f"大模型请求失败，HTTP {response.status_code}。"
                    "请检查密钥、模型及参数兼容性。"
                )

            if not stream:
                data = parse_json_response(response, "大模型")
                choices = data.get("choices")

                if not isinstance(choices, list) or not choices:
                    raise RuntimeError("大模型返回结果缺少 choices。")

                message = choices[0].get("message")

                if not isinstance(message, dict):
                    raise RuntimeError("大模型返回结果缺少 message。")

                return (
                    response_text(message.get("content")),
                    response_text(
                        message.get("reasoning_content")
                        or message.get("reasoning")
                    ),
                )

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            # 某些兼容服务忽略 stream，直接返回普通 JSON。
            if "application/json" in content_type:
                data = parse_json_response(response, "大模型")
                choices = data.get("choices") or []

                if not choices:
                    raise RuntimeError("大模型返回结果缺少 choices。")

                message = choices[0].get("message") or {}

                text = response_text(message.get("content"))
                reasoning = response_text(
                    message.get("reasoning_content")
                    or message.get("reasoning")
                )

                if text:
                    print(text, flush=True)

                return text, reasoning

            text_parts = []
            reasoning_parts = []
            received_choice = False
            finished = False

            print("[API LLM] 流式响应开始：", flush=True)

            # 按 SSE 事件解析，支持同一事件包含多行 data。
            event_lines = []

            def consume_event(lines):
                nonlocal received_choice, finished

                if not lines:
                    return

                raw_data = "\n".join(lines).strip()

                if not raw_data:
                    return

                if raw_data == "[DONE]":
                    finished = True
                    return

                try:
                    event = json.loads(raw_data)
                except ValueError:
                    raise RuntimeError(
                        "流式响应包含无效 JSON。"
                    ) from None

                if not isinstance(event, dict):
                    raise RuntimeError("流式响应格式无效。")

                if event.get("error"):
                    raise RuntimeError("大模型在流式生成期间返回错误。")

                choices = event.get("choices") or []

                # 允许只包含 usage 的事件。
                if not choices:
                    return

                choice = next(
                    (
                        item
                        for item in choices
                        if item.get("index", 0) == 0
                    ),
                    None,
                )

                if choice is None:
                    return

                received_choice = True
                delta = choice.get("delta") or {}

                text = response_text(delta.get("content"))
                reasoning = response_text(
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                )

                if reasoning:
                    reasoning_parts.append(reasoning)

                if text:
                    text_parts.append(text)
                    print(text, end="", flush=True)

                if choice.get("finish_reason") is not None:
                    finished = True

            # 显式使用 UTF-8，避免中文流式片段乱码。
            response.encoding = "utf-8"

            for line in response.iter_lines(
                chunk_size=1,
                decode_unicode=True,
            ):
                mm.throw_exception_if_processing_interrupted()

                if line == "":
                    consume_event(event_lines)
                    event_lines = []
                    continue

                if line.startswith("data:"):
                    value = line[5:]

                    if value.startswith(" "):
                        value = value[1:]

                    event_lines.append(value)

                # SSE 注释、event、id、retry 等字段不参与 JSON 解析。

            if event_lines:
                consume_event(event_lines)

            print("", flush=True)

            if not received_choice:
                raise RuntimeError(
                    "没有收到有效的流式 choices 数据，"
                    "请确认服务支持 OpenAI SSE。"
                )

            if not finished:
                raise RuntimeError(
                    "流式连接提前结束，没有收到结束标记。"
                    "为避免把不完整内容当作完整结果，本次请求失败。"
                )

            return (
                "".join(text_parts),
                "".join(reasoning_parts),
            )

    except requests.Timeout:
        raise RuntimeError(
            "大模型请求超时，请检查网络或增大 timeout。"
        ) from None

    except requests.RequestException:
        raise RuntimeError(
            "大模型连接失败或流式传输中断。"
        ) from None


def search_tavily(query, api_key, max_results=5):
    """
    检索网页，返回提供给模型的参考文本。
    不执行网页中的指令，不抓取或运行网页代码。
    """
    query = str(query or "").strip()
    api_key = str(api_key or "").strip()

    if not query:
        raise ValueError("搜索关键词不能为空。")

    if not api_key:
        raise ValueError(
            "启用联网搜索后，需要填写 Tavily API Key。"
        )

    mm.throw_exception_if_processing_interrupted()

    try:
        with requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": max(1, min(10, int(max_results))),
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=(15, 60),
        ) as response:
            data = parse_json_response(response, "Tavily 搜索")

    except requests.Timeout:
        raise RuntimeError("联网搜索超时。") from None

    except requests.RequestException:
        raise RuntimeError("联网搜索连接失败。") from None

    mm.throw_exception_if_processing_interrupted()

    results = data.get("results") or []

    if not isinstance(results, list) or not results:
        raise RuntimeError(
            "搜索没有返回结果，请调整 search_query。"
        )

    blocks = []

    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "")[:500]
        url = str(item.get("url") or "")[:2000]
        content = str(item.get("content") or "")[:4000]

        blocks.append(
            f"[{index}]\n"
            f"标题：{title}\n"
            f"URL：{url}\n"
            f"摘要：{content}"
        )

    if not blocks:
        raise RuntimeError("搜索结果格式无效。")

    return "\n\n".join(blocks)


def fetch_model_names(base_url, api_key):
    base_url = validate_base_url(base_url)

    headers = {}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with requests.get(
            base_url + "/models",
            headers=headers,
            timeout=(10, 45),
        ) as response:
            data = parse_json_response(response, "模型列表")

    except requests.Timeout:
        raise RuntimeError("获取模型列表超时。") from None

    except requests.RequestException:
        raise RuntimeError("获取模型列表连接失败。") from None

    items = data.get("data") or []

    if not isinstance(items, list):
        raise RuntimeError("模型列表格式无效，需要 data 数组。")

    models = sorted({
        item["id"]
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and item["id"].strip()
    })

    if not models:
        raise RuntimeError("接口没有返回可用模型。")

    return models


@PromptServer.instance.routes.post("/mini_api_llm/models")
async def refresh_models_route(request):
    try:
        data = await request.json()

        if not isinstance(data, dict):
            raise ValueError("请求必须是 JSON 对象。")

        base_url = validate_base_url(data.get("base_url", ""))
        api_key = str(data.get("api_key") or "").strip()

        # requests 放在线程中，避免阻塞 aiohttp 事件循环。
        models = await asyncio.to_thread(
            fetch_model_names,
            base_url,
            api_key,
        )

        return web.json_response({"models": models})

    except ValueError as error:
        return web.json_response(
            {"error": str(error)},
            status=400,
        )

    except RuntimeError as error:
        return web.json_response(
            {"error": str(error)},
            status=502,
        )

    except Exception:
        return web.json_response(
            {"error": "获取模型列表失败。"},
            status=500,
        )
