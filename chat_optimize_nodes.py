import asyncio
import json
import requests
from typing import Dict, List, Optional
from aiohttp import web
from server import PromptServer

# Simple in-memory stores for chat state.
_chat_sessions: Dict[str, List[Dict[str, str]]] = {}

REQUEST_TIMEOUT = 60

def _build_headers() -> Dict[str, str]:
    return {"Content-Type": "application/json"}

def _clean_llm_output(text: str) -> str:
    """
    Strictly clean the LLM output to ensure only the raw prompt remains.
    Removes Markdown code blocks, JSON wrapping if present, and extra whitespace.
    """
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def _call_ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    seed: Optional[int] = None,
    other_options: Optional[Dict] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    
    options = {}
    if seed is not None:
        options["seed"] = seed
    if other_options:
        options.update(other_options)
    
    if options:
        payload["options"] = options

    resp = requests.post(url, headers=_build_headers(), data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")

def _build_hf_prompt(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.capitalize()}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines).strip()

def _call_hf_inference(
    model: str,
    token: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    api_url: Optional[str] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    url = api_url or f"https://api-inference.huggingface.co/models/{model}"
    headers = _build_headers()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    prompt = _build_hf_prompt(messages)
    parameters: Dict[str, object] = {}
    if temperature is not None:
        parameters["temperature"] = temperature
    if top_p is not None:
        parameters["top_p"] = top_p
    if max_new_tokens is not None:
        parameters["max_new_tokens"] = int(max_new_tokens)

    payload: Dict[str, object] = {"inputs": prompt}
    if parameters:
        payload["parameters"] = parameters

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data.get("error"))

    item = data
    if isinstance(data, list) and data:
        item = data[0]

    generated_text = ""
    if isinstance(item, dict):
        generated_text = item.get("generated_text", "")
    elif isinstance(item, str):
        generated_text = item

    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]

    return generated_text.strip()

def _reset_history(session_id: str, system_prompt: str) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})
    _chat_sessions[session_id] = history
    return history

class ChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "llama3"}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "user_message": ("STRING", {"multiline": True, "default": ""}),
                "action": (["send", "regenerate", "clear", "deliver_to_optimizer"], {"default": "send"}),
            },
            "optional": {
                "session_id": ("STRING", {"default": "default"}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "refresh_session": ("BOOLEAN", {"default": False}),
                "auto_clear_input": ("BOOLEAN", {"default": True}),
                "llm_config": ("LLM_CONFIG",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("assistant_response", "readable_history")
    FUNCTION = "chat"
    CATEGORY = "ChatOptimize"
    OUTPUT_NODE = False

    def chat(
        self,
        model_name: str,
        base_url: str,
        user_message: str,
        action: str = "send",
        session_id: str = "default",
        system_prompt: str = "",
        refresh_session: bool = False,
        llm_config: Optional[Dict] = None,
    ):
        provider = "ollama"
        hf_token = ""
        hf_api_url = ""
        temperature = None
        top_p = None
        max_new_tokens = None
        if llm_config:
            provider = llm_config.get("provider", provider)
            model_name = llm_config.get("model_name", model_name)
            base_url = llm_config.get("base_url", base_url)
            hf_token = llm_config.get("hf_token", "")
            hf_api_url = llm_config.get("hf_api_url", "")
            if "temperature" in llm_config:
                temperature = llm_config["temperature"]
            if "top_p" in llm_config:
                top_p = llm_config["top_p"]
            if "max_new_tokens" in llm_config:
                max_new_tokens = llm_config["max_new_tokens"]

        if refresh_session or session_id not in _chat_sessions:
            history = _reset_history(session_id, system_prompt)
        else:
            history = _chat_sessions.get(session_id, [])

        if action == "clear":
            history = _reset_history(session_id, system_prompt)
            return ("", "")

        if system_prompt:
            if not history or history[0].get("role") != "system":
                history = _reset_history(session_id, system_prompt)

        messages = list(history)

        if action == "send":
            if user_message.strip():
                messages.append({"role": "user", "content": user_message})
        elif action == "regenerate":
            for idx in range(len(messages) - 1, -1, -1):
                if messages[idx].get("role") == "assistant":
                    messages = messages[:idx]
                    break
        elif action == "deliver_to_optimizer":
            latest_response = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    latest_response = msg.get("content", "")
                    break
            readable_history = ""
            for msg in messages:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                readable_history += f"{role}: {content}\n\n"
            return (latest_response, readable_history)

        response_text = ""
        if messages and messages[-1].get("role") == "user":
            try:
                if provider == "huggingface":
                    response_text = _call_hf_inference(
                        model_name,
                        hf_token,
                        messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        api_url=hf_api_url or None,
                    )
                else:
                    options = {}
                    if temperature is not None:
                        options["temperature"] = temperature
                    if top_p is not None:
                        options["top_p"] = top_p
                    if max_new_tokens is not None:
                        options["num_predict"] = max_new_tokens
                    response_text = _call_ollama_chat(base_url, model_name, messages, other_options=options)
                messages.append({"role": "assistant", "content": response_text})
            except Exception as exc:
                response_text = f"[chat error] {exc}"

        _chat_sessions[session_id] = messages
        
        readable_history = ""
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            readable_history += f"{role}: {content}\n\n"

        return ("", readable_history)

class LLMConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (["ollama", "huggingface"], {"default": "ollama"}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model_name": ("STRING", {"default": "llama3"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 8192, "step": 1}),
                "hf_token": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LLM_CONFIG",)
    RETURN_NAMES = ("llm_config",)
    FUNCTION = "config"
    CATEGORY = "ChatOptimize"

    def config(self, provider, base_url, model_name, temperature, top_p, max_new_tokens, hf_token):
        return (
            {
                "provider": provider,
                "base_url": base_url,
                "model_name": model_name,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "hf_token": hf_token,
            },
        )

class ChatHistoryViewer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "history": ("STRING", {"default": "", "multiline": True}),
            }
        }

    INPUT_IS_LIST = False
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("history",)
    FUNCTION = "show"
    CATEGORY = "ChatOptimize"
    OUTPUT_NODE = True

    def show(self, history=""):
        return {"ui": {"text": [history]}, "result": (history,)}

def _chat_action(data: Dict[str, object]):
    node = ChatNode()
    return node.chat(
        model_name=data.get("model_name", "llama3"),
        base_url=data.get("base_url", "http://127.0.0.1:11434"),
        user_message=data.get("user_message", ""),
        action=data.get("action", "send"),
        session_id=data.get("session_id", "default"),
        system_prompt=data.get("system_prompt", ""),
        refresh_session=bool(data.get("refresh_session", False)),
        llm_config=data.get("llm_config"),
    )

@PromptServer.instance.routes.post("/chat_optimize/chat")
async def chat_endpoint(request):
    payload = await request.json()
    loop = asyncio.get_event_loop()
    try:
        assistant_response, readable_history = await loop.run_in_executor(None, _chat_action, payload)
        return web.json_response(
            {
                "assistant_response": assistant_response,
                "readable_history": readable_history,
            }
        )
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)

NODE_CLASS_MAPPINGS = {
    "ChatNode": ChatNode,
    "LLMConfigNode": LLMConfigNode,
    "ChatHistoryViewer": ChatHistoryViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatNode": "Chat (Ollama)",
    "LLMConfigNode": "LLM Config",
    "ChatHistoryViewer": "Chat History Viewer",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
