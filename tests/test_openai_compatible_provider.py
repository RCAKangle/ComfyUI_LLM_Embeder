import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DummyRoutes:
    def post(self, _path):
        return lambda handler: handler


def _load_chat_nodes():
    fake_aiohttp = types.ModuleType("aiohttp")
    fake_aiohttp.web = types.SimpleNamespace(
        json_response=lambda payload, status=200: (payload, status)
    )

    fake_server = types.ModuleType("server")
    fake_server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=_DummyRoutes())
    )

    module_path = PROJECT_ROOT / "chat_optimize_nodes.py"
    spec = importlib.util.spec_from_file_location(
        "chat_optimize_nodes_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"aiohttp": fake_aiohttp, "server": fake_server}):
        spec.loader.exec_module(module)
    return module


chat_nodes = _load_chat_nodes()


class OpenAICompatibleProviderTests(unittest.TestCase):
    def setUp(self):
        chat_nodes._chat_sessions.clear()
        chat_nodes._chat_session_locks.clear()

    def test_module_loader_restores_dependency_modules(self):
        original_modules = {
            name: sys.modules.get(name) for name in ("aiohttp", "server")
        }

        _load_chat_nodes()

        for name, original_module in original_modules.items():
            with self.subTest(name=name):
                self.assertIs(sys.modules.get(name), original_module)

    def test_llm_config_exposes_openai_compatible_provider(self):
        providers = chat_nodes.LLMConfigNode.INPUT_TYPES()["required"]["provider"][0]

        self.assertIn("openai_compatible", providers)

    def test_openai_compatible_provider_reuses_existing_chat_client(self):
        config = {
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:3762/v1",
            "model_name": "local-model",
            "api_key": "",
        }

        with patch.object(
            chat_nodes, "_call_openai_compatible_chat", return_value="local response"
        ) as compatible_call, patch.object(chat_nodes, "_call_ollama_chat") as ollama_call:
            _, history = chat_nodes.ChatNode().chat(
                model_name="unused",
                base_url="http://127.0.0.1:11434",
                user_message="hello",
                session_id="openai-compatible-test",
                refresh_session=True,
                llm_config=config,
            )

        compatible_call.assert_called_once()
        ollama_call.assert_not_called()
        self.assertEqual(compatible_call.call_args.args[:3], (
            "http://127.0.0.1:3762/v1",
            "local-model",
            "",
        ))
        self.assertIn("Assistant: local response", history)

    def test_provider_error_propagates_without_storing_failed_turn(self):
        session_id = "provider-error-test"
        config = {
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:3762/v1",
            "model_name": "local-model",
            "api_key": "",
        }

        with patch.object(
            chat_nodes,
            "_call_openai_compatible_chat",
            side_effect=RuntimeError("connection refused"),
        ):
            with self.assertRaisesRegex(RuntimeError, "connection refused"):
                chat_nodes.ChatNode().chat(
                    model_name="unused",
                    base_url="http://127.0.0.1:11434",
                    user_message="hello",
                    session_id=session_id,
                    refresh_session=True,
                    llm_config=config,
                )

        self.assertEqual(chat_nodes._chat_sessions[session_id], [])

    def test_same_session_requests_are_serialized(self):
        session_id = "concurrent-session-test"
        config = {
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:3762/v1",
            "model_name": "local-model",
            "api_key": "",
        }
        first_started = threading.Event()
        second_started = threading.Event()
        release_first = threading.Event()
        call_count = 0
        call_count_lock = threading.Lock()
        errors = []

        def fake_chat(_base_url, _model_name, _api_key, messages, **_kwargs):
            nonlocal call_count
            with call_count_lock:
                call_count += 1
                current_call = call_count
            if current_call == 1:
                first_started.set()
                if not release_first.wait(timeout=2):
                    raise TimeoutError("first request was not released")
            else:
                second_started.set()
            return f"{messages[-1]['content']}-reply"

        def run_chat(message):
            try:
                chat_nodes.ChatNode().chat(
                    model_name="unused",
                    base_url="http://127.0.0.1:11434",
                    user_message=message,
                    session_id=session_id,
                    llm_config=config,
                )
            except Exception as exc:
                errors.append(exc)

        with patch.object(chat_nodes, "_call_openai_compatible_chat", side_effect=fake_chat):
            first_thread = threading.Thread(target=run_chat, args=("first",))
            second_thread = threading.Thread(target=run_chat, args=("second",))
            first_thread.start()
            self.assertTrue(first_started.wait(timeout=1))
            second_thread.start()
            second_started_before_release = second_started.wait(timeout=0.2)
            release_first.set()
            first_thread.join(timeout=2)
            second_thread.join(timeout=2)

        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertFalse(second_started_before_release)
        self.assertEqual(
            chat_nodes._chat_sessions[session_id],
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first-reply"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second-reply"},
            ],
        )

    def test_chat_node_has_provider_neutral_display_name(self):
        self.assertEqual(chat_nodes.NODE_DISPLAY_NAME_MAPPINGS["ChatNode"], "Chat (LLM)")

    def test_readme_documents_supported_local_server_examples(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        expected_text = (
            "`provider=openai_compatible`",
            "LM Studio",
            "http://127.0.0.1:1234/v1",
            "llama.cpp",
            "http://127.0.0.1:8080/v1",
            "LMDeploy",
            "http://127.0.0.1:23333/v1",
        )
        for text in expected_text:
            with self.subTest(text=text):
                self.assertIn(text, readme)


if __name__ == "__main__":
    unittest.main()
