#!/usr/bin/env python3
"""
Test script for nanovllm OpenAI-compatible API server.

Usage:
    1. Start the server:
       python -m nanovllm.server --model /path/to/model --enforce-eager

    2. Run this test (in another terminal):
       python example/test_server.py [--base-url http://localhost:8000]
"""

import argparse
import json
import sys
import time


def test_health(base_url: str) -> bool:
    """Test /health endpoint."""
    import requests
    print("=" * 60)
    print("[Test 1] GET /health")
    print("=" * 60)
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {data}")
        assert data["status"] == "ok"
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_models(base_url: str) -> bool:
    """Test /v1/models endpoint."""
    import requests
    print("\n" + "=" * 60)
    print("[Test 2] GET /v1/models")
    print("=" * 60)
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        print(f"  Status: {resp.status_code}")
        print(f"  Models: {[m['id'] for m in data['data']]}")
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_chat_completions(base_url: str) -> bool:
    """Test POST /v1/chat/completions (non-streaming)."""
    import requests
    print("\n" + "=" * 60)
    print("[Test 3] POST /v1/chat/completions (non-streaming)")
    print("=" * 60)
    try:
        payload = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Say hello in one sentence."}
            ],
            "temperature": 0.7,
            "max_tokens": 64,
            "stream": False,
        }
        print(f"  Request: messages={payload['messages']}")
        t0 = time.time()
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        print(f"  Status: {resp.status_code}")
        print(f"  Response ID: {data['id']}")
        print(f"  Content: {content!r}")
        print(f"  Usage: prompt={usage.get('prompt_tokens', '?')}, "
              f"completion={usage.get('completion_tokens', '?')}, "
              f"total={usage.get('total_tokens', '?')}")
        print(f"  Time: {elapsed:.2f}s")
        assert data["object"] == "chat.completion"
        assert len(content) > 0
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_chat_completions_streaming(base_url: str) -> bool:
    """Test POST /v1/chat/completions (streaming)."""
    import requests
    print("\n" + "=" * 60)
    print("[Test 4] POST /v1/chat/completions (streaming)")
    print("=" * 60)
    try:
        payload = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Count from 1 to 5."}
            ],
            "temperature": 0.7,
            "max_tokens": 64,
            "stream": True,
        }
        print(f"  Request: messages={payload['messages']}")
        t0 = time.time()
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()

        full_content = ""
        chunk_count = 0
        print("  Streaming chunks: ", end="", flush=True)
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    print(" [DONE]")
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_content += content
                        chunk_count += 1
                        print(".", end="", flush=True)
                except json.JSONDecodeError:
                    pass

        elapsed = time.time() - t0
        print(f"\n  Full content: {full_content!r}")
        print(f"  Chunks received: {chunk_count}")
        print(f"  Time: {elapsed:.2f}s")
        assert len(full_content) > 0, "No content received"
        assert chunk_count > 0, "No streaming chunks received"
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_completions(base_url: str) -> bool:
    """Test POST /v1/completions (non-streaming)."""
    import requests
    print("\n" + "=" * 60)
    print("[Test 5] POST /v1/completions (non-streaming)")
    print("=" * 60)
    try:
        payload = {
            "model": "test",
            "prompt": "The capital of France is",
            "temperature": 0.7,
            "max_tokens": 32,
            "stream": False,
        }
        print(f"  Request: prompt={payload['prompt']!r}")
        t0 = time.time()
        resp = requests.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=120,
        )
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["text"]
        usage = data.get("usage", {})
        print(f"  Status: {resp.status_code}")
        print(f"  Text: {text!r}")
        print(f"  Usage: prompt={usage.get('prompt_tokens', '?')}, "
              f"completion={usage.get('completion_tokens', '?')}")
        print(f"  Time: {elapsed:.2f}s")
        assert data["object"] == "text_completion"
        assert len(text) > 0
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_completions_streaming(base_url: str) -> bool:
    """Test POST /v1/completions (streaming)."""
    import requests
    print("\n" + "=" * 60)
    print("[Test 6] POST /v1/completions (streaming)")
    print("=" * 60)
    try:
        payload = {
            "model": "test",
            "prompt": "Once upon a time",
            "temperature": 0.7,
            "max_tokens": 64,
            "stream": True,
        }
        print(f"  Request: prompt={payload['prompt']!r}")
        t0 = time.time()
        resp = requests.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()

        full_text = ""
        chunk_count = 0
        print("  Streaming: ", end="", flush=True)
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    print(" [DONE]")
                    break
                try:
                    chunk = json.loads(data_str)
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        full_text += text
                        chunk_count += 1
                        print(".", end="", flush=True)
                except json.JSONDecodeError:
                    pass

        elapsed = time.time() - t0
        print(f"\n  Full text: {full_text!r}")
        print(f"  Chunks: {chunk_count}")
        print(f"  Time: {elapsed:.2f}s")
        assert len(full_text) > 0
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_openai_sdk(base_url: str) -> bool:
    """Test with OpenAI Python SDK (if installed)."""
    print("\n" + "=" * 60)
    print("[Test 7] OpenAI Python SDK compatibility")
    print("=" * 60)
    try:
        from openai import OpenAI
    except ImportError:
        print("  [SKIP] openai package not installed")
        print("  Install with: pip install openai")
        return True

    try:
        client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy")

        # Non-streaming
        print("  Testing chat (non-streaming)...")
        response = client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "Hi!"}],
            temperature=0.7,
            max_tokens=32,
        )
        content = response.choices[0].message.content
        print(f"    Response: {content!r}")
        assert len(content) > 0

        # Streaming
        print("  Testing chat (streaming)...")
        stream = client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "Say 'test ok'"}],
            temperature=0.7,
            max_tokens=32,
            stream=True,
        )
        full = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full += delta.content
        print(f"    Stream result: {full!r}")
        assert len(full) > 0

        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test nanovllm API server")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Server base URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    print(f"Testing nanovllm API server at: {base_url}")
    print()

    # Check server is running
    import requests
    try:
        requests.get(f"{base_url}/health", timeout=3)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to server at {base_url}")
        print(f"Please start the server first:")
        print(f"  python -m nanovllm.server --model /path/to/model --enforce-eager")
        sys.exit(1)

    results = []
    results.append(("Health check", test_health(base_url)))
    results.append(("List models", test_models(base_url)))
    results.append(("Chat completions", test_chat_completions(base_url)))
    results.append(("Chat streaming", test_chat_completions_streaming(base_url)))
    results.append(("Text completions", test_completions(base_url)))
    results.append(("Text streaming", test_completions_streaming(base_url)))
    results.append(("OpenAI SDK", test_openai_sdk(base_url)))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
