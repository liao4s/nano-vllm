"""
OpenAI-compatible API server for nanovllm.

Usage:
    python -m nanovllm.server --model /path/to/model [--host 0.0.0.0] [--port 8000]

Endpoints:
    POST /v1/chat/completions   — Chat completions (streaming & non-streaming)
    POST /v1/completions        — Text completions (streaming & non-streaming)
    GET  /v1/models             — List available models
    GET  /health                — Health check
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


# ============================================================
# Pydantic models — OpenAI-compatible request/response
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage] = Field(default_factory=list)
    temperature: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    top_p: float = 1.0
    n: int = 1
    stop: Optional[Union[List[str], str]] = None

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: Union[str, List[str]] = ""
    temperature: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    top_p: float = 1.0
    n: int = 1
    stop: Optional[Union[List[str], str]] = None

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo

class ChatCompletionStreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionStreamDelta
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]

class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Optional[str] = "stop"

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo

class CompletionStreamChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Optional[str] = None

class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionStreamChoice]


# ============================================================
# Request tracking
# ============================================================

@dataclass
class PendingRequest:
    request_id: str
    seq_id: int
    prompt_tokens: int
    loop: asyncio.AbstractEventLoop
    # For non-streaming: resolved when generation completes
    future: asyncio.Future
    # For streaming: tokens pushed here incrementally
    token_queue: Optional[asyncio.Queue] = None
    stream: bool = False


# ============================================================
# AsyncEngineWrapper — bridges sync engine with async server
# ============================================================

class AsyncEngineWrapper:
    """Wraps LLMEngine and runs the step loop in a background thread."""

    def __init__(self, model_path: str, **engine_kwargs):
        self.engine = LLMEngine(model_path, **engine_kwargs)
        self.tokenizer = self.engine.tokenizer
        self.model_name = model_path.rstrip("/").split("/")[-1]

        # seq_id -> PendingRequest
        self._pending: dict[int, PendingRequest] = {}
        self._lock = threading.Lock()

        # Track token counts for streaming incremental decode
        self._seq_prev_tokens: dict[int, int] = {}

        # Background engine loop
        self._running = True
        self._has_work = threading.Event()
        self._thread = threading.Thread(target=self._engine_loop, daemon=True)
        self._thread.start()

    def shutdown(self):
        self._running = False
        self._has_work.set()
        self._thread.join(timeout=10)

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        stream: bool = False,
    ) -> PendingRequest:
        """Add a request to the engine. Returns a PendingRequest with future/queue."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Tokenize if needed
        if isinstance(prompt, str):
            prompt_ids = self.tokenizer.encode(prompt)
        else:
            prompt_ids = prompt

        # Add to engine (creates Sequence internally)
        self.engine.add_request(prompt_ids, sampling_params)

        # The seq_id of the just-added sequence is the last one in waiting queue
        seq = self.engine.scheduler.waiting[-1]
        seq_id = seq.seq_id

        token_queue = asyncio.Queue() if stream else None

        pending = PendingRequest(
            request_id=request_id,
            seq_id=seq_id,
            prompt_tokens=len(prompt_ids),
            loop=loop,
            future=future,
            token_queue=token_queue,
            stream=stream,
        )

        with self._lock:
            self._pending[seq_id] = pending
            self._seq_prev_tokens[seq_id] = 0

        # Wake up the engine loop
        self._has_work.set()
        return pending

    def _engine_loop(self):
        """Background thread: continuously steps the engine."""
        while self._running:
            # Wait until there's work
            self._has_work.wait(timeout=0.05)

            if self.engine.is_finished():
                self._has_work.clear()
                continue

            try:
                outputs, _ = self.engine.step()
            except Exception as e:
                # Log the error for debugging
                import traceback
                print(f"[engine_loop] ERROR in step(): {e}")
                traceback.print_exc()
                # Resolve all pending futures with the error
                with self._lock:
                    for pending in self._pending.values():
                        if not pending.future.done():
                            pending.loop.call_soon_threadsafe(
                                pending.future.set_exception, e
                            )
                        # Also send sentinel to streaming queues to unblock clients
                        if pending.stream and pending.token_queue is not None:
                            pending.loop.call_soon_threadsafe(
                                pending.token_queue.put_nowait, None
                            )
                    self._pending.clear()
                    self._seq_prev_tokens.clear()
                continue

            # Check for streaming updates: push incremental tokens
            with self._lock:
                # For streaming requests, check all running sequences for new tokens
                for seq in list(self.engine.scheduler.running):
                    sid = seq.seq_id
                    pending = self._pending.get(sid)
                    if pending is None or not pending.stream:
                        continue
                    cur_count = seq.num_completion_tokens
                    prev_count = self._seq_prev_tokens.get(sid, 0)
                    if cur_count > prev_count:
                        new_ids = seq.completion_token_ids[prev_count:cur_count]
                        self._seq_prev_tokens[sid] = cur_count
                        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                        if text:
                            pending.loop.call_soon_threadsafe(
                                pending.token_queue.put_nowait, text
                            )

            # Process completed sequences
            for seq_id, token_ids in outputs:
                with self._lock:
                    pending = self._pending.pop(seq_id, None)
                    self._seq_prev_tokens.pop(seq_id, None)
                if pending is None:
                    continue

                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                result = {
                    "text": text,
                    "token_ids": token_ids,
                    "prompt_tokens": pending.prompt_tokens,
                    "completion_tokens": len(token_ids),
                }

                if pending.stream:
                    # Push remaining tokens + sentinel
                    prev_count = 0  # already pushed incrementally
                    pending.loop.call_soon_threadsafe(
                        pending.token_queue.put_nowait, None  # sentinel
                    )

                # Resolve the future
                if not pending.future.done():
                    pending.loop.call_soon_threadsafe(
                        pending.future.set_result, result
                    )

            # Check if there's still work
            if self.engine.is_finished():
                self._has_work.clear()


# ============================================================
# FastAPI application
# ============================================================

def create_app(engine: AsyncEngineWrapper) -> FastAPI:
    app = FastAPI(title="nanovllm API Server", version="0.1.0")

    # ----------------------------------------------------------
    # Health check
    # ----------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ----------------------------------------------------------
    # List models
    # ----------------------------------------------------------
    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "nanovllm",
                }
            ],
        }

    # ----------------------------------------------------------
    # Chat completions
    # ----------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        # Apply chat template
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        try:
            prompt = engine.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback: simple concatenation
            prompt = "".join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                for m in messages
            ) + "<|im_start|>assistant\n"

        sampling_params = SamplingParams(
            temperature=max(request.temperature, 0.01),
            max_tokens=request.max_tokens,
        )

        pending = engine.add_request(prompt, sampling_params, stream=request.stream)

        if request.stream:
            return StreamingResponse(
                _stream_chat_response(pending, engine.model_name),
                media_type="text/event-stream",
            )
        else:
            result = await pending.future
            response = ChatCompletionResponse(
                id=pending.request_id,
                created=int(time.time()),
                model=engine.model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=result["text"]),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    total_tokens=result["prompt_tokens"] + result["completion_tokens"],
                ),
            )
            return response

    async def _stream_chat_response(
        pending: PendingRequest, model_name: str
    ) -> AsyncGenerator[str, None]:
        request_id = pending.request_id
        created = int(time.time())

        # First chunk: role
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionStreamDelta(role="assistant", content=""),
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        # Content chunks
        while True:
            token_text = await pending.token_queue.get()
            if token_text is None:  # sentinel: generation done
                break
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=model_name,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatCompletionStreamDelta(content=token_text),
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk: finish_reason
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionStreamDelta(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    # ----------------------------------------------------------
    # Text completions
    # ----------------------------------------------------------
    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        prompt = request.prompt
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""

        sampling_params = SamplingParams(
            temperature=max(request.temperature, 0.01),
            max_tokens=request.max_tokens,
        )

        pending = engine.add_request(prompt, sampling_params, stream=request.stream)

        if request.stream:
            return StreamingResponse(
                _stream_completion_response(pending, engine.model_name),
                media_type="text/event-stream",
            )
        else:
            result = await pending.future
            response = CompletionResponse(
                id=pending.request_id,
                created=int(time.time()),
                model=engine.model_name,
                choices=[
                    CompletionChoice(
                        index=0,
                        text=result["text"],
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    total_tokens=result["prompt_tokens"] + result["completion_tokens"],
                ),
            )
            return response

    async def _stream_completion_response(
        pending: PendingRequest, model_name: str
    ) -> AsyncGenerator[str, None]:
        request_id = pending.request_id
        created = int(time.time())

        while True:
            token_text = await pending.token_queue.get()
            if token_text is None:
                break
            chunk = CompletionStreamResponse(
                id=request_id,
                created=created,
                model=model_name,
                choices=[
                    CompletionStreamChoice(index=0, text=token_text)
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk
        chunk = CompletionStreamResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                CompletionStreamChoice(index=0, text="", finish_reason="stop")
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return app


# ============================================================
# CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="nanovllm OpenAI-compatible API server")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs")
    args = parser.parse_args()

    print(f"Starting nanovllm API server...")
    print(f"  Model: {args.model}")
    print(f"  Host:  {args.host}:{args.port}")
    print(f"  TP:    {args.tensor_parallel_size}")

    engine_kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }

    engine = AsyncEngineWrapper(args.model, **engine_kwargs)
    app = create_app(engine)

    print(f"Server ready at http://{args.host}:{args.port}")
    print(f"  POST /v1/chat/completions")
    print(f"  POST /v1/completions")
    print(f"  GET  /v1/models")
    print(f"  GET  /health")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
