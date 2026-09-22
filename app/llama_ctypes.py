"""Small direct ctypes adapter for the official prebuilt llama.cpp runtime.

This intentionally bypasses Ollama and llama-cpp-python.  The adapter loads
the exported llama.cpp C API from runtime shared libraries (llama.dll / libllama.so / libllama.dylib) and reads
the actual float32 logits returned by ``llama_get_logits_ith``.
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import sys
from pathlib import Path
from threading import RLock
from typing import Iterable, List, Optional, Sequence, Set


class _LlamaModelParams(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("load_mode", ctypes.c_int32),
        ("lazy_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]


class _LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_outputs_max_per_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]


class _LlamaBatch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(ctypes.c_int32)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


_LIBRARY: Optional[ctypes.CDLL] = None
_BACKEND_LIBRARY: Optional[ctypes.CDLL] = None
_DLL_HANDLES: List[object] = []
_LIBRARY_LOCK = RLock()
_BACKEND_INITIALIZED = False


def _lib_candidates(base_name: str) -> List[str]:
    """Return platform-appropriate library file name candidates in order of preference."""
    if os.name == "nt":
        return [f"{base_name}.dll", f"lib{base_name}.dll"]
    if sys.platform == "darwin":
        return [f"lib{base_name}.dylib", f"{base_name}.dylib", f"lib{base_name}.so"]
    return [f"lib{base_name}.so", f"{base_name}.so", f"{base_name}.dll"]


def _find_library_file(directory: Path, base_name: str) -> Optional[Path]:
    """Find a library file in directory matching platform candidates, or None if not found."""
    if not directory.is_dir():
        return None
    for candidate in _lib_candidates(base_name):
        candidate_path = directory / candidate
        if candidate_path.is_file():
            return candidate_path
    return None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _runtime_search_dirs() -> List[Path]:
    """Return search directories for llama.cpp runtime libraries."""
    configured = os.getenv("JEV_LLAMA_RUNTIME")
    if configured:
        base = Path(configured).expanduser().resolve()
        return [base, base / "bin", base / "lib", base / "lib64"]
    root = _project_root() / "runtime" / "llama.cpp"
    return [root / "bin", root / "lib", root / "lib64", root]


def _runtime_dir() -> Path:
    configured = os.getenv("JEV_LLAMA_RUNTIME")
    if configured:
        return Path(configured).expanduser().resolve()
    for search_dir in _runtime_search_dirs():
        if _find_library_file(search_dir, "llama") is not None:
            return search_dir
    if os.name == "nt":
        return _project_root() / "runtime" / "llama.cpp" / "bin"
    return _project_root() / "runtime" / "llama.cpp" / "lib"


def _library_path() -> Path:
    configured = os.getenv("JEV_LLAMA_DLL")
    if configured:
        return Path(configured).expanduser().resolve()
    for search_dir in _runtime_search_dirs():
        found = _find_library_file(search_dir, "llama")
        if found is not None:
            return found
    runtime_dir = _runtime_dir()
    preferred_name = _lib_candidates("llama")[0]
    return runtime_dir / preferred_name


def _configure_library(lib: ctypes.CDLL) -> None:
    lib.llama_version.restype = ctypes.c_char_p
    lib.llama_backend_init.argtypes = []
    lib.llama_backend_init.restype = None
    lib.llama_print_system_info.argtypes = []
    lib.llama_print_system_info.restype = ctypes.c_char_p
    lib.llama_supports_gpu_offload.argtypes = []
    lib.llama_supports_gpu_offload.restype = ctypes.c_bool

    lib.llama_model_default_params.argtypes = []
    lib.llama_model_default_params.restype = _LlamaModelParams
    lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, _LlamaModelParams]
    lib.llama_model_load_from_file.restype = ctypes.c_void_p
    lib.llama_model_free.argtypes = [ctypes.c_void_p]
    lib.llama_model_free.restype = None
    lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    lib.llama_model_get_vocab.restype = ctypes.c_void_p
    lib.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_n_tokens.restype = ctypes.c_int32

    lib.llama_context_default_params.argtypes = []
    lib.llama_context_default_params.restype = _LlamaContextParams
    lib.llama_init_from_model.argtypes = [ctypes.c_void_p, _LlamaContextParams]
    lib.llama_init_from_model.restype = ctypes.c_void_p
    lib.llama_free.argtypes = [ctypes.c_void_p]
    lib.llama_free.restype = None
    lib.llama_get_memory.argtypes = [ctypes.c_void_p]
    lib.llama_get_memory.restype = ctypes.c_void_p
    lib.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    lib.llama_memory_clear.restype = None
    lib.llama_memory_seq_rm.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.llama_memory_seq_rm.restype = ctypes.c_bool

    lib.llama_tokenize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_bool,
        ctypes.c_bool,
    ]
    lib.llama_tokenize.restype = ctypes.c_int32
    lib.llama_token_to_piece.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_bool,
    ]
    lib.llama_token_to_piece.restype = ctypes.c_int32

    lib.llama_batch_get_one.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    lib.llama_batch_get_one.restype = _LlamaBatch
    lib.llama_decode.argtypes = [ctypes.c_void_p, _LlamaBatch]
    lib.llama_decode.restype = ctypes.c_int32
    lib.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)


def _load_library() -> ctypes.CDLL:
    global _LIBRARY, _BACKEND_LIBRARY, _BACKEND_INITIALIZED
    with _LIBRARY_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY

        path = _library_path()
        if not path.is_file():
            preferred_name = path.name if path.name else _lib_candidates("llama")[0]
            raise RuntimeError(
                f"{preferred_name} was not found at {path}. Download the official prebuilt runtime "
                "or set JEV_LLAMA_DLL."
            )

        runtime_dir = path.parent
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(str(runtime_dir)))
            cuda_path = os.getenv("CUDA_PATH")
            if cuda_path and Path(cuda_path, "bin").is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(str(Path(cuda_path, "bin"))))

        cdll_kwargs = {"mode": ctypes.RTLD_GLOBAL} if hasattr(ctypes, "RTLD_GLOBAL") else {}

        # Resolve backend (ggml) library
        backend_file: Optional[Path] = None
        configured_ggml = os.getenv("JEV_GGML_DLL")
        if configured_ggml:
            configured_path = Path(configured_ggml).expanduser().resolve()
            if configured_path.is_file():
                backend_file = configured_path

        if backend_file is None:
            backend_file = _find_library_file(runtime_dir, "ggml")

        if backend_file is None:
            for search_dir in _runtime_search_dirs():
                found = _find_library_file(search_dir, "ggml")
                if found is not None:
                    backend_file = found
                    break

        try:
            backend: Optional[ctypes.CDLL] = None
            if backend_file is not None:
                backend = ctypes.CDLL(str(backend_file), **cdll_kwargs)
            else:
                for candidate in _lib_candidates("ggml"):
                    try:
                        backend = ctypes.CDLL(candidate, **cdll_kwargs)
                        break
                    except OSError:
                        pass

            if backend is not None:
                if hasattr(backend, "ggml_backend_load_all"):
                    backend.ggml_backend_load_all.argtypes = []
                    backend.ggml_backend_load_all.restype = None
                if hasattr(backend, "ggml_backend_load_all_from_path"):
                    backend.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
                    backend.ggml_backend_load_all_from_path.restype = None
                    backend.ggml_backend_load_all_from_path(str(runtime_dir).encode("utf-8"))

            lib = ctypes.CDLL(str(path), **cdll_kwargs)
            _configure_library(lib)
            lib.llama_backend_init()
        except OSError as exc:
            raise RuntimeError(f"Could not load the prebuilt llama.cpp runtime: {exc}") from exc

        _LIBRARY = lib
        _BACKEND_LIBRARY = backend
        _BACKEND_INITIALIZED = True
        return lib


def native_status() -> dict:
    """Return runtime status without loading the GGUF weights."""
    path = _library_path()
    if not path.is_file():
        preferred_name = path.name if path.name else _lib_candidates("llama")[0]
        return {"available": False, "cuda": False, "library": str(path), "error": f"{preferred_name} not found"}
    try:
        lib = _load_library()
        info = (lib.llama_print_system_info() or b"").decode("utf-8", errors="replace")

        devices = []
        try:
            runtime_dir = path.parent
            cdll_kwargs = {"mode": ctypes.RTLD_GLOBAL} if hasattr(ctypes, "RTLD_GLOBAL") else {}

            base_file = _find_library_file(runtime_dir, "ggml-base")
            base_lib: Optional[ctypes.CDLL] = None
            if base_file is not None:
                try:
                    base_lib = ctypes.CDLL(str(base_file), **cdll_kwargs)
                except OSError:
                    base_lib = None

            backend_lib = _BACKEND_LIBRARY
            if backend_lib is None:
                backend_file = _find_library_file(runtime_dir, "ggml")
                if backend_file is not None:
                    try:
                        backend_lib = ctypes.CDLL(str(backend_file), **cdll_kwargs)
                    except OSError:
                        backend_lib = None

            lookup_libs = [l for l in (base_lib, backend_lib, lib) if l is not None]

            dev_count_fn = None
            dev_get_fn = None
            dev_name_fn = None
            dev_desc_fn = None

            for candidate_lib in lookup_libs:
                if dev_count_fn is None and hasattr(candidate_lib, "ggml_backend_dev_count"):
                    dev_count_fn = candidate_lib.ggml_backend_dev_count
                    dev_count_fn.restype = ctypes.c_size_t
                if dev_get_fn is None and hasattr(candidate_lib, "ggml_backend_dev_get"):
                    dev_get_fn = candidate_lib.ggml_backend_dev_get
                    dev_get_fn.argtypes = [ctypes.c_size_t]
                    dev_get_fn.restype = ctypes.c_void_p
                if dev_name_fn is None and hasattr(candidate_lib, "ggml_backend_dev_name"):
                    dev_name_fn = candidate_lib.ggml_backend_dev_name
                    dev_name_fn.argtypes = [ctypes.c_void_p]
                    dev_name_fn.restype = ctypes.c_char_p
                if dev_desc_fn is None and hasattr(candidate_lib, "ggml_backend_dev_description"):
                    dev_desc_fn = candidate_lib.ggml_backend_dev_description
                    dev_desc_fn.argtypes = [ctypes.c_void_p]
                    dev_desc_fn.restype = ctypes.c_char_p

            if dev_count_fn and dev_get_fn:
                count = int(dev_count_fn())
                for i in range(count):
                    dev = dev_get_fn(i)
                    if dev:
                        name = ""
                        desc = ""
                        if dev_name_fn:
                            name = (dev_name_fn(dev) or b"").decode("utf-8", errors="replace")
                        if dev_desc_fn:
                            desc = (dev_desc_fn(dev) or b"").decode("utf-8", errors="replace").strip()
                        devices.append({"name": name, "description": desc})
        except Exception:
            pass

        return {
            "available": True,
            "cuda": bool(lib.llama_supports_gpu_offload()),
            "library": str(path),
            "version": (lib.llama_version() or b"").decode("utf-8", errors="replace"),
            "system_info": info,
            "devices": devices,
        }
    except (OSError, RuntimeError) as exc:
        return {"available": False, "cuda": False, "library": str(path), "error": str(exc)}


class CtypesLlama:
    """One native llama.cpp model/context pair with direct logits access."""

    def __init__(self, model_path: str, n_ctx: int, n_batch: int, n_threads: int, n_gpu_layers: int):
        self._lib = _load_library()
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.n_batch = max(1, int(n_batch))
        self._model = None
        self._ctx = None
        self._vocab = None

        model_params = self._lib.llama_model_default_params()
        model_params.n_gpu_layers = int(n_gpu_layers)
        self._model = self._lib.llama_model_load_from_file(
            self.model_path.encode("utf-8"), model_params
        )
        if not self._model:
            raise RuntimeError(f"llama.cpp could not load GGUF model '{self.model_path}'.")

        self._vocab = self._lib.llama_model_get_vocab(self._model)
        self.n_vocab = int(self._lib.llama_vocab_n_tokens(self._vocab))
        if self.n_vocab <= 0:
            self.close()
            raise RuntimeError("llama.cpp returned an invalid vocabulary size.")

        context_params = self._lib.llama_context_default_params()
        context_params.n_ctx = max(256, int(n_ctx))
        context_params.n_batch = self.n_batch
        context_params.n_ubatch = min(self.n_batch, 512)
        context_params.n_seq_max = 1
        context_params.n_threads = max(1, int(n_threads))
        context_params.n_threads_batch = max(1, int(n_threads))
        self._ctx = self._lib.llama_init_from_model(self._model, context_params)
        if not self._ctx:
            self.close()
            raise RuntimeError("llama.cpp could not create a native context.")
        self._memory = self._lib.llama_get_memory(self._ctx)
        self._stop_tokens = self._find_stop_tokens()

    def close(self) -> None:
        if self._ctx:
            self._lib.llama_free(self._ctx)
            self._ctx = None
        if self._model:
            self._lib.llama_model_free(self._model)
            self._model = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def tokenize(self, text: str, add_special: bool = False, parse_special: bool = False) -> List[int]:
        data = text.encode("utf-8")
        capacity = max(16, len(data) + 8)
        while True:
            tokens = (ctypes.c_int32 * capacity)()
            count = int(self._lib.llama_tokenize(
                self._vocab, data, len(data), tokens, capacity, add_special, parse_special
            ))
            if count >= 0:
                return [int(tokens[index]) for index in range(count)]
            required = -count
            if required <= capacity:
                raise RuntimeError(f"llama.cpp tokenizer failed for {text!r}.")
            capacity = required

    def reset(self) -> None:
        self._lib.llama_memory_clear(self._memory, True)

    def trim_kv(self, pos: int) -> bool:
        """Prune tokens in sequence 0 from index pos onwards, preserving previous tokens."""
        if not self._memory:
            return False
        return bool(self._lib.llama_memory_seq_rm(self._memory, 0, int(pos), -1))

    def supports_trim(self) -> bool:
        return True

    def eval(self, tokens: Sequence[int]) -> None:
        if not tokens:
            raise RuntimeError("llama.cpp cannot evaluate an empty token sequence.")
        for offset in range(0, len(tokens), self.n_batch):
            chunk = [int(token) for token in tokens[offset:offset + self.n_batch]]
            token_array = (ctypes.c_int32 * len(chunk))(*chunk)
            batch = self._lib.llama_batch_get_one(token_array, len(chunk))
            result = int(self._lib.llama_decode(self._ctx, batch))
            if result != 0:
                raise RuntimeError(f"llama_decode failed with return code {result}.")

    def last_logits(self) -> List[float]:
        pointer = self._lib.llama_get_logits_ith(self._ctx, -1)
        if not pointer:
            raise RuntimeError("llama.cpp returned no logits for the last decoded token.")
        values = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_float * self.n_vocab)).contents
        return [float(value) for value in values]

    def token_piece(self, token: int) -> str:
        capacity = 256
        while True:
            buffer = ctypes.create_string_buffer(capacity)
            count = int(self._lib.llama_token_to_piece(
                self._vocab, int(token), buffer, capacity, 0, False
            ))
            if count >= 0:
                return bytes(buffer.raw[:count]).decode("utf-8", errors="replace")
            capacity = -count

    def _find_stop_tokens(self) -> Set[int]:
        result: Set[int] = set()
        for marker in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
            try:
                result.update(self.tokenize(marker, add_special=False, parse_special=True))
            except Exception:
                continue
        return result

    def generate(self, prefix: str, temperature: float, max_tokens: int = 128) -> str:
        self.reset()
        self.eval(self.tokenize(prefix, add_special=True, parse_special=True))
        pieces: List[str] = []
        for _ in range(max_tokens):
            logits = self.last_logits()
            if temperature <= 0.01:
                token = max(range(len(logits)), key=logits.__getitem__)
            else:
                scaled = [value / temperature for value in logits]
                peak = max(scaled)
                weights = [math.exp(value - peak) for value in scaled]
                token = random.choices(range(len(weights)), weights=weights, k=1)[0]
            if token in self._stop_tokens:
                break
            pieces.append(self.token_piece(token))
            self.eval([token])
        return "".join(pieces)
