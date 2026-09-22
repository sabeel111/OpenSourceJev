import os
import sys
from pathlib import Path

import pytest

import app.llama_ctypes as lc
from app.llama_ctypes import (
    _find_library_file,
    _lib_candidates,
    _library_path,
    _runtime_dir,
    _runtime_search_dirs,
    native_status,
)


def test_lib_candidates_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(sys, "platform", "win32")
    candidates = _lib_candidates("llama")
    assert candidates == ["llama.dll", "libllama.dll"]


def test_lib_candidates_linux(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys, "platform", "linux")
    candidates = _lib_candidates("llama")
    assert candidates == ["libllama.so", "llama.so", "llama.dll"]


def test_lib_candidates_darwin(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys, "platform", "darwin")
    candidates = _lib_candidates("llama")
    assert candidates == ["libllama.dylib", "llama.dylib", "libllama.so"]


def test_find_library_file_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(lc, "_lib_candidates", lambda name: [f"lib{name}.so", f"{name}.so"])

    so_file = tmp_path / "libllama.so"
    so_file.touch()

    found = _find_library_file(tmp_path, "llama")
    assert found == so_file

    ggml_so = tmp_path / "libggml.so"
    ggml_so.touch()
    assert _find_library_file(tmp_path, "ggml") == ggml_so


def test_library_path_with_jev_llama_dll(monkeypatch, tmp_path):
    fake_file = tmp_path / "custom_libllama.so"
    fake_file.touch()
    monkeypatch.setenv("JEV_LLAMA_DLL", str(fake_file))
    resolved = _library_path()
    assert resolved == fake_file.resolve()


def test_library_path_with_jev_llama_runtime(monkeypatch, tmp_path):
    monkeypatch.delenv("JEV_LLAMA_DLL", raising=False)
    monkeypatch.setattr(lc, "_lib_candidates", lambda name: [f"lib{name}.so", f"{name}.so", f"{name}.dll"])

    runtime_dir = tmp_path / "lc"
    runtime_dir.mkdir()
    lib_file = runtime_dir / "libllama.so"
    lib_file.touch()

    monkeypatch.setenv("JEV_LLAMA_RUNTIME", str(runtime_dir))

    assert _library_path() == lib_file
    assert _runtime_dir() == runtime_dir


def test_runtime_search_dirs_configured(monkeypatch, tmp_path):
    custom_dir = tmp_path / "custom_runtime"
    monkeypatch.setenv("JEV_LLAMA_RUNTIME", str(custom_dir))

    search_dirs = _runtime_search_dirs()
    assert custom_dir.resolve() in search_dirs
    assert (custom_dir / "lib").resolve() in search_dirs
    assert (custom_dir / "bin").resolve() in search_dirs


def test_native_status_missing_library(monkeypatch, tmp_path):
    missing_file = tmp_path / "libllama.so"
    monkeypatch.setenv("JEV_LLAMA_DLL", str(missing_file))

    status = native_status()
    assert status["available"] is False
    assert status["cuda"] is False
    assert "libllama.so not found" in status["error"]
