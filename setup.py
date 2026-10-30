"""FuseMoeMLA build: plain-nvcc shared library, no torch headers.

The MLA decode kernel (csrc/mla.cu) is pure CUDA + extern "C" and is loaded
through ctypes at runtime, so it builds and works regardless of the installed
torch version (the same trick the dense FusionAttention plugin uses on sm_61).

    python setup.py build_py --build-lib .buildlib

Env knobs:
    FISSION_ATTN_CUDA_ARCHS   ';'-separated arch list (legacy FISSION_ATTN_*)
                            (default "6.1;7.5;8.0;8.6;8.9;9.0")
    FISSION_ATTN_NVCC         nvcc binary (default: nvcc on PATH)
    FISSION_ATTN_CCBIN        host compiler for -ccbin (default g++-13 if present)
    FISSION_ATTN_SKIP_BUILD=1   python-only install (torch fallback decode path)
"""

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "csrc" / "mla.cu"
SHIM = ROOT / "csrc" / "include_shim"
DEFAULT_ARCHS = "6.1;7.5;8.0;8.6;8.9;9.0"



def _env(name, default=""):
    return os.environ.get(name, default)
#
    return os.environ.get("FISSION_ATTN_" + name[len("FISSION_ATTN_"):]) \
        or os.environ.get(name, default)


class NativeBuildPy(build_py):
    def run(self):
        super().run()
        if _env("FISSION_ATTN_SKIP_BUILD") == "1":
            self.announce("fissionattn: native build skipped (FISSION_ATTN_SKIP_BUILD=1)")
            return
        libdir = Path(self.build_lib) / "fissionattn"
        libdir.mkdir(parents=True, exist_ok=True)
        ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        self._build_kernels(libdir / f"_mla{ext}")

    def _build_kernels(self, out: Path):
        nvcc = _env("FISSION_ATTN_NVCC", "nvcc")
        archs = _env("FISSION_ATTN_CUDA_ARCHS", DEFAULT_ARCHS)
        ccbin = _env("FISSION_ATTN_CCBIN", "")
        if not ccbin and shutil.which("g++-13"):
            ccbin = "g++-13"  # nvcc <= 12.9 rejects gcc-15 hosts
        cmd = [nvcc, "-std=c++17", "-O3", "-shared", "-Xcompiler", "-fPIC",
               "-isystem", str(SHIM)]
        for a in archs.split(";"):
            a = a.strip()
            if a:
                num = a.replace("sm_", "").replace(".", "")
                cmd += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if ccbin:
            cmd += ["-ccbin", ccbin]
        cmd += ["-Wno-deprecated-gpu-targets", str(SRC), "-o", str(out)]
        self.announce("fissionattn: " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(ROOT))


setup(
    name="fissionattn",
    version="0.1.0",
    description="CUDA-core MLA attention backend for vLLM (Pascal sm_61)",
    packages=["fissionattn"],
    license="Apache-2.0",
    cmdclass={"build_py": NativeBuildPy},
)
