"""A supervised, persistent Julia runtime with live output and interrupt.

A self-contained package (standard library only): ``JuliaKernel`` spawns and
supervises a Julia process, streams its output live, and evaluates code over a
framed control channel.
"""

from .config import KernelConfig
from .kernel import JuliaKernel, JuliaStartupError
from .result import EvalResult, OnChunk, OutputChunk

__all__ = [
    "EvalResult",
    "JuliaKernel",
    "JuliaStartupError",
    "KernelConfig",
    "OnChunk",
    "OutputChunk",
]
