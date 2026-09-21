"""Blockwise 4-bit storage for the parts of a recreation that never train.

A recreation freezes everything the donor supplied and trains only the strands it could not. Frozen
weights need to be *readable*, not differentiable, so they do not have to be stored at training
precision -- and for a mixture-of-experts donor that distinction is the difference between a model
that fits on one card and one that does not. GPT-OSS-120B is about 117B parameters: 234 GB in
bfloat16, roughly 62 GB at four bits.

The scheme here is the usual blockwise affine one, written in plain PyTorch so it runs on CPU and
CUDA alike and needs no compiled dependency:

* split each tensor into contiguous blocks of ``block_size`` values;
* scale each block by its own absolute maximum, so one outlier cannot flatten a whole tensor;
* map the scaled values onto a fixed 16-entry codebook and pack two codes per byte.

The codebook is NF4's: the quantiles of a standard normal, which is the right prior for neural
network weights and measurably better than uniform spacing for the same four bits. Storage works out
at about 0.53 bytes per parameter including the per-block scales.

This is lossy, and :func:`quantization_error` exists so the loss can be measured rather than
assumed. It is only ever applied to frozen tensors; anything being trained stays at full precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

#: NF4 codebook -- the 16 quantiles of a standard normal, normalised to [-1, 1].
NF4_CODEBOOK: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

DEFAULT_BLOCK_SIZE = 64


def _codebook(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(NF4_CODEBOOK, device=device, dtype=dtype)


@dataclass
class QuantizedTensor:
    """A tensor held as packed 4-bit codes plus per-block scales.

    Attributes:
        codes: ``uint8``, two 4-bit codes per byte.
        absmax: One scale per block.
        shape: The original shape.
        block_size: Values per block.
        dtype: The dtype to dequantize back to.
    """

    codes: torch.Tensor
    absmax: torch.Tensor
    shape: torch.Size
    block_size: int
    dtype: torch.dtype

    @property
    def numel(self) -> int:
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count

    def nbytes(self) -> int:
        """Bytes actually held, scales included."""
        return self.codes.numel() * self.codes.element_size() + self.absmax.numel() * self.absmax.element_size()

    def bytes_per_parameter(self) -> float:
        return self.nbytes() / max(1, self.numel)

    def to(self, device) -> QuantizedTensor:
        return QuantizedTensor(
            self.codes.to(device), self.absmax.to(device), self.shape, self.block_size, self.dtype
        )


def quantize_4bit(
    tensor: torch.Tensor, *, block_size: int = DEFAULT_BLOCK_SIZE
) -> QuantizedTensor:
    """Pack a tensor into blockwise 4-bit codes.

    Args:
        tensor: The tensor to quantize. Its values are read, never modified.
        block_size: Values sharing one scale. Smaller is more faithful and costs more scale storage.

    Returns:
        A :class:`QuantizedTensor`.
    """
    original_shape = tensor.shape
    flat = tensor.detach().reshape(-1).float()
    padding = (-flat.numel()) % block_size
    if padding:
        flat = torch.cat([flat, flat.new_zeros(padding)])
    blocks = flat.view(-1, block_size)

    absmax = blocks.abs().amax(dim=1, keepdim=True)
    # A block of exact zeros would divide by zero; its codes all land on the codebook's own zero.
    scaled = blocks / absmax.clamp_min(1e-12)

    codebook = _codebook(tensor.device)
    # Nearest codebook entry per value. The codebook is tiny, so a full pairwise distance is cheap
    # and exact, where a bucketed search would need the codebook to be evenly spaced -- it is not.
    codes = (scaled.unsqueeze(-1) - codebook).abs().argmin(dim=-1).to(torch.uint8)

    flat_codes = codes.reshape(-1)
    packed = (flat_codes[0::2] << 4) | flat_codes[1::2]
    return QuantizedTensor(
        codes=packed,
        absmax=absmax.squeeze(1).to(torch.float16),
        shape=original_shape,
        block_size=block_size,
        dtype=tensor.dtype,
    )


def dequantize_4bit(quantized: QuantizedTensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Unpack a :class:`QuantizedTensor` back to a dense tensor."""
    packed = quantized.codes
    high = (packed >> 4).to(torch.long)
    low = (packed & 0x0F).to(torch.long)
    codes = torch.stack([high, low], dim=1).reshape(-1)

    codebook = _codebook(packed.device)
    values = codebook[codes].view(-1, quantized.block_size)
    values = values * quantized.absmax.to(values.dtype).unsqueeze(1)

    count = quantized.numel
    return values.reshape(-1)[:count].view(quantized.shape).to(dtype or quantized.dtype)


def quantization_error(tensor: torch.Tensor, *, block_size: int = DEFAULT_BLOCK_SIZE) -> dict[str, float]:
    """Measure what quantizing a tensor costs, rather than assuming it.

    Returns:
        ``relative_error`` (Frobenius), ``max_abs_error``, ``cosine`` similarity and
        ``bytes_per_parameter``.
    """
    quantized = quantize_4bit(tensor, block_size=block_size)
    restored = dequantize_4bit(quantized, dtype=torch.float32)
    reference = tensor.detach().float()
    difference = restored - reference
    return {
        "relative_error": float(difference.norm() / reference.norm().clamp_min(1e-12)),
        "max_abs_error": float(difference.abs().max()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(restored.reshape(1, -1), reference.reshape(1, -1))
        ),
        "bytes_per_parameter": quantized.bytes_per_parameter(),
    }


class Quantized4bitParameter(nn.Module):
    """Holds one frozen parameter at four bits, dequantizing on access.

    Registered as buffers so the packed codes move with ``.to(device)`` and survive a
    ``state_dict`` round trip.
    """

    def __init__(self, tensor: torch.Tensor, *, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        super().__init__()
        quantized = quantize_4bit(tensor, block_size=block_size)
        self.register_buffer("codes", quantized.codes)
        self.register_buffer("absmax", quantized.absmax)
        self.shape = quantized.shape
        self.block_size = quantized.block_size
        self.original_dtype = quantized.dtype

    def forward(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.dequantize(dtype)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        quantized = QuantizedTensor(
            self.codes, self.absmax, self.shape, self.block_size, self.original_dtype
        )
        return dequantize_4bit(quantized, dtype=dtype)

    def nbytes(self) -> int:
        return self.codes.numel() + self.absmax.numel() * self.absmax.element_size()
