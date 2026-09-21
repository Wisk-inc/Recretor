"""What a given context length actually costs.

HELIX's selling point is that a decode step's work does not grow with the context behind it: strand
L reads a fixed window, strand R carries a fixed-size state, and strand I reads a fixed number of
retrieved blocks after a descent that deepens only logarithmically. That is true, and this module
puts numbers on it.

It also puts numbers on the part the claim does not cover. Storage is ``O(N)`` and unavoidable: a
model that can quote an exact token from arbitrarily far back must still have that token written
down somewhere. What HELIX changes is *where* it has to live -- the history is append-only, never
rewritten, and a step touches a bounded slice of it, so it can sit in host memory or on NVMe
instead of VRAM. The arithmetic below separates the three, because they scale differently and fail
differently:

* **hot state** -- what must be in VRAM, constant in ``N``;
* **cold store** -- the key/value history and landmark tree, linear in ``N``, streamable;
* **per-step reads** -- what a decoded token pulls back from the cold store, constant in ``N``.

Only the layers running strand I keep a full history. The rest are capped at their sliding window,
so they contribute a constant regardless of how long the context grows.
"""

from __future__ import annotations

from dataclasses import dataclass

_TB = 1e12
_GB = 1e9
_MB = 1e6


def _humanize(nbytes: float) -> str:
    for scale, suffix in ((_TB, "TB"), (_GB, "GB"), (_MB, "MB")):
        if nbytes >= scale:
            return f"{nbytes / scale:.1f} {suffix}"
    return f"{nbytes / 1e3:.1f} KB" if nbytes >= 1e3 else f"{nbytes:.0f} B"


@dataclass
class ContextCost:
    """The cost of holding and decoding from ``tokens`` of context."""

    tokens: int
    index_layers: int
    local_layers: int
    hot_state: float
    kv_store: float
    landmark_store: float
    read_per_token: float
    descent_levels: int
    prefill_seconds: float

    @property
    def cold_store(self) -> float:
        return self.kv_store + self.landmark_store

    def read_latency_ms(self, bandwidth_gb_s: float = 7.0) -> float:
        """Time to pull one step's reads at a given storage bandwidth (NVMe is ~7 GB/s)."""
        return 1e3 * self.read_per_token / (bandwidth_gb_s * _GB)

    def render(self) -> str:
        prefill = (
            f"{self.prefill_seconds / 3600:.1f} h"
            if self.prefill_seconds < 3 * 86400
            else f"{self.prefill_seconds / 86400:.0f} days"
        )
        return "\n".join(
            [
                f"context            {self.tokens:,} tokens",
                f"  hot state (VRAM) {_humanize(self.hot_state):>10}   constant in N",
                f"  key/value store  {_humanize(self.kv_store):>10}   linear in N, streamable",
                f"  landmark tree    {_humanize(self.landmark_store):>10}",
                f"  COLD TOTAL       {_humanize(self.cold_store):>10}",
                "",
                f"  read per token   {_humanize(self.read_per_token):>10}   constant in N"
                f"  (~{self.read_latency_ms():.1f} ms from NVMe)",
                f"  tree descent     {self.descent_levels:>10} levels   grows as O(log N)",
                f"  prefill          {prefill:>10}",
            ]
        )


def context_cost(
    config,
    tokens: int,
    *,
    bytes_per_element: int = 2,
    prefill_tokens_per_second: float = 20_000,
) -> ContextCost:
    """Cost of ``tokens`` of context for a HELIX config.

    Args:
        config: The student's ``HelixConfig``.
        tokens: Context length to price.
        bytes_per_element: Cache dtype width (2 for bf16).
        prefill_tokens_per_second: Throughput assumed for building the store.

    Returns:
        A :class:`ContextCost`.
    """
    index_layers = sum(1 for kind in config.layer_types if kind == "helix")
    local_layers = config.num_hidden_layers - index_layers

    # One token's key and value, for one layer, across the key/value head groups.
    per_token_per_layer = 2 * config.num_key_value_heads * config.head_dim * bytes_per_element

    # Only strand-I layers retain the whole past; the others never exceed their sliding window.
    kv_store = tokens * index_layers * per_token_per_layer
    windowed = local_layers * config.sliding_window * per_token_per_layer

    # The landmark tree above the leaves converges to branching/(branching - 1) times the leaf count.
    leaves = tokens / config.block_size
    overhead = config.index_branching / (config.index_branching - 1)
    landmark_store = (
        overhead * leaves * config.num_key_value_heads * config.landmark_dim * bytes_per_element * index_layers
    )

    # Strand R's matrix state plus strand L's ring buffer: the part that must stay resident.
    recurrent_state = (
        config.num_recurrent_heads
        * config.recurrent_head_dim
        * config.recurrent_value_head_dim
        * bytes_per_element
        * config.num_hidden_layers
    )
    hot_state = recurrent_state + windowed

    read_per_token = index_layers * config.index_topk * config.block_size * per_token_per_layer

    return ContextCost(
        tokens=tokens,
        index_layers=index_layers,
        local_layers=local_layers,
        hot_state=hot_state,
        kv_store=kv_store,
        landmark_store=landmark_store,
        read_per_token=read_per_token,
        descent_levels=config.index_num_levels(max(1, int(leaves))),
        prefill_seconds=tokens / max(1.0, prefill_tokens_per_second),
    )


def context_table(config, lengths: tuple[int, ...] = (10**6, 10**7, 10**8, 10**9), **kwargs) -> str:
    """A table of cold-store cost against context length."""
    rows = [("context", "cold store", "hot state", "read/token", "descent")]
    for length in lengths:
        cost = context_cost(config, length, **kwargs)
        rows.append(
            (
                f"{length:.0e}",
                _humanize(cost.cold_store),
                _humanize(cost.hot_state),
                _humanize(cost.read_per_token),
                f"{cost.descent_levels} levels",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)) for row in rows)
