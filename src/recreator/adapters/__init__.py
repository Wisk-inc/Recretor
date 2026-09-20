"""Reserved for donor-family-specific handling.

Dense Llama/Qwen-style donors and the MoE families both go through
:mod:`recreator.mapping` and :mod:`recreator.techniques.expert_merge`, so nothing family-specific
is needed yet. This package is where that would live if a donor ever needs it.
"""
