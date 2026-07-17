"""Adapters for external tokenized world-model platforms.

Adapters deliberately keep upstream architectures at arm's length. They translate
data, sampled tokens, and teacher-forced log-probabilities into DoR's verifier and
credit-assignment protocol without changing upstream model code.
"""
