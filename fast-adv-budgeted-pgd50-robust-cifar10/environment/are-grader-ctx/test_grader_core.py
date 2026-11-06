#!/usr/bin/env python3
"""Unit tests for grader_core that need neither the sealed test split nor a GPU.

Run with:  python -m pytest -q environment/are-grader-ctx/test_grader_core.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grader_core


class _Rescaled(torch.nn.Module):
    """Same predictions as `base`, logits multiplied by a positive constant."""

    def __init__(self, base, factor):
        super().__init__()
        self.base = base
        self.factor = factor

    def forward(self, x):
        return self.base(x) * self.factor


def _tiny_model():
    torch.manual_seed(1234)
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10)).eval()


@pytest.fixture
def small_cpu_eval(monkeypatch):
    """Grade 64 synthetic images with a short attack instead of the sealed split."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(64, 3, 32, 32, generator=gen)
    y = torch.randint(0, 10, (64,), generator=gen)
    monkeypatch.setattr(grader_core, "load_test_tensors", lambda device, split: (x.clone(), y.clone()))
    monkeypatch.setattr(grader_core, "ATTACK_ITERS", 5)
    monkeypatch.setattr(grader_core, "RESTARTS", 2)
    return lambda model: grader_core.evaluate(model, torch.device("cpu"), "intermediate")


@pytest.mark.parametrize("factor", [1.0 / 16.0, 16.0])
def test_score_does_not_depend_on_logit_scale(small_cpu_eval, factor):
    base = _tiny_model()
    reference = small_cpu_eval(base)
    rescaled = small_cpu_eval(_Rescaled(base, factor))

    assert rescaled["logit_scale_s"] == pytest.approx(reference["logit_scale_s"] * factor)
    assert rescaled["temperature"] == pytest.approx(reference["temperature"] * factor)
    assert rescaled["clean_accuracy_percent"] == reference["clean_accuracy_percent"]
    assert rescaled["robust_accuracy_percent"] == reference["robust_accuracy_percent"]


def test_all_equal_logits_do_not_divide_by_zero(small_cpu_eval):
    model = _tiny_model()
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    result = small_cpu_eval(model)
    assert result["logit_scale_s"] == 0.0
    assert result["temperature"] == 1.0
