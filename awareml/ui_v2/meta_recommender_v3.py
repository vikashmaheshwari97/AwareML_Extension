"""Compatibility entry point; the main workspace uses the same current recommender."""
from __future__ import annotations


def render_meta_recommender_v3():
    from .copilot_recommender import render_dataset_aware_tab
    return render_dataset_aware_tab()
