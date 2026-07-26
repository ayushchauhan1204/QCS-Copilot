"""
data/ — Synthetic data generator package.

Exposes the five pipeline modules in build order:
  recipes → disturbances → transitions → labels → generate_data

No module outside data/ may import from here directly except
through src/data_loader.py (per PROJECT_CONSTITUTION.md Rule 6).
"""
