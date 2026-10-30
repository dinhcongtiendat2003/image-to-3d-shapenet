import json
from pathlib import Path
import re

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "01_setup_and_dataset.ipynb",
    "02_render_shapenet.ipynb",
    "03_train_refiner.ipynb",
    "04_evaluate.ipynb",
    "05_inference_gradio.ipynb",
]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_colab_notebook_is_valid_and_self_checking(name: str) -> None:
    path = ROOT / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells)
    required = (
        "quick", "requirements-colab.txt", "cuda", "HF_TOKEN", "drive", "resume",
        "RuntimeError",
    )
    for value in required:
        assert value.lower() in source.lower(), f"{name} lacks {value}"
    assert re.search(r"hf_[A-Za-z0-9]{20,}", source) is None
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{name}:cell-{index}", "exec")


def test_notebook_sequence_is_complete() -> None:
    actual = {path.name for path in (ROOT / "notebooks").glob("0[1-5]_*.ipynb")}
    assert set(NOTEBOOKS) <= actual

