"""Execute notebook training imports without Drive, GPU, or training side effects."""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["00_pipeline_colab.ipynb", "03_cnn_training.ipynb"])
def test_e01_e02_notebook_imports_use_keras_api(filename):
    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    imported = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # These notebooks call subprocess rather than IPython magics.
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {"src.train", "src.train_keras"}:
                assert node.module == "src.train_keras", "E01/E02 must not import the E03 PyTorch API"
                namespace = {}
                exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
                for alias in node.names:
                    assert callable(namespace[alias.asname or alias.name])
                    imported.add(alias.name)
    assert {"load_experiment_config", "run_training"} <= imported


@pytest.mark.parametrize("filename", ["00_pipeline_colab.ipynb", "03_cnn_training.ipynb"])
def test_notebooks_audit_before_training(filename):
    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert code.index("audit = run_audit(") < code.index("= run_training(")
    assert 'audit["automated_checks_passed"]' in code
