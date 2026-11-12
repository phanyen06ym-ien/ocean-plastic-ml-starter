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


def test_colab_review_checks_out_pr_branch_even_when_repo_exists():
    notebook = json.loads((ROOT / "notebooks/00_pipeline_colab.ipynb").read_text(encoding="utf-8"))
    setup = "".join(notebook["cells"][2]["source"])
    assert 'BRANCH = "fix/dataset-audit-mobilenetv2"' in setup
    assert 'if not REPO.exists()' in setup
    assert '"fetch", "--depth=1", "origin", BRANCH' in setup
    assert '"checkout", "-B", BRANCH, "FETCH_HEAD"' in setup
    assert 'git", "branch", "--show-current"' in setup
    assert '"rev-parse", "HEAD"' in setup and "COMMIT_SHA" in setup
    assert 'REPO / "src/data_audit.py"' in setup and '"ls-files", "--error-unmatch", "src/data_audit.py"' in setup


def test_colab_never_rebuilds_frozen_v1_or_overwrites_mapping():
    notebook = json.loads((ROOT / "notebooks/00_pipeline_colab.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "v1_crop128" in code
    assert "FileNotFoundError" in code and "khôi phục" in code and "v1_crop128.zip" in code
    assert "run_preprocessing" not in code
    assert "download_trashcan" not in code
    assert "shutil.copy(PROCESSED / \"class_mapping.json\"" not in code
    assert "mapping_out=" not in code
    # The archive restores to a fresh Colab-local path; no script writes to data/processed on Drive.
    assert '"unzip", "-q", str(PROCESSED_ZIP)' in code
    assert 'str(PROCESSED.parent)' in code


def test_colab_results_export_stays_on_drive_without_token_push():
    notebook = json.loads((ROOT / "notebooks/00_pipeline_colab.ipynb").read_text(encoding="utf-8"))
    all_cells = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "EXPORT = DRIVE / \"exports\"" in all_cells
    assert "GH_TOKEN" not in all_cells
    assert "x-access-token" not in all_cells
    assert "userdata.get" not in all_cells
    assert "git push" not in all_cells
    assert "shell=True" not in all_cells
    assert "PR riêng" in all_cells
