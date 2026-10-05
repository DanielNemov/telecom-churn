from pathlib import Path

from telecom_churn.config import PROJECT_ROOT, RANDOM_STATE, TARGET_COL, TEST_SIZE


def test_project_root_exists():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_config_constants():
    assert TARGET_COL == "Churn"
    assert TEST_SIZE == 0.2
    assert RANDOM_STATE == 42
    assert isinstance(PROJECT_ROOT, Path)
