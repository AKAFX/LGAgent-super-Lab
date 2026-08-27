from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.baseline_check import (
    verify_credential_precedence,
    verify_dependency_contract,
    verify_manifest,
)


class BaselineOfflineTest(unittest.TestCase):
    def test_primary_dataset_manifest(self) -> None:
        verify_manifest()

    def test_dependencies_are_reproducible(self) -> None:
        verify_dependency_contract()

    def test_yaml_credentials_precede_environment_fallback(self) -> None:
        verify_credential_precedence()


if __name__ == "__main__":
    unittest.main()
