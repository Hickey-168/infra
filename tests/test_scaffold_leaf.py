"""Contract tests for the per-leaf learning artifact scaffold."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD_PATH = REPO_ROOT / "tools" / "scaffold_leaf.py"


def load_scaffold_module():
    spec = importlib.util.spec_from_file_location("scaffold_leaf", SCAFFOLD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScaffoldLeafTest(unittest.TestCase):
    def test_scaffold_does_not_generate_or_overwrite_curriculum_tutorial(self):
        scaffold_leaf = load_scaffold_module()
        original_root = scaffold_leaf.REPO_ROOT
        original_template = scaffold_leaf.TEMPLATE_LEARNING

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            leaf_rel = "operators/triton/example-leaf"
            leaf_dir = temp_root / leaf_rel
            leaf_dir.mkdir(parents=True)
            (temp_root / "templates").mkdir()
            (temp_root / "templates" / "learning-item.md").write_text(
                "# <Topic>\n", encoding="utf-8"
            )
            learner_note = "# My own explanation\n\nI predicted this first.\n"
            (leaf_dir / "note.md").write_text(learner_note, encoding="utf-8")
            tutorial = leaf_dir / "tutorial.md"
            tutorial.write_text("# AI-authored tutorial\n", encoding="utf-8")

            try:
                scaffold_leaf.REPO_ROOT = temp_root
                scaffold_leaf.TEMPLATE_LEARNING = temp_root / "templates" / "learning-item.md"
                scaffold_leaf.scaffold(leaf_rel, force=False)
            finally:
                scaffold_leaf.REPO_ROOT = original_root
                scaffold_leaf.TEMPLATE_LEARNING = original_template

            self.assertEqual(tutorial.read_text(encoding="utf-8"), "# AI-authored tutorial\n")
            self.assertEqual((leaf_dir / "note.md").read_text(encoding="utf-8"), learner_note)

            new_leaf_rel = "operators/triton/no-prewritten-tutorial"
            (temp_root / new_leaf_rel).mkdir(parents=True)
            try:
                scaffold_leaf.REPO_ROOT = temp_root
                scaffold_leaf.scaffold(new_leaf_rel, force=False)
            finally:
                scaffold_leaf.REPO_ROOT = original_root
                scaffold_leaf.TEMPLATE_LEARNING = original_template

            self.assertFalse((temp_root / new_leaf_rel / "tutorial.md").exists())


if __name__ == "__main__":
    unittest.main()
