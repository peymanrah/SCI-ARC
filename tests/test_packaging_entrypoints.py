from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _extract_console_scripts_from_setup_py(setup_py: Path) -> list[str]:
    tree = ast.parse(setup_py.read_text(encoding="utf-8"))

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.console_scripts: list[str] = []

        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "setup":
                for kw in node.keywords:
                    if kw.arg == "entry_points" and isinstance(kw.value, ast.Dict):
                        for k, v in zip(kw.value.keys, kw.value.values):
                            if isinstance(k, ast.Constant) and k.value == "console_scripts":
                                if isinstance(v, ast.List):
                                    for elt in v.elts:
                                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                            self.console_scripts.append(elt.value)
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    return visitor.console_scripts


@pytest.mark.cpu
def test_setup_py_console_entrypoints_are_importable_modules():
    """Verify console_scripts in setup.py (if any) point to importable modules.
    
    Note: As of the bug fix, console_scripts are commented out because the 
    scripts package is excluded from packages. This test passes if:
    - There are no console_scripts (safe - no broken entry points)
    - OR all listed console_scripts point to importable modules
    """
    setup_py = Path("setup.py")
    assert setup_py.exists(), "Expected setup.py at repo root"

    console_scripts = _extract_console_scripts_from_setup_py(setup_py)
    
    # If no console_scripts are defined, that's fine (and expected after the fix)
    if not console_scripts:
        return  # Pass - no broken entry points
    
    # If console_scripts are defined, verify they're importable
    for entry in console_scripts:
        _, rhs = entry.split("=", 1)
        module_path, _func = rhs.split(":", 1)
        module_path = module_path.strip()

        __import__(module_path)
