"""The library must not import the application layer.

``pyproject.toml`` packages ``src/kg_gen`` and nothing else, so a
``from app... import ...`` inside ``kg_gen`` fails for everyone who installs
the wheel, and for anything that puts only ``src`` on the path -- the
kg-gen-plus-eval benchmarks do exactly that. It stayed unnoticed for
``kg_gen.kg_gen`` because this repo's own tests and the server both make the
repo root importable, so ``app`` happens to be there.

``app.kggen_logger`` is ``logging.getLogger("kg_gen_app")`` and nothing more,
so anything in the library that wants that logger should ask for it by name.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "kg_gen"
LIBRARY_MODULES = sorted(SRC.rglob("*.py"))


def _app_imports(tree: ast.AST) -> list[str]:
    """Every ``app``/``app.x`` import in a module, including inside functions."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "app" or module.startswith("app."):
                found.append(f"from {module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "app" or alias.name.startswith("app."):
                    found.append(f"import {alias.name}")
    return found


def test_there_are_library_modules_to_check():
    """Guard the guard: a bad path would make every case below vacuous."""
    assert len(LIBRARY_MODULES) > 5


@pytest.mark.parametrize(
    "path", LIBRARY_MODULES, ids=[str(p.relative_to(SRC)) for p in LIBRARY_MODULES]
)
def test_module_does_not_import_the_app_package(path):
    tree = ast.parse(path.read_text(), filename=str(path))

    offenders = _app_imports(tree)

    assert not offenders, (
        f"{path.relative_to(SRC)} imports the app package "
        f"({', '.join(offenders)}), which the wheel does not ship. "
        f"For the logger, use logging.getLogger('kg_gen_app')."
    )
