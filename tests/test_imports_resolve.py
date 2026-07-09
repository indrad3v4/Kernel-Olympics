"""Guard against used-but-never-imported stdlib modules in src/.

Regression for the 2026-07-09 notebook run: d944ba7 added a compile
progress bar using `threading.Event()` / `threading.Thread()` inside
`KernelOlympics.run()` without adding `import threading`. Nothing catches
this class of bug today — `py_compile` passes (it's valid syntax), the
test suite passes (nothing executes that code path), and the failure is a
runtime NameError that fired only AFTER a full 22-minute porting run,
destroying its verification step.

This test walks each src module's AST and asserts that any reference to a
well-known stdlib module name is backed by an import SOMEWHERE in the
file (module level or function level — this codebase uses both). It is
deliberately narrow: only module names from the fixed list below are
checked, so locals/params that happen to shadow other names can't false-
positive.
"""
import ast
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

# Only flag these exact names — all stdlib modules this codebase actually
# uses somewhere. A Name reference to one of these is either the module
# (needs an import) or a shadowing local (excluded below).
STDLIB_MODULES = {
    "argparse", "atexit", "difflib", "json", "logging", "os", "re",
    "shutil", "socket", "subprocess", "sys", "tempfile", "threading",
    "time", "traceback", "urllib", "uuid",
}


def _imported_names(tree: ast.AST) -> set:
    """Every name bound by any import statement anywhere in the file."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _locally_bound_names(tree: ast.AST) -> set:
    """Names bound by assignment/params/comprehensions — may shadow modules."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for a in (args.args + args.posonlyargs + args.kwonlyargs):
                bound.add(a.arg)
            if args.vararg:
                bound.add(args.vararg.arg)
            if args.kwarg:
                bound.add(args.kwarg.arg)
    return bound


def _module_files():
    for root, _dirs, files in os.walk(SRC):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                yield Path(root) / f


def test_every_used_stdlib_module_is_imported():
    problems = []
    for path in _module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = _imported_names(tree)
        shadowed = _locally_bound_names(tree)
        used = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in STDLIB_MODULES
        }
        missing = used - imported - shadowed
        if missing:
            rel = path.relative_to(SRC.parent)
            problems.append(f"{rel}: uses {sorted(missing)} but never imports them")
    assert not problems, (
        "Used-but-never-imported stdlib modules (runtime NameError waiting "
        "to fire on a code path no test executes):\n  " + "\n  ".join(problems)
    )
