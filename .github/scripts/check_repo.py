#!/usr/bin/env python3
import argparse
import ast
import re
import subprocess
import sys
import tokenize
from io import BytesIO
from pathlib import Path

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")

HARD_PATH_PATTERNS = [
    re.compile(r"^/[^/].+"),
    re.compile(r"^~/.+"),
    re.compile(r"^[A-Za-z]:\\\\.+"),
    re.compile(r"^\\\\\\\\[^\\].+"),
    re.compile(r"^/Users/[^/].+"),
    re.compile(r"^/home/[^/].+"),
    re.compile(r"^/mnt/[^/].+"),
    re.compile(r"^/var/[^/].+"),
    re.compile(r"^/opt/[^/].+"),
]

# Exact absolute paths that are intentionally allowed in the repo.
# Example: serial device default for specific hardware.
ALLOWED_ABS_PATHS: set[str] = {}

# Allowed absolute path prefixes. Any string beginning with one of these
# prefixes will be considered acceptable and not flagged.
ALLOWED_ABS_PREFIXES: tuple[str, ...] = (
    "/dev/",  # allow device files like /dev/ttyUSB0
)


def get_changed_py_files(base_ref: str) -> list[Path]:
    try:
        cmd = ["git", "diff", "--name-only", f"{base_ref}...HEAD"]
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        return [Path(p) for p in out if p.endswith(".py") and Path(p).exists()]
    except Exception:
        return []


def list_repo_py_files() -> list[Path]:
    return [p for p in Path(".").rglob("*.py") if p.is_file()]


def is_hardcoded_path(s: str) -> bool:
    s_stripped = s.strip()

    # Whitelist specific known-good absolute paths
    if s_stripped in ALLOWED_ABS_PATHS:
        return False

    # Allow any path under /dev/*, including "/dev" itself
    if s_stripped == "/dev" or any(s_stripped.startswith(p) for p in ALLOWED_ABS_PREFIXES):
        return False

    for pat in HARD_PATH_PATTERNS:
        if pat.search(s_stripped):
            return True
    return False


def check_file_for_issues(path: Path) -> list[str]:
    errors = []
    try:
        data = path.read_bytes()
    except Exception as e:
        errors.append(f"{path}: [READ-ERROR] {e}")
        return errors

    # 1) Token-level checks (e.g., Chinese characters in comments)
    try:
        for tok in tokenize.tokenize(BytesIO(data).readline):
            if tok.type == tokenize.COMMENT and CHINESE_RE.search(tok.string):
                errors.append(f"{path}:{tok.start[0]}: Chinese characters found in comment.")
    except tokenize.TokenError:
        pass

    # 2) AST-level checks for hardcoded absolute paths
    try:
        tree = ast.parse(data.decode("utf-8", errors="ignore"))

        class PathCheckVisitor(ast.NodeVisitor):
            def __init__(self):
                self.in_fstring_stack: list[bool] = []

            def visit_JoinedStr(self, node: ast.JoinedStr):
                # Mark that we are inside an f-string; constants inside will be ignored.
                self.in_fstring_stack.append(True)
                self.generic_visit(node)
                self.in_fstring_stack.pop()

            def visit_Constant(self, node: ast.Constant):
                if isinstance(node.value, str):
                    s = node.value
                    in_fstring = any(self.in_fstring_stack)
                    # If this string literal is part of an f-string, skip it.
                    # This prevents false positives like f"{PROJECT_ROOT}/view_point.json".
                    if not in_fstring and is_hardcoded_path(s):
                        lineno = getattr(node, "lineno", "?")
                        errors.append(
                            f"{path}:{lineno}: Hardcoded absolute path in string: {repr(s)[:120]}"
                        )

        PathCheckVisitor().visit(tree)
    except Exception as e:
        errors.append(f"{path}: [PARSE-ERROR] {e}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Repo checks: Chinese comments & hardcoded paths")
    parser.add_argument("--base-ref", type=str, default="origin/main")
    args = parser.parse_args()

    files = get_changed_py_files(args.base_ref)
    if not files:
        files = list_repo_py_files()

    all_errors: list[str] = []
    for f in files:
        all_errors.extend(check_file_for_issues(f))

    if all_errors:
        print("Found issues:\n")
        for e in all_errors:
            print(e)
        print("\nFail: please remove Chinese comments and hardcoded absolute paths.")
        sys.exit(1)
    else:
        print("No issues found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
