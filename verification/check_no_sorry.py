#!/usr/bin/env python3
"""Static proof-integrity guard for the Lean 4 development.

``lake build`` only *warns* when a proof uses ``sorry``, so CI can stay green
while a theorem regresses to a placeholder. This script fails the build when a
Lean source contains ``sorry``/``admit`` or declares a custom ``axiom``.

Comments and string literals are stripped first (Lean block comments nest), so
prose such as "zero sorry" in a header comment is not flagged.

With ``--axioms FILE`` it also validates the output of
``lake env lean CheckAxioms.lean``: every printed dependency must be one of the
three standard Lean axioms, otherwise the job fails.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
FORBIDDEN_TOKENS = ("sorry", "admit")
AXIOM_DECL = re.compile(r"(?m)^[ \t]*(?:private[ \t]+|protected[ \t]+)?axiom\b")
AXIOM_LIST = re.compile(r"depends on axioms:[ \t]*\[([^\]]*)\]")
CERT_LINE = re.compile(r"depends on axioms:|does not depend on any axioms")


def strip_comments_and_strings(text: str) -> str:
    """Remove comments and string literals while preserving line structure."""
    out: list[str] = []
    i, n, depth = 0, len(text), 0
    while i < n:
        pair = text[i : i + 2]
        if depth == 0 and pair == "/-":
            depth, i = 1, i + 2
            continue
        if depth > 0:
            if pair == "/-":
                depth, i = depth + 1, i + 2
            elif pair == "-/":
                depth, i = depth - 1, i + 2
            else:
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            continue
        if pair == "--":
            j = text.find("\n", i)
            if j == -1:
                break
            i = j
            continue
        if text[i] == '"':
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            out.append('""')
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def scan_sources(root: Path) -> list[str]:
    failures: list[str] = []
    files = sorted(root.rglob("*.lean"))
    if not files:
        failures.append(f"{root}: no .lean files found")
    for path in files:
        code = strip_comments_and_strings(path.read_text(encoding="utf-8"))
        for token in FORBIDDEN_TOKENS:
            for match in re.finditer(rf"\b{token}\b", code):
                failures.append(
                    f"{path}:{_line_of(code, match.start())}: forbidden '{token}'"
                )
        for match in AXIOM_DECL.finditer(code):
            failures.append(
                f"{path}:{_line_of(code, match.start())}: custom 'axiom' declaration"
            )
    return failures


def scan_axioms(path: Path) -> list[str]:
    failures: list[str] = []
    text = path.read_text(encoding="utf-8")
    certified = 0
    for line in text.splitlines():
        if "sorryAx" in line:
            failures.append(f"{path}: proof depends on 'sorryAx': {line.strip()}")
        if CERT_LINE.search(line):
            certified += 1
        match = AXIOM_LIST.search(line)
        if match:
            for name in (item.strip() for item in match.group(1).split(",")):
                if name and name not in ALLOWED_AXIOMS:
                    failures.append(f"{path}: unexpected axiom '{name}'")
    if certified == 0:
        failures.append(f"{path}: no '#print axioms' output recognised; refusing to certify")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=str(Path(__file__).resolve().parent),
        help="directory scanned for .lean sources (default: this script's directory)",
    )
    parser.add_argument("--axioms", help="file containing '#print axioms' output")
    args = parser.parse_args()

    failures = scan_sources(Path(args.root))
    if args.axioms:
        failures.extend(scan_axioms(Path(args.axioms)))

    if failures:
        print("PROOF INTEGRITY FAILURE")
        for failure in failures:
            print(" -", failure)
        return 1
    print(f"proof integrity OK ({len(list(Path(args.root).rglob('*.lean')))} Lean files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
