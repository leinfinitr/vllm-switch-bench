"""Check documentation topology and local links without prescribing prose."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from llm_switch_bench.common.provenance import repository_root
from llm_switch_bench.families import FAMILIES

LINK = re.compile(r"!?\[[^]]*]\(([^)]+)\)")
HOME_PATH = re.compile(r"(?<![\w.-])(?:~/|/(?:home|Users)/[^/\s]+/)")


def markdown_files(root: Path) -> list[Path]:
    paths = [root / name for name in ("README.md", "AGENTS.md", "CONTRIBUTING.md")]
    paths.extend((root / "docs").rglob("*.md"))
    for family in FAMILIES:
        paths.extend((root / "results" / family.slug).rglob("*.md"))
    paths.append(root / "results" / "README.md")
    paths.extend((root / directory / "README.md") for directory in ("scripts", "src"))
    return sorted(set(paths))


def local_link_targets(path: Path) -> set[Path]:
    targets: set[Path] = set()
    for match in LINK.finditer(path.read_text(encoding="utf-8")):
        raw = match.group(1).strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1]
        raw = raw.split(maxsplit=1)[0]
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        targets.add((path.parent / unquote(parsed.path)).resolve())
    return targets


def check_markdown(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if HOME_PATH.search(text):
        raise ValueError(f"{path} contains a maintainer-specific home path")
    for target in local_link_targets(path):
        if not target.exists():
            raise ValueError(f"{path} contains a broken local link to {target}")


def check_repository(root: Path | None = None) -> None:
    checkout = (root or repository_root()).resolve()
    experiment_root = checkout / "docs" / "experiments"
    actual_families = {path.name for path in experiment_root.iterdir() if path.is_dir()}
    expected_families = {family.slug for family in FAMILIES}
    if actual_families != expected_families:
        raise ValueError(
            "documentation families differ from the registry: "
            f"expected {sorted(expected_families)}, found {sorted(actual_families)}"
        )

    paths = markdown_files(checkout)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"required documentation is missing: {missing}")
    for path in paths:
        check_markdown(path)

    root_links = local_link_targets(checkout / "README.md")
    docs_links = local_link_targets(checkout / "docs" / "README.md")
    results_links = local_link_targets(checkout / "results" / "README.md")
    for family in FAMILIES:
        protocol = (experiment_root / family.slug / "README.md").resolve()
        result_readme = (checkout / "results" / family.slug / "README.md").resolve()
        png = checkout / "results" / family.slug / "figures" / f"{family.figure_stem}.png"
        pdf = png.with_suffix(".pdf")
        if protocol not in root_links or protocol not in docs_links:
            raise ValueError(f"{family.slug} is missing from a documentation index")
        if result_readme not in results_links:
            raise ValueError(f"{family.slug} is missing from the result index")
        protocol_links = local_link_targets(protocol)
        if png.resolve() not in protocol_links or pdf.resolve() not in protocol_links:
            raise ValueError(f"{family.slug} protocol must link to its PNG and PDF figures")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check documentation topology and local links")
    parser.parse_args(argv)
    try:
        check_repository()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
