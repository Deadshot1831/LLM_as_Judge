"""Load a versioned rubric and render it for humans and for the judge."""
import pathlib

import yaml

RUBRIC_DIR = pathlib.Path(__file__).resolve().parent.parent / "rubrics"


def load(version):
    path = RUBRIC_DIR / f"{version}.yaml"
    if not path.exists():
        raise SystemExit(f"no rubric {version} — have: {[p.stem for p in RUBRIC_DIR.glob('*.yaml')]}")
    rubric = yaml.safe_load(path.read_text())
    rubric["_raw"] = path.read_text()
    return rubric


def versions():
    return sorted(p.stem for p in RUBRIC_DIR.glob("*.yaml"))


def criteria(rubric):
    return [c["name"] for c in rubric["criteria"]]


def register(rubric):
    """Store the rubric text in Postgres so old runs stay interpretable."""
    from judge import db

    db.execute(
        "INSERT INTO rubric_versions (version, body) VALUES (%s, %s) ON CONFLICT (version) DO NOTHING",
        (rubric["version"], rubric["_raw"]),
    )


def render(rubric):
    """Plain-text rubric used verbatim in the judge prompt and in the labeling UI."""
    out = [f"RUBRIC {rubric['version']} — scale {rubric['scale']}"]
    for c in rubric["criteria"]:
        out.append(f"\n## {c['name']}")
        out.append(c["question"].strip())
        for point, text in sorted(c["points"].items()):
            out.append(f"  {point} = {text}")
        for point, anchor in sorted(c.get("anchors", {}).items()):
            out.append(f"  anchor for {point}: {anchor['answer']!r} — {anchor['why']}")
    return "\n".join(out)
