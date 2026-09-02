"""The skill exists twice: `skills/kg-extract/SKILL.md` is the tracked source,
`.claude/skills/kg-extract/SKILL.md` is the copy Claude Code actually loads.
Nothing keeps them equal, and they have silently diverged before — the loaded
copy sat 29 lines behind the source, so guidance that had been written was not
guidance that was being followed. This test is the enforcement.

`.claude/` is gitignored, so the copy is absent in a fresh clone and in CI;
this checks a working copy that has one, and skips where there is nothing to
drift.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "skills" / "kg-extract" / "SKILL.md"
LOADED = REPO / ".claude" / "skills" / "kg-extract" / "SKILL.md"


def test_the_loaded_skill_matches_the_tracked_source():
    assert SOURCE.exists(), SOURCE
    if not LOADED.exists():
        pytest.skip(f"no local {LOADED.relative_to(REPO)} to drift from the source")
    assert SOURCE.read_text() == LOADED.read_text(), (
        f"{LOADED.relative_to(REPO)} has drifted from "
        f"{SOURCE.relative_to(REPO)}. Copy the source over it:\n"
        f"  cp {SOURCE.relative_to(REPO)} {LOADED.relative_to(REPO)}"
    )
