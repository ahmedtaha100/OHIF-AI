#!/usr/bin/env python
"""Development-only freezer for the post-failure EDL fusion hybrid."""

from __future__ import annotations

from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rl_nninteractive.edl_fusion_hybrid import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
