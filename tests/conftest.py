"""Shared test setup."""

import os

# Qt widget tests render offscreen, so they run without a display (CI, SSH).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
