"""Require the release tag to match the version in the checked-out package."""

import os
import tomllib
from pathlib import Path

version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
tag = os.environ["GITHUB_REF_NAME"]
if tag != f"v{version}":
    raise SystemExit(f"Tag {tag!r} does not match package version {version!r}")
print(f"Release tag {tag!r} matches package version {version!r}")
