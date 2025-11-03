"""Check that basic features work.

Catch cases where e.g. files are missing so the import doesn't work. It is
recommended to check that e.g. assets are included."""

import populationsim

if populationsim is not None:
    print("Smoke test succeeded")
