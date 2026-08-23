"""Shared .env loading for every entry point.

Loading happens in entry points (app factories, CLI `main()`s) rather than at
package import, deliberately: the test suite sets `os.environ` at module import
time before importing the app, and a package-level `load_dotenv` would make the
developer's local `.env` visible to tests that depend on a variable being absent.

`load_dotenv` does not override variables that are already set, so an explicit
`FOO=bar python -m ...` still wins over the file.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Optional[Path] = None) -> bool:
    """Load the repo-root .env. Returns True if a file was found.

    Called by the servers *and* by the ingest/query CLIs. Skipping it in the CLIs
    was a real bug: `GOOGLE_AI_STUDIO_API_KEY` would sit in `.env` unread, and the
    backfill would refuse to run with a message telling you to set the key you had
    already set.
    """
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.exists():
        return False
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=env_path)
    return True
