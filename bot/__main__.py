"""Allow ``python -m bot`` to start the ticket responder."""

from __future__ import annotations

import sys

from bot.main import cli

if __name__ == "__main__":
    sys.exit(cli())
