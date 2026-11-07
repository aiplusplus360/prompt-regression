"""Entry point for `python -m promptreg`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
