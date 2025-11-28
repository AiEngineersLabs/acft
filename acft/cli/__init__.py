"""
CLI package for ACFT.

Exposes `main` so that the console script entry point
`acft = "acft.cli:main"` works correctly.
"""

from .acft_cli import main  # re-export main()

__all__ = ["main"]