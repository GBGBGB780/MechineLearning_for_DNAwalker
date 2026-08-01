"""Allow ``python -m dnawalker`` to use the unified CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
