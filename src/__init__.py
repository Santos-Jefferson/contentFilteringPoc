"""Content filtering POC package."""

from pathlib import Path
import os


def _load_workspace_env() -> None:
	root = Path(__file__).resolve().parent.parent
	env_path = root / ".env"
	if not env_path.exists():
		return

	for raw_line in env_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key and key not in os.environ:
			os.environ[key] = value


_load_workspace_env()

