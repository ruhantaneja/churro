from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_system_prompt() -> str:
    return (_PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")