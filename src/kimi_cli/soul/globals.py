import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from kimi_cli.config import Config
from kimi_cli.llm import LLM
from kimi_cli.metadata import Session
from kimi_cli.soul.approval import Approval
from kimi_cli.soul.denwarenji import DenwaRenji
from kimi_cli.utils.logging import logger


class BuiltinSystemPromptArgs(NamedTuple):
    """Builtin system prompt arguments."""

    KIMI_NOW: str
    """The current datetime."""
    KIMI_WORK_DIR: Path
    """The current working directory."""
    KIMI_WORK_DIR_LS: str
    """The directory listing of current working directory."""
    KIMI_AGENTS_MD: str  # TODO: move to first message from system prompt
    """The content of AGENTS.md."""


def load_agents_md(work_dir: Path) -> str | None:
    paths = [
        work_dir / "AGENTS.md",
        work_dir / "agents.md",
    ]
    for path in paths:
        if path.is_file():
            logger.info("Loaded agents.md: {path}", path=path)
            return path.read_text(encoding="utf-8").strip()
    logger.info("No AGENTS.md found in {work_dir}", work_dir=work_dir)
    return None


def _list_work_dir(work_dir: Path) -> str:
    if sys.platform == "win32":
        ls = subprocess.run(
            ["cmd", "/c", "dir", work_dir],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    else:
        ls = subprocess.run(
            ["ls", "-la", work_dir],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    return ls.stdout.strip()


class AgentGlobals(NamedTuple):
    """Agent globals."""

    config: Config
    llm: LLM | None
    session: Session
    builtin_args: BuiltinSystemPromptArgs
    denwa_renji: DenwaRenji
    approval: Approval

    @classmethod
    async def create(
        cls, config: Config, llm: LLM | None, session: Session, yolo: bool
    ) -> "AgentGlobals":
        work_dir = Path(session.work_dir.path)
        # FIXME: do these asynchronously
        ls_output = _list_work_dir(work_dir)
        agents_md = load_agents_md(work_dir) or ""

        return cls(
            config=config,
            llm=llm,
            session=session,
            builtin_args=BuiltinSystemPromptArgs(
                KIMI_NOW=datetime.now().astimezone().isoformat(),
                KIMI_WORK_DIR=work_dir,
                KIMI_WORK_DIR_LS=ls_output,
                KIMI_AGENTS_MD=agents_md,
            ),
            denwa_renji=DenwaRenji(),
            approval=Approval(yolo=yolo),
        )
