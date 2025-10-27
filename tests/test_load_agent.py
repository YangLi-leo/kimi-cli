"""Tests for agent loading functionality."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from kimi_cli.config import Config
from kimi_cli.metadata import Session
from kimi_cli.soul.agent import _load_system_prompt, _load_tools, load_agent
from kimi_cli.soul.approval import Approval
from kimi_cli.soul.denwarenji import DenwaRenji
from kimi_cli.soul.globals import AgentGlobals, BuiltinSystemPromptArgs
from kimi_cli.soul.toolset import CustomToolset


def test_load_system_prompt(system_prompt_file: Path, builtin_args: BuiltinSystemPromptArgs):
    """Test loading system prompt with template substitution."""
    prompt = _load_system_prompt(system_prompt_file, {"CUSTOM_ARG": "test_value"}, builtin_args)

    assert "Test system prompt with " in prompt
    assert "1970-01-01" in prompt  # Should contain the actual timestamp
    assert builtin_args.KIMI_NOW in prompt
    assert "test_value" in prompt


def test_load_tools_valid(agent_globals: AgentGlobals):
    """Test loading valid tools."""
    tool_paths = ["kimi_cli.tools.think:Think", "kimi_cli.tools.bash:Bash"]
    toolset = CustomToolset()
    bad_tools = _load_tools(
        toolset,
        tool_paths,
        {
            AgentGlobals: agent_globals,
            Config: agent_globals.config,
            BuiltinSystemPromptArgs: agent_globals.builtin_args,
            Session: agent_globals.session,
            DenwaRenji: agent_globals.denwa_renji,
            Approval: agent_globals.approval,
        },
    )

    assert len(bad_tools) == 0
    assert toolset is not None


def test_load_tools_invalid(agent_globals: AgentGlobals):
    """Test loading with invalid tool paths."""
    tool_paths = ["kimi_cli.tools.nonexistent:Tool", "kimi_cli.tools.think:Think"]
    toolset = CustomToolset()
    bad_tools = _load_tools(
        toolset,
        tool_paths,
        {
            AgentGlobals: agent_globals,
            Config: agent_globals.config,
            BuiltinSystemPromptArgs: agent_globals.builtin_args,
            Session: agent_globals.session,
            DenwaRenji: agent_globals.denwa_renji,
            Approval: agent_globals.approval,
        },
    )

    assert len(bad_tools) == 1
    assert "kimi_cli.tools.nonexistent:Tool" in bad_tools


@pytest.mark.asyncio
async def test_load_agent_invalid_tools(
    agent_file_invalid_tools: Path, agent_globals: AgentGlobals
):
    """Test loading agent with invalid tools raises ValueError."""
    with pytest.raises(ValueError, match="Invalid tools"):
        await load_agent(agent_file_invalid_tools, agent_globals, mcp_configs=[])


@pytest.fixture
def agent_file_invalid_tools() -> Generator[Path, Any, Any]:
    """Create an agent configuration file with invalid tools."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create system.md
        system_md = tmpdir / "system.md"
        system_md.write_text("You are a test agent")

        # Create agent.yaml with invalid tools
        agent_yaml = tmpdir / "agent.yaml"
        agent_yaml.write_text("""
version: 1
agent:
  name: "Test Agent"
  system_prompt_path: ./system.md
  tools: ["kimi_cli.tools.nonexistent:Tool"]
""")

        yield agent_yaml


@pytest.fixture
def system_prompt_file() -> Generator[Path, Any, Any]:
    """Create a system prompt file with template variables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        system_md = tmpdir / "system.md"
        system_md.write_text("Test system prompt with ${KIMI_NOW} and ${CUSTOM_ARG}")

        yield system_md
