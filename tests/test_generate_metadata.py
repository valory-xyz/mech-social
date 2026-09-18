# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Tests for scripts/generate_metadata.py."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from scripts import generate_metadata

from packages.valory.customs.token_social_sentiment import (
    token_social_sentiment as tool,
)

PACKAGES_ROOT = Path(__file__).parent.parent / "packages"
TOOL = "token_social_sentiment"


def _generate(tmp_path: Path, *args: str) -> Dict[str, Any]:
    """Run the generator on this repo's packages."""
    output = tmp_path / "metadata.json"
    generate_metadata.main(
        ["--packages-root", str(PACKAGES_ROOT), "--output", str(output), *args]
    )
    return json.loads(output.read_text(encoding="utf-8"))


def _kind_schemas() -> Dict[str, Any]:
    """Return the registry schemas of the tool's kind."""
    registry = generate_metadata.load_schema_registry(
        generate_metadata.SCHEMA_REGISTRY_PATH
    )
    return registry["defaults"][registry["tool_kinds"][TOOL]]


def test_tool_is_published_with_its_kind_schemas(tmp_path: Path) -> None:
    """The tool gets its kind's schemas; the per-mech url is added at deploy."""
    metadata = _generate(tmp_path)
    schemas = _kind_schemas()
    assert TOOL in metadata["tools"]
    assert metadata["toolMetadata"][TOOL]["input"] == schemas["input"]
    assert metadata["toolMetadata"][TOOL]["output"] == schemas["output"]
    assert "url" not in metadata


def test_result_example_has_the_tool_output_keys() -> None:
    """The result example lists the keys the tool returns, in order."""
    result = _kind_schemas()["output"]["schema"]["properties"]["result"]
    assert tuple(json.loads(result["example"])) == tool.OUTPUT_KEYS


def test_result_example_is_consistent_with_the_tool() -> None:
    """The published example carries the numbers the tool would compute."""
    result = _kind_schemas()["output"]["schema"]["properties"]["result"]
    example = json.loads(result["example"])
    breakdown = example["breakdown"]
    on_topic = sum(breakdown.values())
    assert example["sentiment"] == round(
        (breakdown["bullish"] - breakdown["bearish"]) / on_topic, 2
    )
    assert example["sentiment_interval"] == tool.sentiment_interval(breakdown)


def test_tool_without_kind_fails() -> None:
    """A wire name missing from tool_kinds fails instead of getting a default."""
    registry: Dict[str, Any] = {"defaults": {}, "tool_kinds": {}}
    entry = {"tool_name": "new_tool", "description": "", "allowed_tools": ["new"]}
    with pytest.raises(ValueError, match="has no kind"):
        generate_metadata.build_tools_metadata(
            [entry], registry, generate_metadata.METADATA_TEMPLATE, []
        )


def test_stale_result_example_fails() -> None:
    """An example whose keys differ from the tool's OUTPUT_KEYS is not published."""
    schemas = {
        "input": {},
        "output": {
            "schema": {"properties": {"result": {"example": '{"sentiment": 0.5}'}}}
        },
    }
    registry = {"defaults": {"kind": schemas}, "tool_kinds": {"new": "kind"}}
    entry = {
        "tool_name": "new_tool",
        "description": "",
        "allowed_tools": ["new"],
        "output_keys": ("sentiment", "error"),
    }
    with pytest.raises(ValueError, match="do not match"):
        generate_metadata.build_tools_metadata(
            [entry], registry, generate_metadata.METADATA_TEMPLATE, []
        )


def test_missing_output_keys_is_reported(capsys: Any) -> None:
    """A tool without OUTPUT_KEYS is published, and the skipped check is printed."""
    schemas: Dict[str, Any] = {"input": {}, "output": {}}
    registry = {"defaults": {"kind": schemas}, "tool_kinds": {"new": "kind"}}
    entry = {"tool_name": "new_tool", "description": "", "allowed_tools": ["new"]}
    metadata = generate_metadata.build_tools_metadata(
        [entry], registry, generate_metadata.METADATA_TEMPLATE, []
    )
    assert metadata["tools"] == ["new"]
    assert "has no OUTPUT_KEYS" in capsys.readouterr().out


def test_output_keys_must_be_strings(tmp_path: Path) -> None:
    """OUTPUT_KEYS that is not a list or tuple of strings fails like ALLOWED_TOOLS."""
    tool_dir = tmp_path / "bad_tool"
    tool_dir.mkdir()
    (tool_dir / "component.yaml").write_text(
        "name: bad_tool\nauthor: valory\ndescription: x\nentry_point: bad_tool.py\n",
        encoding="utf-8",
    )
    (tool_dir / "bad_tool.py").write_text(
        'ALLOWED_TOOLS = ["bad"]\nOUTPUT_KEYS = "token"\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must be a list or tuple of strings"):
        generate_metadata.parse_tool_folder(tool_dir)


def test_empty_output_keys_are_checked() -> None:
    """An empty OUTPUT_KEYS is checked against the example, not treated as missing."""
    schemas = {
        "input": {},
        "output": {
            "schema": {"properties": {"result": {"example": '{"sentiment": 0.5}'}}}
        },
    }
    registry = {"defaults": {"kind": schemas}, "tool_kinds": {"new": "kind"}}
    entry = {
        "tool_name": "new_tool",
        "description": "",
        "allowed_tools": ["new"],
        "output_keys": (),
    }
    with pytest.raises(ValueError, match="do not match"):
        generate_metadata.build_tools_metadata(
            [entry], registry, generate_metadata.METADATA_TEMPLATE, []
        )
