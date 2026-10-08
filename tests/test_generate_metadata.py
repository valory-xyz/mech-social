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
import textwrap
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
from scripts import generate_metadata

from packages.valory.customs.token_social_sentiment import (
    token_social_sentiment as tool,
)

PACKAGES_ROOT = Path(__file__).parent.parent / "packages"
TOOL = "token_social_sentiment"
SAMPLE_NAME = "Test Mech"
SAMPLE_URL = "https://mech.example.test"
SAMPLE_OPERATOR = ("Valory", "valory.xyz", "mechs@valory.xyz")
SAMPLE_BENCHMARK_URL = "https://analytics.example.test/v1/metrics/mech/100/0xabc"
SAMPLE_TOOL = "sentiment_alpha"
OTHER_TOOL = "sentiment_beta"
SAMPLE_KIND = "sample_kind"
TOOL_SCHEMA_FIELDS = ("name", "description", "input", "output")
TOOL_DESCRIPTION = "Measures things."
BASE_ARGS = ("--name", SAMPLE_NAME)
BENCHMARK_ARGS = (
    "--benchmark-window",
    "30d",
    "--benchmark-url",
    SAMPLE_BENCHMARK_URL,
)


def _write_tool_package(packages_root: Path, wire_name: str) -> None:
    """Create a minimal custom tool package that registers one wire name."""
    tool_dir = packages_root / "author" / "customs" / wire_name
    tool_dir.mkdir(parents=True)
    (tool_dir / "component.yaml").write_text(
        textwrap.dedent(f"""\
            name: {wire_name}
            author: author
            description: {TOOL_DESCRIPTION}
            entry_point: {wire_name}.py
            """),
        encoding="utf-8",
    )
    (tool_dir / f"{wire_name}.py").write_text(
        f'ALLOWED_TOOLS = ["{wire_name}"]\n', encoding="utf-8"
    )


def _write_schema_registry(path: Path, wire_names: Tuple[str, ...]) -> None:
    """Write a registry that maps every wire name to one minimal kind."""
    registry = {
        "defaults": {SAMPLE_KIND: {"input": {"type": "text"}, "output": {}}},
        "tool_kinds": {wire_name: SAMPLE_KIND for wire_name in wire_names},
    }
    path.write_text(json.dumps(registry), encoding="utf-8")


def _generate(
    tmp_path: Path, *args: str, tools: Tuple[str, ...] = ()
) -> Dict[str, Any]:
    """Run the generator on this repo's packages, or on fake packages for `tools`."""
    output = tmp_path / "metadata.json"
    argv = ["--output", str(output), *args]
    if tools:
        packages_root = tmp_path / "packages"
        registry = tmp_path / "tool_schemas.yaml"
        for wire_name in tools:
            _write_tool_package(packages_root, wire_name)
        _write_schema_registry(registry, tools)
        argv += ["--packages-root", str(packages_root)]
        argv += ["--schema-registry", str(registry)]
    else:
        argv += ["--packages-root", str(PACKAGES_ROOT)]
    generate_metadata.main(argv)
    return json.loads(output.read_text(encoding="utf-8"))


def _kind_schemas() -> Dict[str, Any]:
    """Return the registry schemas of the tool's kind."""
    registry = generate_metadata.load_schema_registry(
        generate_metadata.SCHEMA_REGISTRY_PATH
    )
    return registry["defaults"][registry["tool_kinds"][TOOL]]


def test_tool_is_published_with_its_kind_schemas(tmp_path: Path) -> None:
    """The tool gets its kind's schemas; without --url there is no url key."""
    metadata = _generate(tmp_path, *BASE_ARGS)
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


def test_name_is_required(tmp_path: Path) -> None:
    """There is no default mech name; omitting --name is a usage error."""
    with pytest.raises(SystemExit):
        _generate(tmp_path)
    assert "name" not in generate_metadata.METADATA_TEMPLATE


def test_name_flag_reaches_the_output(tmp_path: Path) -> None:
    """--name is the only source of the manifest name."""
    assert _generate(tmp_path, *BASE_ARGS)["name"] == SAMPLE_NAME


def test_generated_metadata_carries_the_terms_link_by_default(tmp_path: Path) -> None:
    """A regenerate-from-source keeps termsUrl; nothing has to add it by hand."""
    metadata = _generate(tmp_path, *BASE_ARGS)
    assert (
        metadata["termsUrl"]
        == generate_metadata.TERMS_URL
        == "https://www.valory.xyz/terms/mechs"
    )


def test_terms_url_flag_overrides_the_default(tmp_path: Path) -> None:
    """--terms-url reaches the output, like --name and --image do."""
    metadata = _generate(tmp_path, *BASE_ARGS, "--terms-url", "https://example.test/t")
    assert metadata["termsUrl"] == "https://example.test/t"


@pytest.mark.parametrize("field", ["description", "image", "termsUrl"])
def test_template_fixed_fields_reach_the_output(tmp_path: Path, field: str) -> None:
    """Each fixed template field reaches the output."""
    metadata = _generate(tmp_path, *BASE_ARGS)
    assert metadata[field] == generate_metadata.METADATA_TEMPLATE[field]


def test_url_flag_reaches_the_output(tmp_path: Path) -> None:
    """--url carries the off-chain endpoint, so a regenerate keeps it."""
    assert _generate(tmp_path, *BASE_ARGS, "--url", SAMPLE_URL)["url"] == SAMPLE_URL


def test_operator_block_reaches_the_output(tmp_path: Path) -> None:
    """The three operator flags land under one `operator` object."""
    name, domain, contact = SAMPLE_OPERATOR
    metadata = _generate(
        tmp_path,
        *BASE_ARGS,
        "--operator-name",
        name,
        "--operator-domain",
        domain,
        "--operator-contact",
        contact,
    )
    assert metadata["operator"] == {"name": name, "domain": domain, "contact": contact}


def test_operator_contact_is_optional(tmp_path: Path) -> None:
    """Without --operator-contact the block has no contact key."""
    name, domain, _ = SAMPLE_OPERATOR
    metadata = _generate(
        tmp_path, *BASE_ARGS, "--operator-name", name, "--operator-domain", domain
    )
    assert metadata["operator"] == {"name": name, "domain": domain}


def test_operator_is_omitted_when_no_operator_flag_is_given(tmp_path: Path) -> None:
    """No operator flags means no operator block."""
    assert "operator" not in _generate(tmp_path, *BASE_ARGS)


@pytest.mark.parametrize("name", ["", "   "])
def test_blank_name_is_rejected(tmp_path: Path, name: str) -> None:
    """--name must carry text; whitespace does not satisfy the requirement."""
    with pytest.raises(ValueError, match="--name must not be blank"):
        _generate(tmp_path, "--name", name)


@pytest.mark.parametrize(
    "flags",
    [
        ("--operator-name", "Valory"),
        ("--operator-domain", "valory.xyz"),
        ("--operator-contact", "mechs@valory.xyz"),
        ("--operator-name", "Valory", "--operator-contact", "mechs@valory.xyz"),
    ],
)
def test_operator_requires_both_name_and_domain(
    tmp_path: Path, flags: Tuple[str, ...]
) -> None:
    """A partial operator block is an error, not a silently incomplete one."""
    with pytest.raises(ValueError, match="--operator-name and --operator-domain"):
        _generate(tmp_path, *BASE_ARGS, *flags)


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--operator-name", ""),
        ("--operator-name", "  "),
        ("--operator-contact", ""),
        ("--operator-contact", " \t"),
    ],
)
def test_blank_operator_fields_are_rejected(
    tmp_path: Path, flag: str, value: str
) -> None:
    """A blank operator name or contact is an error, not dropped or written as-is."""
    name, domain, _ = SAMPLE_OPERATOR
    with pytest.raises(ValueError, match=f"{flag} must not be blank"):
        _generate(
            tmp_path,
            *BASE_ARGS,
            "--operator-name",
            name,
            "--operator-domain",
            domain,
            flag,
            value,
        )


@pytest.mark.parametrize(
    "domain",
    [
        "https://valory.xyz",
        "valory.xyz/",
        "valory.xyz/.well-known",
        "valory.xyz.",
        ".valory.xyz",
        "valory.xyz:443",
        "valory",
        "val ory.xyz",
        "Valory.xyz",
        "VALORY.XYZ",
        "valory.\u212ayz",
        "valory-.xyz",
        "valory..xyz",
        "",
        ".".join(["a" * 63] * 3 + ["a" * 62]),
    ],
)
def test_operator_domain_rejects_anything_but_a_lowercase_bare_hostname(
    tmp_path: Path, domain: str
) -> None:
    """Scheme, path, port, dots at either end, upper case, non-ASCII, edge hyphens and length are rejected."""
    with pytest.raises(ValueError, match="lowercase bare hostname"):
        _generate(
            tmp_path,
            *BASE_ARGS,
            "--operator-name",
            "Valory",
            "--operator-domain",
            domain,
        )


def test_validate_operator_domain_rejects_a_leading_hyphen_label() -> None:
    """A label starting with a hyphen is rejected by the validator itself."""
    with pytest.raises(ValueError, match="lowercase bare hostname"):
        generate_metadata.validate_operator_domain("-valory.xyz")


@pytest.mark.parametrize(
    "domain",
    [
        "valory.xyz",
        "mechs.valory.xyz",
        "a-b.co.uk",
        "x.y",
        ".".join(["a" * 63] * 3 + ["a" * 61]),
    ],
)
def test_operator_domain_accepts_lowercase_bare_hostnames(
    tmp_path: Path, domain: str
) -> None:
    """A lowercase dotted hostname of at most 253 characters passes through unchanged."""
    assert len(domain) <= generate_metadata.MAX_HOSTNAME_LENGTH
    metadata = _generate(
        tmp_path, *BASE_ARGS, "--operator-name", "Valory", "--operator-domain", domain
    )
    assert metadata["operator"]["domain"] == domain


def test_benchmark_url_and_window_alone_give_every_tool_a_benchmark(
    tmp_path: Path,
) -> None:
    """The url is the live figure, so no per-tool value is needed to emit a benchmark."""
    metadata = _generate(
        tmp_path, *BASE_ARGS, *BENCHMARK_ARGS, tools=(SAMPLE_TOOL, OTHER_TOOL)
    )
    for wire_name in (SAMPLE_TOOL, OTHER_TOOL):
        benchmark = metadata["toolMetadata"][wire_name]["benchmark"]
        assert list(benchmark) == ["metric", "window", "url"]
        assert benchmark == {
            "metric": generate_metadata.DEFAULT_BENCHMARK_METRIC,
            "window": "30d",
            "url": SAMPLE_BENCHMARK_URL,
        }


def test_tool_entry_keeps_schema_fields_alongside_benchmark(tmp_path: Path) -> None:
    """A benchmark is added to the tool entry without displacing its schema fields."""
    metadata = _generate(
        tmp_path,
        *BASE_ARGS,
        *BENCHMARK_ARGS,
        "--benchmark-value",
        f"{SAMPLE_TOOL}=0.83",
        tools=(SAMPLE_TOOL,),
    )
    entry = metadata["toolMetadata"][SAMPLE_TOOL]
    assert metadata["tools"] == [SAMPLE_TOOL]
    assert set(entry) == {*TOOL_SCHEMA_FIELDS, "benchmark"}
    assert entry["name"] == SAMPLE_TOOL
    assert entry["description"] == TOOL_DESCRIPTION
    assert entry["input"] == {"type": "text"}
    assert entry["output"] == {}
    assert list(entry["benchmark"]) == ["metric", "value", "window", "url"]
    assert entry["benchmark"] == {
        "metric": generate_metadata.DEFAULT_BENCHMARK_METRIC,
        "value": 0.83,
        "window": "30d",
        "url": SAMPLE_BENCHMARK_URL,
    }


def test_benchmark_metric_flag_overrides_the_default(tmp_path: Path) -> None:
    """--benchmark-metric replaces the default metric name."""
    metadata = _generate(
        tmp_path,
        *BASE_ARGS,
        *BENCHMARK_ARGS,
        "--benchmark-metric",
        "brier",
        tools=(SAMPLE_TOOL,),
    )
    assert metadata["toolMetadata"][SAMPLE_TOOL]["benchmark"]["metric"] == "brier"


@pytest.mark.parametrize("metric", ["", "  "])
def test_blank_benchmark_metric_is_rejected(tmp_path: Path, metric: str) -> None:
    """A blank metric is an error, not an empty string in the manifest."""
    with pytest.raises(ValueError, match="--benchmark-metric must not be blank"):
        _generate(
            tmp_path,
            *BASE_ARGS,
            *BENCHMARK_ARGS,
            "--benchmark-metric",
            metric,
            tools=(SAMPLE_TOOL,),
        )


def test_tool_without_a_value_gets_the_benchmark_without_a_value_key(
    tmp_path: Path,
) -> None:
    """Only tools that were given a value carry one; the others still carry the link."""
    metadata = _generate(
        tmp_path,
        *BASE_ARGS,
        *BENCHMARK_ARGS,
        "--benchmark-value",
        f"{SAMPLE_TOOL}=0.83",
        tools=(SAMPLE_TOOL, OTHER_TOOL),
    )
    assert metadata["toolMetadata"][SAMPLE_TOOL]["benchmark"]["value"] == 0.83
    other = metadata["toolMetadata"][OTHER_TOOL]["benchmark"]
    assert "value" not in other
    assert other["url"] == SAMPLE_BENCHMARK_URL


def test_benchmark_value_for_a_tool_missing_from_the_output_raises(
    tmp_path: Path,
) -> None:
    """A mistyped or skipped tool name fails loudly instead of dropping the value."""
    with pytest.raises(ValueError, match="missing from the output"):
        _generate(
            tmp_path,
            *BASE_ARGS,
            *BENCHMARK_ARGS,
            "--benchmark-value",
            f"{OTHER_TOOL}=0.83",
            tools=(SAMPLE_TOOL,),
        )


def test_benchmark_value_given_twice_for_one_tool_raises(tmp_path: Path) -> None:
    """Two values for one tool is ambiguous; neither silently wins."""
    with pytest.raises(ValueError, match="given twice"):
        _generate(
            tmp_path,
            *BASE_ARGS,
            *BENCHMARK_ARGS,
            "--benchmark-value",
            f"{SAMPLE_TOOL}=0.8",
            "--benchmark-value",
            f"{SAMPLE_TOOL}=0.9",
            tools=(SAMPLE_TOOL,),
        )


@pytest.mark.parametrize(
    "raw",
    [
        f"{SAMPLE_TOOL}=-0.1",
        f"{SAMPLE_TOOL}=1.01",
        f"{SAMPLE_TOOL}=nan",
        f"{SAMPLE_TOOL}=high",
        f"{SAMPLE_TOOL}=",
        SAMPLE_TOOL,
        "=0.5",
    ],
)
def test_benchmark_value_must_be_tool_equals_unit_interval(
    tmp_path: Path, raw: str
) -> None:
    """Anything but TOOL=<number in 0..1> is a usage error."""
    with pytest.raises(SystemExit):
        _generate(
            tmp_path,
            *BASE_ARGS,
            *BENCHMARK_ARGS,
            "--benchmark-value",
            raw,
            tools=(SAMPLE_TOOL,),
        )


@pytest.mark.parametrize("value", ["0", "1", "0.5"])
def test_benchmark_value_accepts_the_unit_interval_bounds(
    tmp_path: Path, value: str
) -> None:
    """0 and 1 are valid values, not off-by-one rejections; a measured zero is written."""
    metadata = _generate(
        tmp_path,
        *BASE_ARGS,
        *BENCHMARK_ARGS,
        "--benchmark-value",
        f"{SAMPLE_TOOL}={value}",
        tools=(SAMPLE_TOOL,),
    )
    assert metadata["toolMetadata"][SAMPLE_TOOL]["benchmark"]["value"] == float(value)


def test_benchmark_window_rejects_unknown_values(tmp_path: Path) -> None:
    """The window is one of the spec's four literals."""
    with pytest.raises(SystemExit):
        _generate(tmp_path, *BASE_ARGS, "--benchmark-window", "14d")


@pytest.mark.parametrize(
    "flags",
    [
        ("--benchmark-window", "30d"),
        ("--benchmark-metric", "brier"),
        ("--benchmark-value", f"{SAMPLE_TOOL}=0.5"),
        ("--benchmark-window", "30d", "--benchmark-value", f"{SAMPLE_TOOL}=0.5"),
    ],
)
def test_benchmark_flags_without_url_are_an_error(
    tmp_path: Path, flags: Tuple[str, ...]
) -> None:
    """A benchmark flag that cannot reach the output is an error, not ignored."""
    with pytest.raises(ValueError, match="--benchmark-url is required"):
        _generate(tmp_path, *BASE_ARGS, *flags, tools=(SAMPLE_TOOL,))


def test_benchmark_url_without_window_is_an_error(tmp_path: Path) -> None:
    """The window says what the linked figure covers, so the url alone is incomplete."""
    with pytest.raises(ValueError, match="--benchmark-window is required"):
        _generate(
            tmp_path,
            *BASE_ARGS,
            "--benchmark-url",
            SAMPLE_BENCHMARK_URL,
            tools=(SAMPLE_TOOL,),
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://analytics.example.test/v1/metrics/mech/100/0xabc",
        "ftp://analytics.example.test/x",
        "analytics.example.test/v1/metrics",
        "foo",
        "https://",
        "https:// analytics.example.test/x",
        "",
    ],
)
def test_benchmark_url_must_be_https_with_a_host(tmp_path: Path, url: str) -> None:
    """Only an https URL with a host is a link the marketplace will show."""
    with pytest.raises(ValueError, match="https URL with a host"):
        _generate(
            tmp_path,
            *BASE_ARGS,
            "--benchmark-window",
            "30d",
            "--benchmark-url",
            url,
            tools=(SAMPLE_TOOL,),
        )


def test_regenerated_manifest_keeps_every_spec_field(tmp_path: Path) -> None:
    """One full invocation yields url, termsUrl, operator and a per-tool benchmark."""
    name, domain, contact = SAMPLE_OPERATOR
    metadata = _generate(
        tmp_path,
        *BASE_ARGS,
        "--url",
        SAMPLE_URL,
        "--operator-name",
        name,
        "--operator-domain",
        domain,
        "--operator-contact",
        contact,
        *BENCHMARK_ARGS,
        "--benchmark-value",
        f"{TOOL}=0.83",
    )
    assert list(metadata) == [
        "name",
        "description",
        "inputFormat",
        "outputFormat",
        "image",
        "url",
        "termsUrl",
        "operator",
        "tools",
        "toolMetadata",
    ]
    assert metadata["url"] == SAMPLE_URL
    assert metadata["termsUrl"] == generate_metadata.TERMS_URL
    assert metadata["operator"]["domain"] == domain
    assert metadata["toolMetadata"][TOOL]["benchmark"]["value"] == 0.83
    assert set(metadata["toolMetadata"][TOOL]) == {*TOOL_SCHEMA_FIELDS, "benchmark"}
