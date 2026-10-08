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
"""Generate the metadata.json a mech publishes for the tools in this repo."""

import argparse
import copy
import importlib.util
import json
import re
import urllib.parse
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT_DIR = "./packages"
CUSTOMS = "customs"
METADATA_FILE_PATH = "metadata.json"
COMPONENT_YAML = "component.yaml"
ENTRY_POINT = "entry_point"
ALLOWED_TOOLS = "ALLOWED_TOOLS"
OUTPUT_KEYS = "OUTPUT_KEYS"
SCHEMA_REGISTRY_PATH = Path(__file__).parent / "tool_schemas.yaml"
# Every Valory operated mech must identify the Mech Terms in its metadata.
TERMS_URL = "https://www.valory.xyz/terms/mechs"
DEFAULT_BENCHMARK_METRIC = "accuracy"
BENCHMARK_WINDOWS: Tuple[str, ...] = ("7d", "30d", "90d", "all")
MAX_HOSTNAME_LENGTH = 253
# One lowercase ASCII DNS label: letters, digits and inner hyphens, 1 to 63 characters.
HOSTNAME_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
METADATA_TEMPLATE: Dict[str, Any] = {
    "description": "The mech executes AI tasks requested on-chain and delivers the results to the requester.",
    "inputFormat": "ipfs-v0.1",
    "outputFormat": "ipfs-v0.1",
    "image": "tbd",
    "termsUrl": TERMS_URL,
    "tools": [],
    "toolMetadata": {},
}


def find_customs_folders(packages_root: Path) -> List[Path]:
    """Find all the customs folders inside the packages dir."""
    return sorted(
        p for p in packages_root.rglob("*") if p.is_dir() and p.name == CUSTOMS
    )


def import_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    """Import a py file as a module."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module '{module_name}' from '{file_path}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_tool_folder(sub: Path) -> Optional[Dict[str, Any]]:
    """Read a tool package: its component.yaml and its ALLOWED_TOOLS."""
    yaml_path = sub / COMPONENT_YAML
    if not yaml_path.is_file():
        print(f"Skipping {sub}: no {COMPONENT_YAML}")
        return None
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    py_path = sub / data[ENTRY_POINT]
    module = import_module_from_path(
        f"{data['author']}_{sub.name}_{py_path.stem}", py_path
    )
    tools = getattr(module, ALLOWED_TOOLS, None)
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{py_path} does not define a non-empty {ALLOWED_TOOLS}")
    output_keys = getattr(module, OUTPUT_KEYS, None)
    if output_keys is not None and not (
        isinstance(output_keys, (list, tuple))
        and all(isinstance(key, str) for key in list(output_keys))
    ):
        raise ValueError(f"{py_path} {OUTPUT_KEYS} must be a list or tuple of strings")
    return {
        "tool_name": data["name"],
        "description": data["description"],
        "allowed_tools": tools,
        "output_keys": output_keys,
    }


def generate_tools_data(packages_root: Path) -> List[Dict[str, Any]]:
    """Read every tool package under the packages dir."""
    tools_data: List[Dict[str, Any]] = []
    for folder in find_customs_folders(packages_root):
        print(f"Matched folder: {folder}")
        for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
            entry = parse_tool_folder(sub)
            if entry:
                tools_data.append(entry)
    return tools_data


def load_schema_registry(path: Path) -> Dict[str, Any]:
    """Load the schema registry and check that every tool maps to a known kind."""
    reg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = reg.get("defaults") or {}
    tool_kinds = reg.get("tool_kinds") or {}
    for kind, schemas in defaults.items():
        if "input" not in schemas or "output" not in schemas:
            raise ValueError(
                f"Schema registry kind '{kind}' missing 'input' or 'output'"
            )
    for wire_name, kind in tool_kinds.items():
        if kind not in defaults:
            raise ValueError(
                f"Schema registry maps '{wire_name}' to unknown kind '{kind}'; "
                f"known kinds: {sorted(defaults)}"
            )
    return {"defaults": defaults, "tool_kinds": tool_kinds}


def check_example_keys(tool: str, schemas: Dict[str, Any], output_keys: Any) -> None:
    """Fail when a published result example does not list the tool's output keys."""
    if output_keys is None:
        print(f"'{tool}' has no {OUTPUT_KEYS}: its result example is not checked")
        return
    properties = (schemas["output"].get("schema") or {}).get("properties") or {}
    example = (properties.get("result") or {}).get("example")
    if example is not None and list(json.loads(example)) != list(output_keys):
        raise ValueError(
            f"'{tool}': the result example keys do not match the tool's {OUTPUT_KEYS}"
        )


def build_tools_metadata(
    tools_data: List[Dict[str, Any]],
    registry: Dict[str, Any],
    template: Dict[str, Any],
    skip_tools: List[str],
) -> Dict[str, Any]:
    """Build the metadata.json content from the tools data."""
    result: Dict[str, Any] = copy.deepcopy(template)
    for entry in tools_data:
        for tool in entry["allowed_tools"]:
            if tool in skip_tools:
                print(f"Skipping tool (via --skip-tool): {tool}")
                continue
            if tool in result["toolMetadata"]:
                raise ValueError(
                    f"Duplicate wire name '{tool}' found in '{entry['tool_name']}'"
                )
            # no fallback kind: a tool without an entry would publish a
            # schema that does not describe it
            kind = registry["tool_kinds"].get(tool)
            if kind is None:
                raise ValueError(
                    f"'{tool}' has no kind in the schema registry tool_kinds"
                )
            schemas = registry["defaults"][kind]
            check_example_keys(tool, schemas, entry.get("output_keys"))
            result["tools"].append(tool)
            result["toolMetadata"][tool] = {
                "name": entry["tool_name"],
                "description": entry["description"],
                "input": schemas["input"],
                "output": schemas["output"],
            }
    return result


def _require_text(flag: str, value: str) -> str:
    """Return the value when it has non-whitespace content; blank is an error."""
    if not value.strip():
        raise ValueError(f"{flag} must not be blank")
    return value


def validate_operator_domain(domain: str) -> str:
    """Return the domain if it is a lowercase bare hostname; reject, never normalise."""
    labels = domain.split(".")
    if (
        len(domain) > MAX_HOSTNAME_LENGTH
        or len(labels) < 2
        or not all(HOSTNAME_LABEL.fullmatch(label) for label in labels)
    ):
        raise ValueError(
            "--operator-domain must be a lowercase bare hostname such as 'valory.xyz' "
            "(at least one dot, no scheme, path, port or trailing dot, at most "
            f"{MAX_HOSTNAME_LENGTH} characters), got {domain!r}"
        )
    return domain


def build_operator(
    name: Optional[str], domain: Optional[str], contact: Optional[str]
) -> Optional[Dict[str, str]]:
    """Build the operator block, or None when no operator flag was given."""
    if name is None and domain is None and contact is None:
        return None
    if name is None or domain is None:
        raise ValueError(
            "--operator-name and --operator-domain are both required "
            "to emit an operator block"
        )
    operator = {
        "name": _require_text("--operator-name", name),
        "domain": validate_operator_domain(domain),
    }
    if contact is not None:
        operator["contact"] = _require_text("--operator-contact", contact)
    return operator


def is_https_url(url: str) -> bool:
    """Return True for an https:// URL with a host and no whitespace."""
    if any(char.isspace() for char in url):
        return False
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme == "https" and bool(parsed.netloc)


def build_benchmark(
    metric: Optional[str], window: Optional[str], url: Optional[str]
) -> Optional[Dict[str, str]]:
    """Return the benchmark link shared by every tool, or None when none was asked for."""
    if url is None:
        if metric is not None or window is not None:
            raise ValueError(
                "--benchmark-url is required when any other --benchmark-* flag is given"
            )
        return None
    if window is None:
        raise ValueError("--benchmark-window is required when --benchmark-url is given")
    if not is_https_url(url):
        raise ValueError(
            f"--benchmark-url must be an https URL with a host, got {url!r}"
        )
    if metric is None:
        metric = DEFAULT_BENCHMARK_METRIC
    return {
        "metric": _require_text("--benchmark-metric", metric),
        "window": window,
        "url": url,
    }


def attach_benchmarks(
    metadata: Dict[str, Any], benchmark: Optional[Dict[str, str]]
) -> None:
    """Give every tool the shared benchmark link; the live figure is behind its url."""
    if benchmark is None:
        return
    for entry in metadata["toolMetadata"].values():
        entry["benchmark"] = dict(benchmark)


def build_template(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the top-level manifest fields from the CLI arguments."""
    fixed = copy.deepcopy(METADATA_TEMPLATE)
    template: Dict[str, Any] = {
        "name": _require_text("--name", args.name),
        "description": args.description,
        "inputFormat": fixed["inputFormat"],
        "outputFormat": fixed["outputFormat"],
        "image": args.image,
    }
    if args.url:
        template["url"] = args.url
    template["termsUrl"] = args.terms_url
    operator = build_operator(
        args.operator_name, args.operator_domain, args.operator_contact
    )
    if operator:
        template["operator"] = operator
    template["tools"] = fixed["tools"]
    template["toolMetadata"] = fixed["toolMetadata"]
    return template


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate metadata.json from custom tool packages."
    )
    parser.add_argument("--packages-root", type=Path, default=Path(ROOT_DIR))
    parser.add_argument("--output", type=Path, default=Path(METADATA_FILE_PATH))
    parser.add_argument(
        "--name", type=str, required=True, help="Human-readable name of the mech."
    )
    parser.add_argument(
        "--description", type=str, default=METADATA_TEMPLATE["description"]
    )
    parser.add_argument("--image", type=str, default=METADATA_TEMPLATE["image"])
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Off-chain request endpoint of the mech; omitted when not given.",
    )
    parser.add_argument("--terms-url", type=str, default=METADATA_TEMPLATE["termsUrl"])
    parser.add_argument(
        "--operator-name", type=str, default=None, help="Operator, free text."
    )
    parser.add_argument(
        "--operator-domain",
        type=str,
        default=None,
        help="Operator's bare hostname, e.g. valory.xyz; serves the domain proof.",
    )
    parser.add_argument(
        "--operator-contact", type=str, default=None, help="Operator contact."
    )
    parser.add_argument(
        "--benchmark-metric",
        type=str,
        default=None,
        help=f"Benchmark metric shared by every tool (default {DEFAULT_BENCHMARK_METRIC}).",
    )
    parser.add_argument(
        "--benchmark-window",
        choices=BENCHMARK_WINDOWS,
        default=None,
        help="Benchmark window shared by every tool.",
    )
    parser.add_argument(
        "--benchmark-url",
        type=str,
        default=None,
        help="https analytics endpoint covering every tool of this mech.",
    )
    parser.add_argument(
        "--skip-tool",
        action="append",
        default=[],
        metavar="WIRE_NAME",
        help="Exclude this tool from the output (repeatable).",
    )
    parser.add_argument("--schema-registry", type=Path, default=SCHEMA_REGISTRY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Run the generate_metadata script."""
    args = parse_args(argv)

    template = build_template(args)
    benchmark = build_benchmark(
        args.benchmark_metric, args.benchmark_window, args.benchmark_url
    )
    registry = load_schema_registry(args.schema_registry)
    tools_data = generate_tools_data(args.packages_root)

    metadata = build_tools_metadata(tools_data, registry, template, args.skip_tool)
    attach_benchmarks(metadata, benchmark)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata has been stored to {args.output}")


if __name__ == "__main__":
    main()
