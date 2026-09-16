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
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

import yaml

ROOT_DIR = "./packages"
CUSTOMS = "customs"
METADATA_FILE_PATH = "metadata.json"
COMPONENT_YAML = "component.yaml"
ENTRY_POINT = "entry_point"
ALLOWED_TOOLS = "ALLOWED_TOOLS"
SCHEMA_REGISTRY_PATH = Path(__file__).parent / "tool_schemas.yaml"
# Every Valory operated mech must identify the Mech Terms in its metadata.
TERMS_URL = "https://www.valory.xyz/terms/mechs"
METADATA_TEMPLATE: Dict[str, Any] = {
    "name": "Autonolas Mech III",
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
    return {
        "tool_name": data["name"],
        "description": data["description"],
        "allowed_tools": tools,
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
            result["tools"].append(tool)
            result["toolMetadata"][tool] = {
                "name": entry["tool_name"],
                "description": entry["description"],
                "input": schemas["input"],
                "output": schemas["output"],
            }
    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate metadata.json from custom tool packages."
    )
    parser.add_argument("--packages-root", type=Path, default=Path(ROOT_DIR))
    parser.add_argument("--output", type=Path, default=Path(METADATA_FILE_PATH))
    parser.add_argument("--name", type=str, default=METADATA_TEMPLATE["name"])
    parser.add_argument(
        "--description", type=str, default=METADATA_TEMPLATE["description"]
    )
    parser.add_argument("--image", type=str, default=METADATA_TEMPLATE["image"])
    parser.add_argument("--terms-url", type=str, default=METADATA_TEMPLATE["termsUrl"])
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
    registry = load_schema_registry(args.schema_registry)
    tools_data = generate_tools_data(args.packages_root)

    template = copy.deepcopy(METADATA_TEMPLATE)
    template["name"] = args.name
    template["description"] = args.description
    template["image"] = args.image
    template["termsUrl"] = args.terms_url

    metadata = build_tools_metadata(tools_data, registry, template, args.skip_tool)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata has been stored to {args.output}")


if __name__ == "__main__":
    main()
