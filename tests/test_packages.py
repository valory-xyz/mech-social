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

"""Smoke tests for the mech-social package registry."""

import json
from pathlib import Path

PACKAGES_JSON = Path(__file__).parent.parent / "packages" / "packages.json"


def test_dev_packages_contain_agent_and_service() -> None:
    """The dev registry must ship the mech_social agent and service."""
    dev = json.loads(PACKAGES_JSON.read_text(encoding="utf-8"))["dev"]
    assert "agent/valory/mech_social/0.1.0" in dev
    assert "service/valory/mech_social/0.1.0" in dev
