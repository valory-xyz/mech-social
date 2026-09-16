# mech-social

Mech tools that interact with social media APIs, such as X.

This repository follows the layout of
[mech-predict](https://github.com/valory-xyz/mech-predict). It contains the
`mech_social` agent and service. The third-party packages, such as the mech
skills, contracts, and connections, come from a pinned
[mech](https://github.com/valory-xyz/mech) release. The repository doesn't
fork mech. The `upstream_pins` setting in `pyproject.toml` names the release.

## Requirements

- [Python](https://www.python.org/) `3.10` or later
- [uv](https://docs.astral.sh/uv/)
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Tendermint](https://docs.tendermint.com/v0.34/introduction/install.html) `==0.34.19`

## Set up your environment

1. Install the dependencies:

    ```bash
    uv sync
    source .venv/bin/activate
    ```

2. Fetch the third-party packages:

    ```bash
    autonomy packages sync --update-packages
    ```

## Add a tool

1. Create the tool under `packages/<author>/customs/<tool_name>/`.
2. Add the tool to the `customs` list in
   `packages/valory/agents/mech_social/aea-config.yaml`.
3. Update the package hashes:

    ```bash
    autonomy packages lock
    ```

## Update the mech packages

1. Copy the `third_party` hashes from `packages/packages.json` in the target
   [mech release](https://github.com/valory-xyz/mech/releases).
2. Update `upstream_pins` in `pyproject.toml` to the same release.
3. Run `autonomy packages sync --update-packages` and
   `autonomy packages lock`.

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md).
