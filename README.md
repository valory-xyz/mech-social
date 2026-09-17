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

4. Add each third-party service that the tool calls to the
   [Integrations](#integrations) section.

## Integrations

The tools in this repository call third-party services. The service operator
supplies the API keys through the `API_KEYS` environment variable, which maps
each key name to a list of keys. The repository contains no keys.

The following table lists the services that the `token_social_sentiment` tool
calls:

| Service | Purpose | Key name in `API_KEYS` | If the service fails |
| --- | --- | --- | --- |
| [X API](https://docs.x.com/x-api) (`api.x.com`) | Recent post search and post counts | `x_bearer` | The tool continues with news only and reports `x`, `x_partial`, or `x_counts` in `degraded_sources`. |
| [Serper](https://serper.dev/) (`google.serper.dev`) | News headlines | `serperapi` | The tool continues with X posts only and reports `news` in `degraded_sources`. |
| [DexScreener](https://docs.dexscreener.com/) (`api.dexscreener.com`) | Token symbol, address, and chain verification | None | The tool continues with unverified input and reports `dexscreener` in `degraded_sources`. |
| [OpenAI API](https://platform.openai.com/docs) | Token extraction from free text and sentiment labels | `openai` | The request fails with an `llm_error` error. Without a key, the request fails with an `internal` error. |

If both X and Serper fail, or if the `API_KEYS` variable has neither an
`x_bearer` key nor a `serperapi` key, the request fails with a
`source_unavailable` error.

None of these services has a fallback provider. The tool doesn't hold or move
funds. Review this list at each release.

## Update the mech packages

1. Copy the `third_party` hashes from `packages/packages.json` in the target
   [mech release](https://github.com/valory-xyz/mech/releases).
2. Update `upstream_pins` in `pyproject.toml` to the same release.
3. Run `autonomy packages sync --update-packages` and
   `autonomy packages lock`.

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md).
