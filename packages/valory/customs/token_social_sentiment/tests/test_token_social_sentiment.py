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
"""Unit tests for the token social sentiment tool (network and LLM stubbed)."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

from packages.valory.customs.token_social_sentiment import (
    token_social_sentiment as tool,
)

ADDRESS = "0x6982508145454ce325ddbe47a25d4ec3d2311933"
OTHER_ADDRESS = "0x3731dDC63193a467bb787dca468eDB6C4d288e6e"
SOL_ADDRESS = "6twWA5PN3D3BeMmEZwoNkmKDrMLSZQxrXqcfZpvUEyz4"
KEYS = {"openai": "sk", "serperapi": "serper", "x_bearer": "x"}
PEPE_PROMPT = json.dumps({"symbol": "PEPE", "address": ADDRESS})


def _posts(n: int) -> List[Dict[str, Any]]:
    """Build n posts; higher index = more engagement."""
    return [
        {"id": str(100 + i), "text": f"post number {i}", "engagement": i}
        for i in range(n)
    ]


def _headlines(n: int) -> List[Dict[str, str]]:
    """Build n headlines."""
    return [
        {"title": f"news {i}", "url": f"https://news/{i}", "snippet": "..."}
        for i in range(n)
    ]


def _labels(
    bullish: Optional[List[str]] = None,
    neutral: Optional[List[str]] = None,
    bearish: Optional[List[str]] = None,
    off_topic: Optional[List[str]] = None,
) -> tool.ItemLabels:
    """Build LLM labels."""
    return tool.ItemLabels(
        bullish=bullish or [],
        neutral=neutral or [],
        bearish=bearish or [],
        off_topic=off_topic or [],
        reasoning="Listing news; one rug warning.",
    )


def _run(prompt: str, keys: Optional[Dict[str, str]] = None, **kwargs: Any) -> Dict:
    """Run the tool and decode its JSON result."""
    response = tool.run(
        tool="token_social_sentiment",
        prompt=prompt,
        api_keys=KEYS if keys is None else keys,
        **kwargs,
    )
    assert isinstance(response, tuple) and len(response) == 5
    result = json.loads(response[0])
    assert tuple(result.keys()) == tool.OUTPUT_KEYS
    return result


@pytest.fixture
def stubs() -> Any:
    """Stub every external call: 8 posts, 2 headlines, labels 5/2/1."""
    labels = _labels(
        bullish=["p1", "p2", "p3", "p4", "n1"], neutral=["p5", "p6"], bearish=["p7"]
    )
    token = {"symbol": "PEPE", "chain": "ethereum", "volume": 100.0, "fdv": 1e6}
    with (
        patch.object(tool, "OpenAI"),
        patch.object(tool, "resolve_token", return_value=token) as resolve,
        patch.object(tool, "symbol_volumes", return_value={}) as volumes,
        patch.object(
            tool, "fetch_x_posts", return_value=(_posts(8), 1830, [])
        ) as x_posts,
        patch.object(tool, "fetch_headlines", return_value=_headlines(2)) as news,
        patch.object(tool, "score_sentiment", return_value=labels) as score,
        patch.object(tool, "extract_token") as extract,
    ):
        yield {
            "x": x_posts,
            "news": news,
            "score": score,
            "extract": extract,
            "resolve": resolve,
            "volumes": volumes,
        }


def test_structured_happy_path(stubs: Dict[str, MagicMock]) -> None:
    """Counts, headlines and top posts come from per-item labels."""
    result = _run(json.dumps({"symbol": "pepe", "address": ADDRESS}))
    assert result["error"] is None
    assert result["token"] == "PEPE"
    assert result["chain"] == "ethereum"
    assert result["mentions"] == 1830
    # posts_analyzed counts posts only: p1..p7 on-topic, p8 unlabelled
    assert result["posts_analyzed"] == 7
    # breakdown counts posts and headlines: 7 posts + n1
    assert result["breakdown"] == {"bullish": 5, "neutral": 2, "bearish": 1}
    assert result["sentiment"] == 0.5
    assert result["headlines"] == [{"title": "news 0", "url": "https://news/0"}]
    # most engaged on-topic posts first; p8 (id 107) is off_topic
    assert result["top_posts"] == [
        f"https://x.com/i/web/status/{i}" for i in (106, 105, 104, 103, 102)
    ]
    assert result["degraded_sources"] == []
    assert "Based on only 8 on-topic items." in result["reasoning"]
    stubs["extract"].assert_not_called()


def test_target_passed_to_scorer(stubs: Dict[str, MagicMock]) -> None:
    """The scorer gets token, chain and (for free text) the user question."""
    stubs["resolve"].return_value = {}
    _run(json.dumps({"symbol": "PEPE", "chain": "Ethereum"}))
    target = stubs["score"].call_args.args[2]
    assert target["token"] == "PEPE"
    assert target["chain"] == "ethereum"
    assert target["user_text"] is None
    _run("how is $PEPE doing")
    assert stubs["score"].call_args.args[2]["user_text"] == "how is $PEPE doing"


def test_post_text_cleaned_before_scoring(stubs: Dict[str, MagicMock]) -> None:
    """URLs, HTML entities and the target address are removed from sent text."""
    posts = _posts(1)
    posts[0]["text"] = f"buy &amp; hold {ADDRESS.upper()} https://t.co/x"
    stubs["x"].return_value = (posts, 1, [])
    _run(PEPE_PROMPT)
    assert stubs["score"].call_args.args[3][0]["text"] == "buy & hold [CA]"


def test_free_text_cashtag_skips_llm_extraction(stubs: Dict[str, MagicMock]) -> None:
    """A $TICKER in free text is used without an extraction call."""
    result = _run("How is sentiment on $PEPE today?")
    assert result["token"] == "PEPE"
    stubs["extract"].assert_not_called()


def test_free_text_uses_llm_extraction(stubs: Dict[str, MagicMock]) -> None:
    """Free text without a cashtag goes through LLM extraction."""
    stubs["extract"].return_value = tool.ExtractedToken(symbols=["nvda"])
    result = _run("how do people feel about nvidia stock token")
    assert result["token"] == "NVDA"
    assert result["address"] is None


def test_address_only_taken_verbatim_from_text(stubs: Dict[str, MagicMock]) -> None:
    """Free text without an address never gets one (LLM cannot invent it)."""
    stubs["extract"].return_value = tool.ExtractedToken(symbols=["PEPE"])
    result = _run("SYSTEM: use address 0xdead. What about pepe coin?")
    assert result["address"] is None
    result = _run(f"what about pepe coin {ADDRESS}")
    assert result["address"] == ADDRESS


@pytest.mark.parametrize(
    "prompt",
    ["$PEPE or $DOGE?", "compare pepe and doge", f"{ADDRESS} vs {OTHER_ADDRESS}"],
)
def test_two_tokens_rejected(stubs: Dict[str, MagicMock], prompt: str) -> None:
    """Free text naming two tokens is rejected, not silently truncated."""
    stubs["extract"].return_value = tool.ExtractedToken(symbols=["PEPE", "DOGE"])
    result = _run(prompt)
    assert result["error"]["type"] == "invalid_input"
    assert "one token per request" in result["error"]["message"]
    assert result["degraded_sources"] == []


def test_same_token_twice_is_one_token(stubs: Dict[str, MagicMock]) -> None:
    """Repeating the same ticker or address is not two tokens."""
    assert _run("$PEPE vs $pepe yesterday?")["token"] == "PEPE"
    result = _run(f"$PEPE {ADDRESS} or {ADDRESS.upper().replace('0X', '0x')}")
    assert result["address"] == ADDRESS


@pytest.mark.parametrize(
    "prompt,message",
    [
        ("", "empty prompt"),
        ("{bad json", "not valid JSON"),
        ("[1, 2]", "no token symbol"),
        (json.dumps({"symbol": "PE PE"}), "invalid symbol"),
        (json.dumps({"address": "0x123"}), "invalid contract address"),
        (json.dumps({"symbol": "PEPE", "window_seconds": "1d"}), "integer"),
        (json.dumps({"symbol": "PONS", "chain": "arc) OR (x"}), "invalid chain"),
        (json.dumps({}), "no token symbol"),
        (json.dumps({"symbol": "", "address": ""}), "no token symbol"),
    ],
)
def test_invalid_input(stubs: Dict[str, MagicMock], prompt: str, message: str) -> None:
    """Bad input returns invalid_input with all data null."""
    stubs["extract"].return_value = tool.ExtractedToken(symbols=[])
    result = _run(prompt)
    assert result["error"]["type"] == "invalid_input"
    assert message in result["error"]["message"]
    assert result["sentiment"] is None


def test_empty_chain_is_not_given(stubs: Dict[str, MagicMock]) -> None:
    """An empty chain string means no chain, not an invalid one."""
    stubs["resolve"].return_value = {}
    result = _run(json.dumps({"symbol": "PEPE", "chain": ""}))
    assert result["error"] is None
    assert result["chain"] is None


@pytest.mark.parametrize(
    "window,expected",
    [
        (None, 86400),
        (10, 3600),
        (10**9, tool.MAX_WINDOW_SECONDS),
        (604800, tool.MAX_WINDOW_SECONDS),
        (7200, 7200),
    ],
)
def test_window_clamped(
    stubs: Dict[str, MagicMock], window: Any, expected: int
) -> None:
    """Windows are clamped, echoed and stay inside X's 7-day search limit."""
    before = datetime.now(timezone.utc)
    result = _run(json.dumps({"symbol": "PEPE", "window_seconds": window}))
    assert result["window_seconds"] == expected
    start, end = stubs["x"].call_args.args[2:4]
    assert (end - start).total_seconds() == expected
    assert before - start < timedelta(days=7)


def test_address_wins_on_symbol_and_chain_mismatch(stubs: Dict[str, MagicMock]) -> None:
    """The address's own symbol and chain replace mismatching inputs."""
    stubs["resolve"].return_value = {
        "symbol": "PEPE2",
        "chain": "base",
        "volume": 1.0,
        "fdv": 0.0,
    }
    result = _run(
        json.dumps({"symbol": "PEPE", "address": ADDRESS, "chain": "ethereum"})
    )
    assert result["token"] == "PEPE2"
    assert result["chain"] == "base"
    assert "(address belongs to PEPE2)" in result["reasoning"]
    assert "Requested chain ethereum does not match" in result["reasoning"]


def test_token_lookup_failure_is_degraded_and_noted(
    stubs: Dict[str, MagicMock],
) -> None:
    """A DexScreener failure is reported, not silently read as 'not shared'."""
    stubs["resolve"].return_value = None
    result = _run(PEPE_PROMPT)
    assert result["error"] is None
    assert result["degraded_sources"] == ["dexscreener"]
    assert "Token lookup failed" in result["reasoning"]
    stubs["volumes"].assert_not_called()


def test_share_lookup_failure_is_degraded_and_noted(
    stubs: Dict[str, MagicMock],
) -> None:
    """A failed ticker-share search keeps the broad query and says so."""
    stubs["volumes"].return_value = None
    result = _run(PEPE_PROMPT)
    assert result["degraded_sources"] == ["dexscreener"]
    assert "Ticker share unknown" in result["reasoning"]
    assert stubs["x"].call_args.args[1].startswith("($PEPE OR")


@pytest.mark.parametrize(
    "others,narrow",
    [(900.0, True), (300.0, False), (0.0, False)],
)
def test_shared_ticker_narrows_search(
    stubs: Dict[str, MagicMock], others: float, narrow: bool
) -> None:
    """Share below 0.25 (100 / (100 + others)) narrows X, news and LLM scope."""
    stubs["resolve"].return_value = {
        "symbol": "FUN",
        "chain": "robinhood",
        "volume": 100.0,
        "fdv": 38_000.0,
    }
    stubs["volumes"].return_value = {"0xother": others, ADDRESS: 100.0}
    _run(json.dumps({"symbol": "FUN", "address": ADDRESS}))
    query = stubs["x"].call_args.args[1]
    assert query.startswith('(($FUN "robinhood")') is narrow
    assert (stubs["news"].call_args.args[1] == '"FUN" robinhood token') is narrow
    notes = stubs["score"].call_args.args[2]["notes"]
    assert any("shared with other, larger tokens" in n for n in notes) is narrow


def test_share_exactly_at_threshold_is_not_narrowed(
    stubs: Dict[str, MagicMock],
) -> None:
    """Share of exactly 0.25 keeps the broad query."""
    stubs["resolve"].return_value = {
        "symbol": "FUN",
        "chain": "robinhood",
        "volume": 25.0,
        "fdv": 38_000.0,
    }
    stubs["volumes"].return_value = {"0xother": 75.0}
    _run(json.dumps({"symbol": "FUN", "address": ADDRESS}))
    assert stubs["x"].call_args.args[1].startswith("($FUN OR")


def test_shared_ticker_without_chain_uses_address(stubs: Dict[str, MagicMock]) -> None:
    """No chain known: X and news both search by address, with a note."""
    stubs["resolve"].return_value = {
        "symbol": "FUN",
        "chain": None,
        "volume": 0.0,
        "fdv": 0.0,
    }
    stubs["volumes"].return_value = {"0xother": 500.0}
    result = _run(json.dumps({"symbol": "FUN", "address": ADDRESS}))
    assert stubs["x"].call_args.args[1] == f'("{ADDRESS}") -is:retweet'
    assert stubs["news"].call_args.args[1] == f'"{ADDRESS}"'
    assert "searched by contract address only" in result["reasoning"]


def test_ambiguous_ticker_without_address_is_noted(stubs: Dict[str, MagicMock]) -> None:
    """No address + ambiguous ticker: note in output and guidance to the LLM."""
    stubs["volumes"].return_value = {"0xa": 50.0, "0xb": 50.0}
    result = _run("Is sentiment on $AI positive right now?")
    assert result["reasoning"].startswith("No contract address given")
    assert result["sentiment"] is not None
    notes = stubs["score"].call_args.args[2]["notes"]
    assert any("which asset is meant" in n for n in notes)


def test_dominant_ticker_without_address_is_not_noted(
    stubs: Dict[str, MagicMock],
) -> None:
    """A ticker with one token holding exactly 90% of volume is not ambiguous."""
    stubs["volumes"].return_value = {"0xa": 90.0, "0xb": 10.0}
    result = _run(json.dumps({"symbol": "PEPE"}))
    assert "No contract address given" not in result["reasoning"]


def test_one_source_failing_continues(stubs: Dict[str, MagicMock]) -> None:
    """X failing still scores from news, says so and flags the source."""
    stubs["x"].side_effect = requests.ConnectionError("down")
    stubs["news"].return_value = _headlines(6)
    stubs["score"].return_value = _labels(
        bullish=["n1", "n2", "n3"], neutral=["n4", "n5"]
    )
    result = _run(PEPE_PROMPT)
    assert result["error"] is None
    assert result["posts_analyzed"] == 0
    assert result["breakdown"] == {"bullish": 3, "neutral": 2, "bearish": 0}
    assert result["sentiment"] == 0.6
    assert "X unavailable" in result["reasoning"]
    assert result["degraded_sources"] == ["x"]


def test_news_failing_while_x_works(stubs: Dict[str, MagicMock]) -> None:
    """Serper failing still scores from X and flags news."""
    stubs["news"].side_effect = requests.HTTPError("401")
    stubs["score"].return_value = _labels(bullish=["p1", "p2", "p3", "p4", "p5"])
    result = _run(PEPE_PROMPT)
    assert result["sentiment"] == 1.0
    assert result["degraded_sources"] == ["news"]
    assert "news unavailable" in result["reasoning"]


def test_x_partial_and_counts_degradation_reported(stubs: Dict[str, MagicMock]) -> None:
    """Degraded sources from the X fetch reach the output."""
    stubs["x"].return_value = (_posts(8), None, ["x_partial", "x_counts"])
    result = _run(PEPE_PROMPT)
    assert result["degraded_sources"] == ["x_partial", "x_counts"]
    assert result["mentions"] is None


def test_missing_x_key_continues(stubs: Dict[str, MagicMock]) -> None:
    """No X key behaves like X being unavailable."""
    result = _run(
        PEPE_PROMPT,
        keys={"openai": "sk", "serperapi": "s"},
    )
    assert result["error"] is None
    assert result["degraded_sources"] == ["x"]
    stubs["x"].assert_not_called()


@pytest.mark.parametrize("keys", [{"serperapi": "s", "x_bearer": "x"}, None])
def test_missing_openai_key_is_internal_error(
    stubs: Dict[str, MagicMock], keys: Any
) -> None:
    """No OpenAI key (or no api_keys at all) returns an internal error."""
    response = tool.run(
        tool="token_social_sentiment",
        prompt=json.dumps({"symbol": "PEPE"}),
        api_keys=keys,
    )
    result = json.loads(response[0])
    assert result["error"]["type"] == "internal"


def test_both_sources_failing(stubs: Dict[str, MagicMock]) -> None:
    """Both sources failing returns source_unavailable with degraded list."""
    stubs["x"].side_effect = requests.ConnectionError("down")
    stubs["news"].side_effect = requests.Timeout("slow")
    result = _run(PEPE_PROMPT)
    assert result["error"]["type"] == "source_unavailable"
    assert result["degraded_sources"] == ["x", "news"]


def test_no_data_is_not_an_error(stubs: Dict[str, MagicMock]) -> None:
    """No posts and no news gives null sentiment, empty lists, no LLM call."""
    stubs["x"].return_value = ([], 0, [])
    stubs["news"].return_value = []
    result = _run(json.dumps({"symbol": "NEWTOKEN", "address": ADDRESS}))
    assert result["error"] is None
    assert result["sentiment"] is None
    assert result["top_posts"] == []
    assert result["headlines"] == []
    assert "No posts or organic news" in result["reasoning"]
    stubs["score"].assert_not_called()


@pytest.mark.parametrize(
    "labels",
    [
        _labels(),
        _labels(bullish=["p1", "p2"], neutral=["p3", "n1"]),
        _labels(bullish=["p1", "p99", "x"], bearish=["n9"]),
    ],
)
def test_too_few_on_topic_gives_null_sentiment(
    stubs: Dict[str, MagicMock], labels: tool.ItemLabels
) -> None:
    """Fewer than MIN_ON_TOPIC_ITEMS valid on-topic items gives no score."""
    stubs["score"].return_value = labels
    result = _run(PEPE_PROMPT)
    assert result["error"] is None
    assert result["sentiment"] is None
    assert result["breakdown"] is None
    assert "too few for a reliable score" in result["reasoning"]


def test_min_on_topic_boundary_scores(stubs: Dict[str, MagicMock]) -> None:
    """Exactly MIN_ON_TOPIC_ITEMS on-topic items gives a score."""
    stubs["score"].return_value = _labels(
        bullish=["p1", "p2", "p3"], neutral=["p4"], bearish=["p5"]
    )
    result = _run(PEPE_PROMPT)
    assert result["breakdown"] == {"bullish": 3, "neutral": 1, "bearish": 1}
    assert result["sentiment"] == 0.4


def test_large_sample_has_no_small_sample_note(stubs: Dict[str, MagicMock]) -> None:
    """Ten or more on-topic items are not flagged as a small sample."""
    stubs["x"].return_value = (_posts(12), 50, [])
    stubs["score"].return_value = _labels(bullish=[f"p{i}" for i in range(1, 11)])
    result = _run(PEPE_PROMPT)
    assert "Based on only" not in result["reasoning"]


def test_tally_drops_unknown_and_conflicting_ids() -> None:
    """Unknown ids are ignored and an id in two classes is dropped."""
    labels = _labels(
        bullish=["p1", "p2", "p2", "zz"],
        neutral=["p2", "n1"],
        bearish=["n3", "p3"],
        off_topic=["p3", "p4"],
    )
    assert tool.tally(labels, n_posts=4, n_headlines=1) == {
        "p1": "bullish",
        "n1": "neutral",
    }


def test_llm_error(stubs: Dict[str, MagicMock]) -> None:
    """An OpenAI failure returns llm_error."""
    stubs["score"].side_effect = tool.openai.OpenAIError("boom")
    result = _run(PEPE_PROMPT)
    assert result["error"]["type"] == "llm_error"


def test_unexpected_exception_is_internal(stubs: Dict[str, MagicMock]) -> None:
    """Any other exception returns internal instead of raising."""
    stubs["score"].side_effect = KeyError("x")
    result = _run(PEPE_PROMPT)
    assert result["error"]["type"] == "internal"


def test_model_not_allowed() -> None:
    """Requester-chosen models outside the allow-list are rejected."""
    result = _run(json.dumps({"symbol": "PEPE"}), model="o3-pro")
    assert result["error"]["type"] == "invalid_input"


def test_unknown_tool_raises() -> None:
    """An unsupported tool name raises like the other tools."""
    with pytest.raises(ValueError):
        tool.run(tool="other", prompt="x", api_keys=KEYS)


def test_openai_client_has_timeout(stubs: Dict[str, MagicMock]) -> None:
    """The OpenAI client is built with a bounded timeout and retries."""
    with patch.object(tool, "OpenAI") as client_cls:
        _run(PEPE_PROMPT)
    assert client_cls.call_args.kwargs == {
        "api_key": "sk",
        "timeout": tool.LLM_TIMEOUT,
        "max_retries": tool.LLM_MAX_RETRIES,
    }


def _parse_client(parsed: Any, prompt_tokens: int = 12) -> MagicMock:
    """Build an OpenAI client whose parse() returns `parsed`."""
    client = MagicMock()
    response = MagicMock()
    response.choices[0].message.parsed = parsed
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = 3
    client.beta.chat.completions.parse.return_value = response
    return client


def test_score_sentiment_builds_prompt_and_counts_tokens() -> None:
    """score_sentiment formats the real prompt and reports usage."""
    labels = _labels(bullish=["p1"])
    client = _parse_client(labels)
    callback = MagicMock()
    target: Dict[str, Any] = {
        "token": "PONS",
        "address": None,
        "chain": "robinhood",
        "window_seconds": 7200,
        "user_text": None,
        "notes": ["note one"],
    }
    accented = "caf" + chr(0xE9)
    posts = [{"id": "1", "text": accented + " " + chr(0xD83D), "engagement": 0}]
    result = tool.score_sentiment(
        client, tool.DEFAULT_MODEL, target, posts, _headlines(1), callback
    )
    assert result is labels
    kwargs = client.beta.chat.completions.parse.call_args.kwargs
    assert kwargs["max_tokens"] == tool.SCORE_MAX_TOKENS
    assert kwargs["response_format"] is tool.ItemLabels
    user = kwargs["messages"][1]["content"]
    assert "Token: PONS" in user and "Chain: robinhood" in user
    assert "Contract address: unknown" in user
    assert "note one" in user and "last 2 hours" in user
    assert "none (structured request)" in user
    assert '"id": "p1"' in user and '"id": "n1"' in user
    # non-ASCII is sent as-is; the lone surrogate is replaced, not escaped
    assert accented in user and "\\u" not in user
    user.encode("utf-8")
    assert callback.call_args.kwargs["input_tokens"] == 12


def test_score_sentiment_parsed_none_is_llm_error() -> None:
    """A refusal (parsed None) surfaces as llm_error."""
    client = _parse_client(None)
    target: Dict[str, Any] = {
        "token": "PONS",
        "address": ADDRESS,
        "chain": None,
        "window_seconds": 86400,
        "user_text": "q",
        "notes": [],
    }
    with pytest.raises(tool.ToolError) as err:
        tool.score_sentiment(client, tool.DEFAULT_MODEL, target, [], [], None)
    assert err.value.error_type == "llm_error"


def test_extract_token_formats_prompt_and_handles_none() -> None:
    """extract_token sends the user text and maps a refusal to llm_error."""
    extracted = tool.ExtractedToken(symbols=["PEPE"])
    client = _parse_client(extracted)
    assert (
        tool.extract_token(client, tool.DEFAULT_MODEL, "how is pepe", None) is extracted
    )
    kwargs = client.beta.chat.completions.parse.call_args.kwargs
    assert "how is pepe" in kwargs["messages"][0]["content"]
    assert kwargs["max_tokens"] == tool.EXTRACT_MAX_TOKENS
    with pytest.raises(tool.ToolError):
        tool.extract_token(_parse_client(None), tool.DEFAULT_MODEL, "x", None)


def _x_response(data: Any = None, meta: Any = None) -> MagicMock:
    """Build an X API response."""
    response = MagicMock()
    response.json.return_value = {"data": data, "meta": meta}
    return response


def test_fetch_x_posts_slices_window_and_filters(monkeypatch: Any) -> None:
    """Window split in X_SLICES; near-duplicates and ticker lists dropped."""
    pages = [
        [
            {
                "id": "1",
                "text": "@a @b buy $PEPE now https://t.co/x1",
                "public_metrics": {"like_count": 2},
            },
            {"id": "2", "text": "buy $pepe  NOW https://t.co/other"},
        ],
        [{"id": "3", "text": "$PEPE $DOGE $SHIB $WIF top gainers"}],
        [{"id": "4", "text": "$PEPE $DOGE $SHIB comparison, PEPE looks weak"}],
        [],
    ]
    calls: List[Dict[str, Any]] = []

    def fake_get(url: str, params: Dict[str, Any], **_: Any) -> Any:
        calls.append({"url": url, **params})
        if url == tool.X_COUNTS_URL:
            return _x_response(meta={"total_tweet_count": 77})
        return _x_response(data=pages[len(calls) - 1])

    monkeypatch.setattr(tool.requests, "get", fake_get)
    end = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    posts, mentions, degraded = tool.fetch_x_posts(
        "x", "q", end - timedelta(hours=24), end
    )
    assert [p["id"] for p in posts] == ["1", "4"]
    assert posts[0]["engagement"] == 2
    assert mentions == 77
    assert not degraded
    search = [c for c in calls if c["url"] == tool.X_SEARCH_URL]
    assert [c["start_time"] for c in search] == [
        "2026-09-15T12:00:00Z",
        "2026-09-15T18:00:00Z",
        "2026-09-16T00:00:00Z",
        "2026-09-16T06:00:00Z",
    ]
    assert search[-1]["end_time"] == "2026-09-16T12:00:00Z"
    assert all(c["max_results"] == tool.POSTS_PER_SLICE for c in search)
    assert all(c["sort_order"] == "relevancy" for c in search)


def test_fetch_x_posts_keeps_posts_when_a_slice_fails(monkeypatch: Any) -> None:
    """One failed slice keeps the other slices' posts and flags x_partial."""
    calls: List[str] = []

    def fake_get(url: str, **_: Any) -> Any:
        calls.append(url)
        if len(calls) == 3:
            raise requests.HTTPError("429")
        if url == tool.X_COUNTS_URL:
            return _x_response(meta={"total_post_count": 9})
        return _x_response(data=[{"id": str(len(calls)), "text": f"t {len(calls)}"}])

    monkeypatch.setattr(tool.requests, "get", fake_get)
    end = datetime(2026, 9, 16, tzinfo=timezone.utc)
    posts, mentions, degraded = tool.fetch_x_posts(
        "x", "q", end - timedelta(hours=4), end
    )
    assert [p["id"] for p in posts] == ["1", "2", "4"]
    assert mentions == 9
    assert degraded == ["x_partial"]


@pytest.mark.parametrize("meta", [None, {}])
def test_fetch_x_posts_counts_missing_is_degraded(monkeypatch: Any, meta: Any) -> None:
    """A counts failure or empty meta leaves mentions None and flags it."""

    def fake_get(url: str, **_: Any) -> Any:
        if url == tool.X_COUNTS_URL:
            if meta is None:
                raise requests.HTTPError("403 tier")
            return _x_response(meta=meta)
        return _x_response(data=[{"id": "1", "text": "hi $PEPE"}])

    monkeypatch.setattr(tool.requests, "get", fake_get)
    end = datetime(2026, 9, 16, tzinfo=timezone.utc)
    posts, mentions, degraded = tool.fetch_x_posts(
        "x", "q", end - timedelta(hours=1), end
    )
    assert len(posts) == 1
    assert mentions is None
    assert degraded == ["x_counts"]


def test_fetch_x_posts_all_slices_failing_raises(monkeypatch: Any) -> None:
    """Every slice failing propagates so the caller marks X unavailable."""
    monkeypatch.setattr(
        tool.requests, "get", MagicMock(side_effect=requests.ConnectionError("down"))
    )
    end = datetime(2026, 9, 16, tzinfo=timezone.utc)
    with pytest.raises(requests.RequestException):
        tool.fetch_x_posts("x", "q", end - timedelta(hours=1), end)


@pytest.mark.parametrize(
    "symbol,address,chain,narrow,expected",
    [
        ("PEPE", None, None, False, "($PEPE) -is:retweet"),
        ("PEPE", ADDRESS, "ethereum", False, f'($PEPE OR "{ADDRESS}") -is:retweet'),
        (
            "FUN",
            ADDRESS,
            "robinhood",
            True,
            f'(($FUN "robinhood") OR "{ADDRESS}") -is:retweet',
        ),
        ("FUN", ADDRESS, None, True, f'("{ADDRESS}") -is:retweet'),
    ],
)
def test_build_x_query(
    symbol: str, address: Any, chain: Any, narrow: bool, expected: str
) -> None:
    """A shared ticker is never searched as a bare cashtag."""
    assert tool.build_x_query(symbol, address, chain, narrow) == expected


def _dex_response(pairs: Any) -> MagicMock:
    """Build a DexScreener response."""
    response = MagicMock()
    response.json.return_value = {"pairs": pairs}
    return response


def _pair(
    symbol: str, address: str, volume: Any, chain: str = "robinhood"
) -> Dict[str, Any]:
    """Build a DexScreener pair."""
    return {
        "chainId": chain,
        "baseToken": {"symbol": symbol, "address": address},
        "volume": {"h24": volume},
    }


def test_resolve_token(monkeypatch: Any) -> None:
    """Symbol, chain of the busiest pair and summed volume of own pairs."""
    pairs = [
        {
            **_pair("$fun", ADDRESS.upper().replace("0X", "0x"), 10, chain="base"),
            "fdv": 5e6,
        },
        {**_pair("FUN", ADDRESS, 90, chain="robinhood"), "fdv": "4000000"},
        _pair("OTHER", "0xother", 10**6),
    ]
    monkeypatch.setattr(
        tool.requests, "get", MagicMock(return_value=_dex_response(pairs))
    )
    assert tool.resolve_token(ADDRESS) == {
        "symbol": "FUN",
        "chain": "robinhood",
        "volume": 100.0,
        "fdv": 5e6,
    }


@pytest.mark.parametrize(
    "pairs,expected",
    [
        ([], {}),
        (
            [_pair("WIF HAT", ADDRESS, 5)],
            {"symbol": None, "chain": "robinhood", "volume": 5.0, "fdv": 0.0},
        ),
        ("oops", None),
        (
            [None, _pair("A", ADDRESS, "n/a")],
            {"symbol": "A", "chain": "robinhood", "volume": 0.0, "fdv": 0.0},
        ),
    ],
)
def test_resolve_token_edge_cases(monkeypatch: Any, pairs: Any, expected: Any) -> None:
    """Not listed, bad symbol, malformed body and malformed volume."""
    monkeypatch.setattr(
        tool.requests, "get", MagicMock(return_value=_dex_response(pairs))
    )
    assert tool.resolve_token(ADDRESS) == expected


def test_dexscreener_lookup_failure(monkeypatch: Any) -> None:
    """A failed request returns None (distinct from not listed)."""
    monkeypatch.setattr(
        tool.requests, "get", MagicMock(side_effect=requests.Timeout("slow"))
    )
    assert tool.resolve_token(ADDRESS) is None
    assert tool.symbol_volumes("FUN") is None


def test_symbol_volumes_and_share(monkeypatch: Any) -> None:
    """Share uses the target's own volume against other same-ticker tokens."""
    pairs = [
        _pair("FUN", "0xother", 900),
        _pair("$fun", ADDRESS.upper().replace("0X", "0x"), 100),
        _pair("FUNNY", "0xx", 10**9),
    ]
    monkeypatch.setattr(
        tool.requests, "get", MagicMock(return_value=_dex_response(pairs))
    )
    volumes = tool.symbol_volumes("FUN")
    assert volumes == {"0xother": 900.0, ADDRESS: 100.0}
    assert tool.ticker_share(volumes, ADDRESS, 100.0) == 0.1
    # target missing from the capped search results still gets its own volume
    assert tool.ticker_share({"0xother": 300.0}, ADDRESS, 100.0) == 0.25
    assert tool.ticker_share({}, ADDRESS, 0.0) is None


@pytest.mark.parametrize(
    "volumes,expected",
    [
        ({"0xa": 100.0}, False),
        ({"0xa": 50.0, "0xb": 50.0}, True),
        ({"0xa": 90.0, "0xb": 10.0}, False),
        ({"0xa": 89.0, "0xb": 11.0}, True),
        ({}, True),
        (None, True),
    ],
)
def test_is_ambiguous_ticker(volumes: Any, expected: bool) -> None:
    """Ambiguous unless one DEX token has at least 90% of the volume."""
    assert tool.is_ambiguous_ticker(volumes) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("PEPE looks strong today", False),
        (f"accumulating PEPE, CA {ADDRESS}", False),
        (f"new gem CA: {OTHER_ADDRESS}", True),
        (f"better than PEPE {SOL_ADDRESS}", True),
        ("join our group https://t.me/pepepump", True),
        ("WhatsApp group for PEPE holders", True),
        ("PEPE AIRDROP live", True),
        ("thoughts on pepe? https://t.co/abc", False),
        ("bought more $PEPE, tx 0x" + "ab" * 32, False),
        ("spotted $PEPE in the listing feed, cast your vote", True),
        ("@GoPlusSecurity I nominate $PEPE #DeepScanAudit", True),
        ("$PEPE Telegram is live", True),
        ("Don't miss $PEPE", True),
        ("devoted $PEPE holder since launch", False),
    ],
)
def test_is_promo(text: str, expected: bool) -> None:
    """Other-token addresses and group/giveaway markers are promotion."""
    assert tool.is_promo(text, ADDRESS) is expected


def test_clean_post_text() -> None:
    """URLs, entities and the target address are stripped or replaced."""
    upper = ADDRESS.upper().replace("0X", "0x")
    text = f"PEPE &gt; DOGE https://t.co/abc  CA {upper}"
    assert tool.clean_post_text(text, ADDRESS) == "PEPE > DOGE CA [CA]"
    assert tool.clean_post_text(text, None) == f"PEPE > DOGE CA {upper}"


def test_promo_posts_dropped_before_scoring(stubs: Dict[str, MagicMock]) -> None:
    """Promo posts never reach the LLM."""
    posts = _posts(3)
    posts[1]["text"] = "join https://t.me/x"
    stubs["x"].return_value = (posts, 3, [])
    _run(PEPE_PROMPT)
    sent = stubs["score"].call_args.args[3]
    assert [p["id"] for p in sent] == ["100", "102"]


@pytest.mark.parametrize(
    "source,url,expected",
    [
        ("openPR.com", "https://www.openpr.com/news/1", True),
        ("TechBullion", "https://techbullion.com/x", True),
        ("StreetInsider", "https://streetinsider.com/MarketMediaWire/x", True),
        ("Coinpedia", "https://coinpedia.org/press-release/x", True),
        ("Reuters", "https://reuters.com/markets/pepe", False),
    ],
)
def test_is_press_release(source: str, url: str, expected: bool) -> None:
    """Paid press releases are recognized by source or URL."""
    assert tool.is_press_release(source, url) is expected


def test_fetch_headlines_drops_press_releases_and_duplicates(monkeypatch: Any) -> None:
    """Serper results from PR wires and repeated titles are filtered out."""
    response = MagicMock()
    response.json.return_value = {
        "news": [
            {
                "title": "promo",
                "link": "https://www.openpr.com/a",
                "source": "openPR.com",
            },
            {"title": "Real News", "link": "https://reuters.com/b", "source": "R"},
            {"title": "real  news", "link": "https://other.com/b", "source": "O"},
        ]
    }
    post = MagicMock(return_value=response)
    monkeypatch.setattr(tool.requests, "post", post)
    headlines = tool.fetch_headlines("k", '"PEPE" token', 86400)
    assert [h["title"] for h in headlines] == ["Real News"]
    assert post.call_args.kwargs["json"] == {
        "q": '"PEPE" token',
        "tbs": "qdr:d",
        "num": 10,
    }


@pytest.mark.parametrize(
    "window,tbs", [(3600, "qdr:h"), (86400, "qdr:d"), (604000, "qdr:w")]
)
def test_fetch_headlines_time_filter(monkeypatch: Any, window: int, tbs: str) -> None:
    """Serper's time filter covers the requested window."""
    response = MagicMock()
    response.json.return_value = {"news": []}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(tool.requests, "post", post)
    tool.fetch_headlines("k", "q", window)
    assert post.call_args.kwargs["json"]["tbs"] == tbs


@pytest.mark.parametrize(
    "text,expected",
    [
        ("robinhood:" + ADDRESS, False),
        ("@user Gm robinhood:" + ADDRESS, True),
        ("Bullish $PONS", True),
        ("@user $PONS #HTX", False),
        (ADDRESS + " " + SOL_ADDRESS, False),
        ("excited for $PONS", True),
        ("Great launch robinhood:" + ADDRESS + " another pump and dump", True),
        ("".join(chr(c) for c in (0x6211, 0x5F88, 0x770B, 0x597D)) + " $PONS", True),
        ("https://t.co/abc $PONS", False),
    ],
)
def test_has_words(text: str, expected: bool) -> None:
    """Posts made only of handles, tags, links and addresses have no words."""
    assert tool.has_words(text) is expected


def test_wordless_posts_dropped_before_scoring(stubs: Dict[str, MagicMock]) -> None:
    """Posts without words never reach the LLM."""
    posts = _posts(3)
    posts[1]["text"] = f"@user robinhood:{ADDRESS}"
    stubs["x"].return_value = (posts, 3, [])
    _run(PEPE_PROMPT)
    assert [p["id"] for p in stubs["score"].call_args.args[3]] == ["100", "102"]


@pytest.mark.parametrize(
    "symbol,address,chain,narrow,expected",
    [
        ("ARGUS", ADDRESS, "arc", False, '"ARGUS" token'),
        ("PEPE", None, None, False, '"PEPE" token'),
        ("FUN", ADDRESS, "robinhood", True, '"FUN" robinhood token'),
        ("FUN", ADDRESS, None, True, f'"{ADDRESS}"'),
        (None, ADDRESS, None, False, f'"{ADDRESS}"'),
    ],
)
def test_news_query(
    symbol: Any, address: Any, chain: Any, narrow: bool, expected: str
) -> None:
    """The ticker is quoted and the scope matches the X query."""
    assert tool.news_query(symbol, address, chain, narrow) == expected


def test_established_shared_ticker_is_not_narrowed(stubs: Dict[str, MagicMock]) -> None:
    """A low DEX share but a large valuation keeps the broad query, with notes."""
    stubs["resolve"].return_value = {
        "symbol": "MAGIC",
        "chain": "arbitrum",
        "volume": 2_786.0,
        "fdv": 12_900_000.0,
    }
    stubs["volumes"].return_value = {"0xother": 377_990.0}
    result = _run(json.dumps({"symbol": "MAGIC", "address": ADDRESS}))
    assert stubs["x"].call_args.args[1].startswith("($MAGIC OR")
    assert "shared with tokens that have more DEX volume" in result["reasoning"]
    notes = stubs["score"].call_args.args[2]["notes"]
    assert any("shared with other, larger tokens" in n for n in notes)


def test_x_partial_and_counts_notes(stubs: Dict[str, MagicMock]) -> None:
    """Partial X failures and missing counts are also explained in reasoning."""
    stubs["x"].return_value = (_posts(8), None, ["x_partial", "x_counts"])
    result = _run(PEPE_PROMPT)
    assert "Some X time slices failed" in result["reasoning"]
    assert "X post count unavailable." in result["reasoning"]


def test_missing_serper_key_continues(stubs: Dict[str, MagicMock]) -> None:
    """No Serper key behaves like news being unavailable."""
    result = _run(PEPE_PROMPT, keys={"openai": "sk", "x_bearer": "x"})
    assert result["error"] is None
    assert result["degraded_sources"] == ["news"]
    stubs["news"].assert_not_called()


def test_symbol_only_lookup_failure_is_degraded_and_ambiguous(
    stubs: Dict[str, MagicMock],
) -> None:
    """A failed DexScreener search on a symbol-only request is flagged and noted."""
    stubs["volumes"].return_value = None
    result = _run(json.dumps({"symbol": "PEPE"}))
    assert result["degraded_sources"] == ["dexscreener"]
    assert result["reasoning"].startswith("No contract address given")


def test_resolve_token_rejects_odd_chain_id(monkeypatch: Any) -> None:
    """A DexScreener chainId that fails the chain check is not used."""
    pairs = [_pair("FUN", ADDRESS, 5, chain="robin) OR (x")]
    monkeypatch.setattr(
        tool.requests, "get", MagicMock(return_value=_dex_response(pairs))
    )
    info = tool.resolve_token(ADDRESS)
    assert info is not None and info["chain"] is None
