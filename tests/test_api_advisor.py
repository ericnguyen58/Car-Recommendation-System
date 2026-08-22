"""Tests for the advisor's tool-use loop, mocking the Anthropic client
entirely — no network call and no ANTHROPIC_API_KEY needed to run in CI."""

from unittest.mock import MagicMock, patch

from car_recommender.core.config import Settings
from car_recommender.llm.advisor import run_advisor_turn


class FakeBlock:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return dict(self.__dict__)


class FakeResponse:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


def _settings(**overrides) -> Settings:
    return Settings(ANTHROPIC_API_KEY="test-key", ADVISOR_MODEL="claude-haiku-4-5", **overrides)


def test_advisor_executes_tool_and_returns_reply(model_store):
    tool_use_block = FakeBlock(
        type="tool_use", name="get_car_recommendations", id="tool_1", input={"bodytype": ["Sedan"]}
    )
    turn1 = FakeResponse(content=[tool_use_block], stop_reason="tool_use")

    text_block = FakeBlock(type="text", text="Here are some sedans for you.")
    turn2 = FakeResponse(content=[text_block], stop_reason="end_turn")

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [turn1, turn2]

    with patch("car_recommender.llm.advisor.anthropic.Anthropic", return_value=fake_client):
        reply, history = run_advisor_turn(
            settings=_settings(),
            store=model_store,
            history=[],
            user_message="I want a sedan",
        )

    assert reply == "Here are some sedans for you."
    assert fake_client.messages.create.call_count == 2

    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant", "user", "assistant"]
    tool_result_message = history[2]
    assert tool_result_message["content"][0]["type"] == "tool_result"
    assert tool_result_message["content"][0]["tool_use_id"] == "tool_1"


def test_advisor_stops_immediately_on_end_turn(model_store):
    text_block = FakeBlock(type="text", text="Sure, tell me your budget.")
    only_turn = FakeResponse(content=[text_block], stop_reason="end_turn")

    fake_client = MagicMock()
    fake_client.messages.create.return_value = only_turn

    with patch("car_recommender.llm.advisor.anthropic.Anthropic", return_value=fake_client):
        reply, _history = run_advisor_turn(
            settings=_settings(), store=model_store, history=[], user_message="hi"
        )

    assert reply == "Sure, tell me your budget."
    assert fake_client.messages.create.call_count == 1


def test_advisor_caps_tool_loop_iterations(model_store):
    tool_use_block = FakeBlock(type="tool_use", name="get_car_recommendations", id="tool_x", input={})
    always_tool_use = FakeResponse(content=[tool_use_block], stop_reason="tool_use")

    fake_client = MagicMock()
    fake_client.messages.create.return_value = always_tool_use

    settings = _settings(MAX_TOOL_LOOPS=3)
    with patch("car_recommender.llm.advisor.anthropic.Anthropic", return_value=fake_client):
        reply, _history = run_advisor_turn(
            settings=settings, store=model_store, history=[], user_message="anything"
        )

    assert fake_client.messages.create.call_count == 3
    assert reply == ""  # no text block was ever returned before the loop cap was hit
