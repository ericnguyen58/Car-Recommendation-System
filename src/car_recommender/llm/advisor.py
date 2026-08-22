"""Shared advisor service: builds the system prompt and runs the tool-use
loop against Claude, executing tool calls server-side.

This is the ONE place that talks to the Anthropic API — used only by the
POST /advisor/chat route. The Streamlit UI (and any other client) talks to
that HTTP endpoint, never to Anthropic directly, so there is exactly one
implementation of the tool-calling loop instead of the old app.py /
chat_recommender.py duplication.

The API is stateless: the caller owns `history` and sends it back on every
turn (same shape app.py used with Streamlit session_state), so every
message — including assistant turns — is stored as a plain JSON-safe dict
via `.model_dump()`, never as a raw SDK object.
"""

import json
import logging

import anthropic

from car_recommender.core.config import Settings
from car_recommender.llm.tools import ADVISOR_TOOLS, build_known_categorical, validate_tool_input
from car_recommender.ml.filters import get_recommendations
from car_recommender.ml.model_store import ModelStore

_log = logging.getLogger("car_recommender")


def build_system_prompt(store: ModelStore, active_filters: list[str] | None = None) -> str:
    df = store.df
    valid_bodytypes = ", ".join(sorted(df["bodytype"].dropna().unique()))
    valid_drives = ", ".join(sorted(df["Drivetrain"].dropna().unique()))
    valid_fuels = ", ".join(sorted(df["Fuel Type"].dropna().unique())[:12])

    system = f"""You are a knowledgeable, friendly car buying advisor helping first-time buyers \
find a reliable used car. The catalog covers model years 2001-2024 across all major makes.

You have one tool: get_car_recommendations. Use it whenever the buyer asks for \
suggestions, wants to compare options, or mentions any preference that maps to a filter.

When presenting results:
- Highlight the top 2-3 picks and explain specifically why each fits the buyer's stated needs
- Always call out the reliability rating (out of 5.0) when available — this is the most \
important factor for used car buyers
- Mention fuel economy for budget-conscious buyers
- Remind buyers that listed prices are new-car MSRP, not used market value
- If a search returns zero results, diagnose which filter is likely too strict, \
suggest an adjustment, and call the tool again with relaxed criteria
- Keep explanations plain, concise, and free of jargon

Valid catalog values for reference:
  Body types  : {valid_bodytypes}
  Drivetrains : {valid_drives}
  Fuel types  : {valid_fuels} (and more)
"""

    if active_filters:
        system += (
            "\n\nThe user has already pre-set the following sidebar filters. "
            "Your tool is already restricted to cars matching these criteria — "
            "do NOT re-apply them as tool parameters (that would over-constrain the results). "
            "Instead, use the tool with only the ADDITIONAL filters the user mentions in chat. "
            "Always acknowledge the active sidebar filters when presenting recommendations "
            "so the user understands the scope.\n\nActive sidebar filters:\n"
            + "\n".join(f"  - {line}" for line in active_filters)
        )
    return system


def _execute_tool(name: str, tool_input: dict, df, known_categorical: dict) -> str:
    if name != "get_car_recommendations":
        _log.warning("Unknown tool requested: %s", str(name)[:64])
        return json.dumps({"error": "Unknown tool."})
    validated = validate_tool_input(dict(tool_input), known_categorical)
    top_n = validated.pop("top_n", 5)
    result = get_recommendations(validated, df, top_n=top_n)
    _log.info("Tool call: filters=%s matched=%s", list(validated.keys()), result.get("total_matched", 0))
    return json.dumps(result, default=str)


def run_advisor_turn(
    *,
    settings: Settings,
    store: ModelStore,
    history: list[dict],
    user_message: str,
    active_filters: list[str] | None = None,
    base_df=None,
) -> tuple[str, list[dict]]:
    """Run one user turn through Claude, executing tool calls server-side.

    `history` is the caller-owned message list from the previous turn (empty
    on the first message of a conversation). Returns (reply_text, new_history)
    — `new_history` is what the caller should send back on the next turn.
    """
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY, max_retries=3)
    system = build_system_prompt(store, active_filters)
    known_categorical = build_known_categorical(store.df)
    df = base_df if base_df is not None else store.df

    messages = list(history) + [{"role": "user", "content": user_message}]

    response = None
    for _ in range(settings.MAX_TOOL_LOOPS):
        response = client.messages.create(
            model=settings.ADVISOR_MODEL,
            max_tokens=2048,
            system=system,
            tools=ADVISOR_TOOLS,
            messages=messages,
        )
        messages.append({
            "role": "assistant",
            "content": [block.model_dump() for block in response.content],
        })

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = _execute_tool(block.name, dict(block.input), df, known_categorical)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})

    reply = next((b.text for b in response.content if b.type == "text"), "") if response else ""
    return reply, messages
