"""POST /advisor/chat — the conversational advisor, backed by llm/advisor.py."""

import anthropic
from fastapi import APIRouter, Depends, HTTPException, status

from car_recommender.api.dependencies import enforce_rate_limit, get_model_store, require_api_key
from car_recommender.core.config import Settings, get_settings
from car_recommender.core.logging import get_logger
from car_recommender.core.sanitize import sanitize_text
from car_recommender.llm.advisor import run_advisor_turn
from car_recommender.ml.model_store import ModelStore
from car_recommender.schemas.advisor import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/advisor",
    tags=["advisor"],
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)

_log = get_logger()


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    store: ModelStore = Depends(get_model_store),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    if not settings.ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Advisor is not configured (missing ANTHROPIC_API_KEY).",
        )

    message = sanitize_text(req.message, settings.MAX_INPUT_LEN)
    if not message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty.")
    if len(req.history) >= settings.MAX_TURNS * 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conversation limit reached. Start a new conversation.",
        )

    try:
        reply, history = run_advisor_turn(
            settings=settings,
            store=store,
            history=req.history,
            user_message=message,
            active_filters=req.active_filters,
        )
    except anthropic.RateLimitError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Advisor rate limit reached. Please wait a moment and try again.",
        )
    except anthropic.APIStatusError as e:
        _log.error("Advisor APIStatusError status=%d", e.status_code)
        detail = (
            "The advisor service is temporarily unavailable."
            if e.status_code >= 500
            else "The advisor could not process your request."
        )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Could not reach the advisor service.")

    _log.info("Advisor reply sent (len=%d)", len(reply))
    return ChatResponse(reply=reply, history=history)
