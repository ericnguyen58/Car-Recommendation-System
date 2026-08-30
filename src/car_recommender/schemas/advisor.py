from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    # Opaque passthrough: exactly what /advisor/chat returned as `history` on
    # the previous turn (or [] for a new conversation). The API is stateless —
    # the caller owns conversation state, same shape app.py kept in
    # st.session_state, just carried over HTTP instead of in-process.
    history: list[dict] = []
    active_filters: list[str] = []


class ChatResponse(BaseModel):
    reply: str
    history: list[dict]
