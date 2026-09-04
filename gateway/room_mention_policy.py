"""Exact, authorization-adjacent current-room mention policy phrases."""

CURRENT_ROOM_MENTION_REQUIRED_PHRASES = frozenset(
    {
        "이 방에서는 멘션할 때만 답해",
        "이 방에서는 멘션해야 답해",
        "이 방에서는 멘션해야만 답해",
        "이 방에서는 나를 멘션할 때만 답해",
        "이 방에서는 멘션할 때만 대답해",
        "이 방에서는 멘션할 때만 응답해",
        "이 방에선 멘션할 때만 답해",
        "이 방에서 멘션할 때만 답해",
        "이 방에서 멘션해야 답해",
        "이 방에서 멘션해야만 답해",
        "이 방에서는 내가 태그할 때만 답해",
        "이 방에서는 내가 태그할 때만 대답해",
        "이 방에서는 내가 태그하기 전에는 말하지 마",
        "이 방에서는 내가 태그하기 전까지 말하지 마",
        "내가 태그하기 전에는 말하지 마",
        "내가 태그하기 전까지 말하지 마",
        "내가 태그할 때만 답해",
        "내가 태그할 때만 대답해",
        "only respond when mentioned in this room",
        "in this room, only respond when mentioned",
        "only answer when mentioned in this room",
        "in this room, only answer when mentioned",
        "only respond to mentions in this room",
        "require a mention in this room",
        "require mentions in this room",
    }
)

CURRENT_ROOM_MENTION_OPTIONAL_PHRASES = frozenset(
    {
        "이 방에서는 멘션 없이 답해",
        "이 방에서는 멘션 없이도 답해",
        "이 방에서는 멘션 없이도 대답해",
        "이 방에서는 멘션 없이도 응답해",
        "이 방에선 멘션 없이 답해",
        "이 방에서 멘션 없이 답해",
        "이 방에서 멘션 없이도 답해",
        "너 안불러도 작동하게해",
        "너 안 불러도 작동하게 해",
        "너 안불러도 일하게해",
        "너 안불러도 일하게 하자",
        "너 안불러도 일하게 세팅해",
        "너 안불러도 반응하게 세팅해",
        "이 방에서 너 안불러도 일하게 세팅해",
        "이 방에서 너 안불러도 반응하게 세팅해",
        "에이전트 태그 안해도 응답해",
        "에이전트 태그 안 해도 응답해",
        "에이전트 태그 안해도 답해",
        "에이전트 태그 안 해도 답해",
        "이 방에서는 에이전트 태그 안 해도 응답해",
        "이 채널에서는 에이전트 태그 안 해도 응답해",
        "이 채널에서 에이전트 태그 안 해도 응답해",
        "태그 안해도 응답해",
        "태그 안 해도 응답해",
        "멘션 안해도 응답해",
        "멘션 안 해도 응답해",
        "이 채널에서는 멘션 없이도 응답해",
        "respond without mentions in this room",
        "in this room, respond without mentions",
        "answer without mentions in this room",
        "in this room, answer without mentions",
        "don't require mentions in this room",
        "do not require mentions in this room",
        "mentions are not required in this room",
    }
)


def current_room_mention_policy(text: object) -> str | None:
    """Return ``required``/``optional`` for an exact registered phrase."""
    normalized = str(text or "").strip().casefold()
    if normalized in CURRENT_ROOM_MENTION_REQUIRED_PHRASES:
        return "required"
    if normalized in CURRENT_ROOM_MENTION_OPTIONAL_PHRASES:
        return "optional"
    return None
