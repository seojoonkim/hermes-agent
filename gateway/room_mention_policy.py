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
        "너 안불러도 일할 수 있게 세팅해",
        "너 안 불러도 일할 수 있게 세팅해",
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


import re

# Tolerant Korean phrasing: owners rarely repeat a registered sentence
# verbatim ("너 안불러도 일할 수 있게 세팅해" vs "...일하게 세팅해").  The
# fallback stays conservative: a short message (no task content) that
# contains a "without calling/tagging" cue AND a "respond/work" cue AND is
# phrased as an instruction.  Anything longer is left to the model.
_MAX_FUZZY_LEN = 40
_NO_CALL = re.compile(
    r"(안\s*불러|부르지\s*않|안\s*부르|호명\s*(없이|안)|"
    r"(멘션|태그|언급)\s*(없이|안\s*해|안\s*하|하지\s*않))"
)
_ACT = re.compile(r"(일|답|대답|응답|반응|작동|동작|말\s*듣|알아듣)")
_DIRECTIVE = re.compile(
    r"(세팅|설정|셋업|해\s*줘|해줘|하게|있게|해|하자|되게|돼)\s*[.!~?]*$"
)
_REQUIRE = re.compile(
    r"((멘션|태그|호명|부를)\s*(할|했을|해야|하면)\s*(때만|만|경우에만)|"
    r"(태그|멘션)\s*하기\s*전(에는|까지)\s*(말하지|답하지))"
)


# Questions/meta-requests about the policy are conversation, not commands.
_NOT_COMMAND = re.compile(r"(\?|왜|어떻게|확인|정리|원인|로그|리서치|알려|뭐야|된\s*거|된\s*건)")


def _fuzzy_policy(normalized: str) -> str | None:
    if not normalized or len(normalized) > _MAX_FUZZY_LEN or "\n" in normalized:
        return None
    if _NOT_COMMAND.search(normalized):
        return None
    if _REQUIRE.search(normalized) and _DIRECTIVE.search(normalized) or (
        _REQUIRE.search(normalized) and "말하지" in normalized
    ):
        return "required"
    if (
        _NO_CALL.search(normalized)
        and _ACT.search(normalized)
        and _DIRECTIVE.search(normalized)
    ):
        return "optional"
    return None


def current_room_mention_policy(text: object) -> str | None:
    """Return ``required``/``optional`` for a room mention-policy instruction.

    Exact registered phrases win; a conservative fuzzy matcher covers short
    Korean paraphrases so the gateway hot-applies them without a restart.
    """
    normalized = str(text or "").strip().casefold()
    if normalized in CURRENT_ROOM_MENTION_REQUIRED_PHRASES:
        return "required"
    if normalized in CURRENT_ROOM_MENTION_OPTIONAL_PHRASES:
        return "optional"
    return _fuzzy_policy(normalized)
