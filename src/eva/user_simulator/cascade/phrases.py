"""Caller phrases by language code, data only. Languages missing here fall back to ``DEFAULT_LANGUAGE``."""

from __future__ import annotations

DEFAULT_LANGUAGE = "en"

OPENERS: dict[str, list[str]] = {
    "en": ["Sorry, wait—", "Hold on—", "Wait one sec—", "Oh, actually—", "Wait, sorry—"],
    "es": ["Perdona, espera—", "Un momento—", "Espera un segundo—", "Ah, en realidad—", "Espera, perdón—"],
    "fr": ["Pardon, attendez—", "Une seconde—", "Attendez un instant—", "Ah, en fait—", "Attendez, désolé—"],
    "de": ["Entschuldigung, warten Sie—", "Moment mal—", "Einen Augenblick—", "Ach, eigentlich—", "Warten Sie, sorry—"],
    "it": ["Scusi, aspetti—", "Un attimo—", "Aspetti un secondo—", "Ah, in realtà—", "Aspetti, scusi—"],
    "fr-CA": [
        "Excusez-moi, attendez—",
        "Une petite seconde—",
        "Attendez une minute—",
        "Ah, en fait—",
        "Attendez, désolé—",
    ],
    "hi": ["माफ़ कीजिए, रुकिए—", "एक सेकंड—", "ज़रा रुकिए—", "अरे, असल में—", "रुकिए, माफ़ कीजिए—"],
    "ko": ["죄송한데, 잠깐만요—", "잠시만요—", "잠깐만요—", "아, 사실—", "잠깐, 죄송해요—"],
    "nl": ["Sorry, wacht even—", "Wacht even—", "Eén momentje—", "O, eigenlijk—", "Wacht, sorry—"],
    "ja": [
        "すみません、ちょっと待って—",
        "ちょっと待ってください—",
        "あの、すみません—",
        "あ、実は—",
        "待ってください、すみません—",
    ],
}
"""Pre-rendered at startup and played the moment a decision says interrupt, masking the reply's latency."""

BACKCHANNELS: dict[str, list[str]] = {
    "en": ["Mhm.", "Yeah.", "Uh-huh.", "Okay.", "Right."],
    "es": ["Ajá.", "Sí.", "Mm-hm.", "Vale.", "Claro."],
    "fr": ["Mhm.", "Oui.", "Ouais.", "D'accord.", "Je vois."],
    "de": ["Mhm.", "Ja.", "Aha.", "Okay.", "Genau."],
    "it": ["Mhm.", "Sì.", "Ah-ah.", "Okay.", "Certo."],
    "fr-CA": ["Mhm.", "Oui.", "Ouais.", "Ok.", "Correct."],
    "hi": ["हम्म।", "हाँ।", "जी।", "अच्छा।", "ठीक।"],
    "ko": ["음.", "네.", "아.", "응.", "그래요."],
    "nl": ["Mhm.", "Ja.", "Oké.", "Juist.", "Precies."],
    "ja": ["うん。", "はい。", "ええ。", "そう。", "なるほど。"],
}
"""Pre-rendered like OPENERS, and played when a decision says back_channel. Keep them to a word or two:
the caller counts as speaking while one plays."""


def phrases_for(table: dict[str, list[str]], language: str) -> list[str]:
    """The phrases for ``language`` (``pt-BR`` tries ``pt-BR``, then ``pt``), or the default language's if absent."""
    return table.get(language) or table.get(language.split("-")[0].lower()) or table[DEFAULT_LANGUAGE]
