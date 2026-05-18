# This file is part of wger Workout Manager.
#
# wger Workout Manager is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Deterministic prompt parser for workout plan generation."""

# Standard Library
import re
from dataclasses import (
    dataclass,
    field,
)

# wger
from wger.exercises.services.prompt_parser import (
    CATEGORY_SYNONYMS,
    EQUIPMENT_SYNONYMS,
    MUSCLE_SYNONYMS,
)


DEFAULT_DAYS = 3
MIN_DAYS = 1
MAX_DAYS = 7

_NUMBER_WORDS = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
}

_GOAL_SYNONYMS = {
    'strength': ['strength', 'strong', 'power', 'powerlifting', 'heavy'],
    'muscle_gain': [
        'build muscle',
        'muscle gain',
        'gain muscle',
        'hypertrophy',
        'bulk',
        'size',
    ],
    'fat_loss': [
        'fat loss',
        'lose fat',
        'lose weight',
        'weight loss',
        'burn fat',
        'cutting',
        'tone',
        'toning',
    ],
    'endurance': ['endurance', 'conditioning', 'stamina', 'cardio', 'aerobic'],
    'mobility': ['mobility', 'flexibility', 'stretch', 'recovery'],
}

_LEVEL_SYNONYMS = {
    'beginner': ['beginner', 'new', 'starter', 'easy', 'simple'],
    'intermediate': ['intermediate', 'moderate'],
    'advanced': ['advanced', 'experienced', 'hard', 'intense'],
}


@dataclass
class ParsedWorkoutPlanPrompt:
    days: int = DEFAULT_DAYS
    goal: str = 'general'
    equipment_ids: list[int] = field(default_factory=list)
    muscle_ids: list[int] = field(default_factory=list)
    category_ids: list[int] = field(default_factory=list)
    level: str = 'intermediate'
    notes: list[str] = field(default_factory=list)
    explanation: str = ''
    confidence: float = 0.0
    missing_fields: list[str] = field(default_factory=list)
    used_llm: bool = False
    llm_note: str = ''


def _normalise(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())


def _scan(text: str, synonym_map: dict[int, list[str]]) -> list[int]:
    matched: list[int] = []
    for pk, synonyms in synonym_map.items():
        for syn in sorted(synonyms, key=len, reverse=True):
            if re.search(r'(?<!\w)' + re.escape(syn) + r'(?!\w)', text):
                matched.append(pk)
                break
    return matched


def _parse_days(text: str) -> int:
    patterns = [
        r'(?P<days>[1-7])\s*(?:day|days|x|times)(?:\s*(?:per|a)\s*week)?',
        r'(?P<days>[1-7])[-\s]*day',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group('days'))

    for word, value in _NUMBER_WORDS.items():
        if re.search(rf'(?<!\w){word}\s*(?:day|days|times)(?!\w)', text):
            return value

    return DEFAULT_DAYS


def _has_days(text: str) -> bool:
    if any(
        re.search(pattern, text)
        for pattern in (
            r'[1-7]\s*(?:day|days|x|times)(?:\s*(?:per|a)\s*week)?',
            r'[1-7][-\s]*day',
        )
    ):
        return True

    return any(
        re.search(rf'(?<!\w){word}\s*(?:day|days|times)(?!\w)', text)
        for word in _NUMBER_WORDS
    )


def _parse_goal(text: str) -> str:
    for goal, synonyms in _GOAL_SYNONYMS.items():
        if any(syn in text for syn in synonyms):
            return goal
    return 'general'


def _parse_level(text: str) -> str:
    for level, synonyms in _LEVEL_SYNONYMS.items():
        if any(re.search(r'(?<!\w)' + re.escape(syn) + r'(?!\w)', text) for syn in synonyms):
            return level
    return 'intermediate'


def _build_explanation(parsed: ParsedWorkoutPlanPrompt) -> str:
    parts = [f'{parsed.days} day plan', parsed.goal.replace('_', ' '), parsed.level]

    if parsed.equipment_ids:
        names = [EQUIPMENT_SYNONYMS[i][0] for i in parsed.equipment_ids if i in EQUIPMENT_SYNONYMS]
        parts.append('equipment: ' + ', '.join(names))
    if parsed.muscle_ids:
        names = [MUSCLE_SYNONYMS[i][0] for i in parsed.muscle_ids if i in MUSCLE_SYNONYMS]
        parts.append('muscles: ' + ', '.join(names))
    if parsed.category_ids:
        names = [CATEGORY_SYNONYMS[i][0] for i in parsed.category_ids if i in CATEGORY_SYNONYMS]
        parts.append('categories: ' + ', '.join(names))

    return 'Matched ' + '; '.join(parts) + '.'


def _missing_fields(parsed: ParsedWorkoutPlanPrompt, has_days: bool) -> list[str]:
    missing = []
    if not has_days:
        missing.append('days')
    if parsed.goal == 'general':
        missing.append('goal')
    if not parsed.equipment_ids:
        missing.append('equipment')
    if not parsed.muscle_ids and not parsed.category_ids:
        missing.append('target_muscles')
    if parsed.level == 'intermediate':
        missing.append('intensity')
    return missing


def _confidence(missing_fields: list[str]) -> float:
    return max(0.0, round(1 - (len(missing_fields) / 5), 2))


def _has_vague_intent(text: str) -> bool:
    triggers = (
        'tone up',
        'lean body',
        'leaner',
        'athletic',
        'not too hard',
        'home friendly',
        'get fit',
        'fit body',
        'in shape',
        'healthy',
    )
    return any(trigger in text for trigger in triggers)


def parse_workout_plan_prompt(
    text: str,
    allow_llm: bool = True,
    force_llm: bool = False,
) -> ParsedWorkoutPlanPrompt:
    """Convert a workout plan prompt into structured fields, rule-based first."""
    if not text or not text.strip():
        return ParsedWorkoutPlanPrompt(
            explanation='Empty prompt.',
            missing_fields=['days', 'goal', 'equipment', 'target_muscles', 'intensity'],
        )

    norm = _normalise(text)
    has_days = _has_days(norm)
    days = max(MIN_DAYS, min(MAX_DAYS, _parse_days(norm)))
    equipment_ids = _scan(norm, EQUIPMENT_SYNONYMS)
    category_ids = _scan(norm, CATEGORY_SYNONYMS)

    if not equipment_ids and any(phrase in norm for phrase in ('home', 'no gym', 'no equipment')):
        equipment_ids.append(7)

    notes: list[str] = []
    if any(phrase in norm for phrase in ('full body', 'full-body', 'whole body', 'total body')):
        for category_id in (8, 9, 10, 11, 12, 13):
            if category_id not in category_ids:
                category_ids.append(category_id)
        notes.append('full_body')

    parsed = ParsedWorkoutPlanPrompt(
        days=days,
        goal=_parse_goal(norm),
        equipment_ids=equipment_ids,
        muscle_ids=_scan(norm, MUSCLE_SYNONYMS),
        category_ids=category_ids,
        level=_parse_level(norm),
        notes=notes,
    )
    parsed.missing_fields = _missing_fields(parsed, has_days)
    parsed.confidence = _confidence(parsed.missing_fields)
    parsed.explanation = _build_explanation(parsed)

    if allow_llm and (force_llm or parsed.confidence < 0.75 or _has_vague_intent(norm)):
        from wger.manager.services.workout_plan_llm import enhance_with_gemini  # noqa: PLC0415

        parsed = enhance_with_gemini(parsed, text)

    return parsed
