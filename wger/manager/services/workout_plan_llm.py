# This file is part of wger Workout Manager.
#
# wger Workout Manager is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Optional Gemini fallback for workout plan prompt parsing."""

# Standard Library
import json
import logging
import re
from typing import Any

# Django
from django.conf import settings

# wger
from wger.exercises.services.prompt_parser import (
    CATEGORY_SYNONYMS,
    EQUIPMENT_SYNONYMS,
    MUSCLE_SYNONYMS,
)


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    'Normalize workout plan prompts into JSON only. '
    'Allowed goals: strength, fat_loss, hypertrophy, general, mobility, endurance. '
    'Allowed intensity: beginner, intermediate, advanced. '
    'Allowed equipment: barbell, dumbbell, bodyweight, bench, kettlebell, resistance band, '
    'pull-up bar, gym mat, swiss ball. '
    'Allowed target muscles: chest, legs, arms, abs, core, back, shoulders, calves, glutes, '
    'quads, hamstrings, lats, biceps, triceps. '
    'Return only JSON with keys: days_per_week, goal, intensity, equipment, target_muscles, '
    'workout_style, notes.'
)

_GOALS = {
    'strength': 'strength',
    'fat_loss': 'fat_loss',
    'hypertrophy': 'muscle_gain',
    'muscle_gain': 'muscle_gain',
    'general': 'general',
    'mobility': 'mobility',
    'endurance': 'endurance',
}
_LEVELS = {'beginner', 'intermediate', 'advanced'}
_WORKOUT_STYLES = {
    'full_body': [8, 9, 10, 11, 12, 13],
    'upper_lower': [8, 9, 11, 12, 13],
    'push_pull_legs': [8, 9, 11, 12, 13],
}


def _normalise(value: Any) -> str:
    return re.sub(r'\s+', ' ', str(value or '').lower().strip()).replace('_', ' ')


def _scan_name(name: str, synonym_map: dict[int, list[str]]) -> list[int]:
    norm = _normalise(name)
    matched = []
    for pk, synonyms in synonym_map.items():
        if any(_normalise(syn) == norm for syn in synonyms):
            matched.append(pk)
    return matched


def _list_values(data: dict, key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _call_gemini(prompt: str) -> dict | None:
    if not getattr(settings, 'ENABLE_WORKOUT_PLAN_LLM', False):
        return None
    if not getattr(settings, 'GEMINI_API_KEY', ''):
        return None

    try:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=getattr(settings, 'GEMINI_MODEL', 'gemini-2.5-flash-lite'),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=180,
                temperature=0.0,
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError as exc:
        logger.warning('Workout plan Gemini returned invalid JSON: %s', exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning('Workout plan Gemini call failed: %s', exc)
    return None


def _apply_validated_data(parsed, data: dict):
    changed = False

    try:
        days = int(data.get('days_per_week'))
        if 1 <= days <= 7 and 'days' in parsed.missing_fields:
            parsed.days = days
            changed = True
    except (TypeError, ValueError):
        pass

    goal = _GOALS.get(_normalise(data.get('goal')).replace(' ', '_'))
    if goal and (parsed.goal == 'general' or 'goal' in parsed.missing_fields):
        parsed.goal = goal
        changed = True

    level = _normalise(data.get('intensity'))
    if level in _LEVELS and 'intensity' in parsed.missing_fields:
        parsed.level = level
        changed = True

    equipment_ids = []
    for item in _list_values(data, 'equipment'):
        equipment_ids.extend(_scan_name(item, EQUIPMENT_SYNONYMS))
    if equipment_ids:
        merged = list(dict.fromkeys(parsed.equipment_ids + equipment_ids))
        if merged != parsed.equipment_ids:
            parsed.equipment_ids = merged
            changed = True

    muscle_ids = []
    category_ids = []
    for item in _list_values(data, 'target_muscles'):
        muscle_ids.extend(_scan_name(item, MUSCLE_SYNONYMS))
        category_ids.extend(_scan_name(item, CATEGORY_SYNONYMS))
    if muscle_ids:
        merged = list(dict.fromkeys(parsed.muscle_ids + muscle_ids))
        if merged != parsed.muscle_ids:
            parsed.muscle_ids = merged
            changed = True
    if category_ids:
        merged = list(dict.fromkeys(parsed.category_ids + category_ids))
        if merged != parsed.category_ids:
            parsed.category_ids = merged
            changed = True

    style = _normalise(data.get('workout_style')).replace(' ', '_')
    if style in _WORKOUT_STYLES:
        merged = list(dict.fromkeys(parsed.category_ids + _WORKOUT_STYLES[style]))
        if merged != parsed.category_ids:
            parsed.category_ids = merged
            changed = True

    note = str(data.get('notes') or '').strip()
    if note:
        parsed.llm_note = note[:200]

    return changed


def enhance_with_gemini(parsed, prompt: str):
    """Safely merge Gemini-normalized fields into a rule-based parsed result."""
    data = _call_gemini(prompt)
    if data is None:
        return parsed

    if _apply_validated_data(parsed, data):
        parsed.used_llm = True
        parsed.missing_fields = [
            field
            for field in parsed.missing_fields
            if not (
                field == 'days' and parsed.days
                or field == 'goal' and parsed.goal != 'general'
                or field == 'equipment' and parsed.equipment_ids
                or field == 'target_muscles' and (parsed.muscle_ids or parsed.category_ids)
                or field == 'intensity' and parsed.level != 'intermediate'
            )
        ]
        parsed.confidence = max(parsed.confidence, 0.9)
        parsed.explanation = f'{parsed.explanation} AI-assisted interpretation applied.'

    return parsed
