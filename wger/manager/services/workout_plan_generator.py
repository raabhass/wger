# This file is part of wger Workout Manager.
#
# wger Workout Manager is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Preview-only workout plan generation using existing exercises."""

# Standard Library
from typing import Optional

# Django
from django.db.models import (
    Prefetch,
    Q,
    QuerySet,
)

# wger
from wger.exercises.models import (
    Exercise,
    Translation,
)
from wger.manager.services.workout_plan_prompt_parser import ParsedWorkoutPlanPrompt
from wger.utils.constants import ENGLISH_SHORT_NAME


MAX_OPTIONS = 3
EXERCISES_PER_DAY = 4
MAX_FETCH = 80

_CATEGORY_FOCUS = {
    8: 'Arms',
    9: 'Legs',
    10: 'Core',
    11: 'Chest',
    12: 'Back',
    13: 'Shoulders',
    14: 'Calves',
    15: 'Cardio',
}

_DEFAULT_SPLITS = {
    1: ['Full body'],
    2: ['Upper body', 'Lower body'],
    3: ['Push', 'Pull', 'Legs'],
    4: ['Upper body', 'Lower body', 'Push', 'Pull'],
    5: ['Push', 'Pull', 'Legs', 'Upper body', 'Conditioning'],
    6: ['Push', 'Pull', 'Legs', 'Upper body', 'Lower body', 'Conditioning'],
    7: ['Push', 'Pull', 'Legs', 'Upper body', 'Lower body', 'Conditioning', 'Mobility'],
}

_FOCUS_CATEGORY_IDS = {
    'Full body': [8, 9, 10, 11, 12, 13],
    'Upper body': [8, 11, 12, 13],
    'Lower body': [9, 14],
    'Push': [11, 13, 8],
    'Pull': [12, 8],
    'Legs': [9, 14],
    'Conditioning': [15, 9, 10],
    'Mobility': [10, 9, 13],
    'Core': [10],
    'Chest': [11],
    'Back': [12],
    'Shoulders': [13],
    'Arms': [8],
    'Calves': [14],
    'Cardio': [15],
}

_GOAL_PRESCRIPTIONS = {
    'strength': ('4', '4-6', '120 sec'),
    'muscle_gain': ('3', '8-12', '90 sec'),
    'fat_loss': ('3', '12-15', '45 sec'),
    'endurance': ('2', '15-20', '30 sec'),
    'mobility': ('2', '8-12', '30 sec'),
    'general': ('3', '8-12', '60 sec'),
}

_LEVEL_SET_OVERRIDE = {
    'beginner': '2',
    'advanced': '4',
}


def _base_qs() -> QuerySet:
    return (
        Exercise.with_translations.all()
        .select_related('category')
        .prefetch_related(
            'equipment',
            'muscles',
            'muscles_secondary',
            Prefetch(
                'translations',
                queryset=Translation.objects.select_related('language'),
            ),
        )
    )


def _exercise_name(exercise: Exercise, language: str) -> str:
    translation = exercise.get_translation(language)
    return translation.name if translation else f'Exercise {exercise.id}'


def _serialize_exercise(exercise: Exercise, language: str, prescription: dict) -> dict:
    return {
        'id': exercise.id,
        'name': _exercise_name(exercise, language),
        'url': exercise.get_absolute_url(),
        'category': exercise.category.name,
        'equipment': [equipment.name for equipment in exercise.equipment.all()],
        'muscles': [muscle.name_en or muscle.name for muscle in exercise.muscles.all()],
        'sets': prescription['sets'],
        'reps': prescription['reps'],
        'rest': prescription['rest'],
    }


def _prescription(parsed: ParsedWorkoutPlanPrompt, option_index: int) -> dict:
    sets, reps, rest = _GOAL_PRESCRIPTIONS.get(parsed.goal, _GOAL_PRESCRIPTIONS['general'])
    sets = _LEVEL_SET_OVERRIDE.get(parsed.level, sets)

    if option_index == 1 and parsed.goal in ('fat_loss', 'endurance'):
        rest = '30 sec'
    if option_index == 2 and parsed.level != 'beginner':
        sets = str(min(int(sets) + 1, 5))

    return {'sets': sets, 'reps': reps, 'rest': rest}


def _requested_focuses(parsed: ParsedWorkoutPlanPrompt) -> list[str]:
    focuses = [
        _CATEGORY_FOCUS[cat_id]
        for cat_id in parsed.category_ids
        if cat_id in _CATEGORY_FOCUS
    ]

    if parsed.muscle_ids and not focuses:
        focuses.append('Full body')

    if not focuses:
        focuses = list(_DEFAULT_SPLITS.get(parsed.days, _DEFAULT_SPLITS[3]))

    out = list(dict.fromkeys(focuses))
    while len(out) < parsed.days:
        for focus in _DEFAULT_SPLITS.get(parsed.days, _DEFAULT_SPLITS[3]):
            if len(out) >= parsed.days:
                break
            out.append(focus)
    return out[:parsed.days]


def _score(exercise: Exercise, parsed: ParsedWorkoutPlanPrompt, category_ids: list[int]) -> int:
    equipment_ids = {equipment.id for equipment in exercise.equipment.all()}
    primary_ids = {muscle.id for muscle in exercise.muscles.all()}
    secondary_ids = {muscle.id for muscle in exercise.muscles_secondary.all()}

    score = 0
    score += 3 * len(set(parsed.muscle_ids) & primary_ids)
    score += 2 * len(set(parsed.equipment_ids) & equipment_ids)
    score += 2 if exercise.category_id in category_ids else 0
    score += len(set(parsed.muscle_ids) & secondary_ids)
    return score


def _query_candidates(
    parsed: ParsedWorkoutPlanPrompt,
    focus: str,
    used_ids: set[int],
) -> list[Exercise]:
    category_ids = _FOCUS_CATEGORY_IDS.get(focus, [])
    qs = _base_qs()

    if parsed.equipment_ids:
        qs = qs.filter(Q(equipment__in=parsed.equipment_ids) | Q(equipment__isnull=True))

    focus_filter = Q()
    if category_ids:
        focus_filter |= Q(category__in=category_ids)
    if parsed.muscle_ids:
        focus_filter |= Q(muscles__in=parsed.muscle_ids) | Q(
            muscles_secondary__in=parsed.muscle_ids
        )
    if focus_filter:
        qs = qs.filter(focus_filter)

    candidates = [
        exercise
        for exercise in qs.distinct()[:MAX_FETCH]
        if exercise.id not in used_ids
    ]
    candidates.sort(key=lambda exercise: _score(exercise, parsed, category_ids), reverse=True)

    if len(candidates) >= EXERCISES_PER_DAY:
        return candidates

    fallback_qs = _base_qs()
    if parsed.equipment_ids:
        fallback_qs = fallback_qs.filter(
            Q(equipment__in=parsed.equipment_ids) | Q(equipment__isnull=True)
        )
    candidate_ids = {candidate.id for candidate in candidates}
    fallback = [
        exercise
        for exercise in fallback_qs.distinct()[:MAX_FETCH]
        if exercise.id not in used_ids and exercise.id not in candidate_ids
    ]
    fallback.sort(key=lambda exercise: _score(exercise, parsed, category_ids), reverse=True)
    return candidates + fallback


def _option_title(parsed: ParsedWorkoutPlanPrompt, index: int) -> str:
    goal = parsed.goal.replace('_', ' ').title()
    titles = ['Balanced', 'Efficient', 'Higher Volume']
    return f'{titles[index]} {parsed.days}-Day {goal} Plan'


def generate_workout_plan_options(
    parsed: ParsedWorkoutPlanPrompt,
    language: Optional[str] = None,
) -> list[dict]:
    """Generate 2-3 structured workout plan previews."""
    lang = language or ENGLISH_SHORT_NAME
    option_count = 2 if parsed.days <= 2 else MAX_OPTIONS
    focuses = _requested_focuses(parsed)
    options = []

    for option_index in range(option_count):
        used_ids: set[int] = set()
        prescription = _prescription(parsed, option_index)
        days = []

        rotated_focuses = focuses[option_index:] + focuses[:option_index]
        for day_index, focus in enumerate(rotated_focuses, start=1):
            exercises = []
            for exercise in _query_candidates(parsed, focus, used_ids)[:EXERCISES_PER_DAY]:
                used_ids.add(exercise.id)
                exercises.append(_serialize_exercise(exercise, lang, prescription))

            days.append(
                {
                    'day': day_index,
                    'name': f'Day {day_index}: {focus}',
                    'focus': focus,
                    'exercises': exercises,
                }
            )

        options.append(
            {
                'title': _option_title(parsed, option_index),
                'goal': parsed.goal,
                'level': parsed.level,
                'days_count': parsed.days,
                'summary': (
                    f'{parsed.level.title()} {parsed.goal.replace("_", " ")} preview '
                    f'with {parsed.days} training days.'
                ),
                'days': days,
            }
        )

    return options
