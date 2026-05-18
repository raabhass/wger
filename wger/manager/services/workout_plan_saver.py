# This file is part of wger Workout Manager.
#
# wger Workout Manager is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Save generated workout plan previews as regular wger routines."""

# Standard Library
import datetime
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

# Django
from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone

# wger
from wger.exercises.models import Exercise
from wger.manager.models import (
    Day,
    MaxRepetitionsConfig,
    RepetitionsConfig,
    RestConfig,
    Routine,
    SetsConfig,
    Slot,
    SlotEntry,
)
from wger.manager.models.abstract_config import OperationChoices


ROUTINE_WEEKS = 6
DEFAULT_SETS = 3
DEFAULT_REPS = 10


@dataclass
class SavedWorkoutPlan:
    routine: Routine
    warnings: list[str]
    saved_exercises: int


def _truncate(value: Any, max_length: int, fallback: str) -> str:
    text = str(value or '').strip() or fallback
    return text[:max_length]


def _parse_first_int(value: Any, default: int) -> int:
    match = re.search(r'\d+', str(value or ''))
    return int(match.group(0)) if match else default


def _parse_range(value: Any, default: int) -> tuple[int, int | None]:
    numbers = [int(match) for match in re.findall(r'\d+', str(value or ''))]
    if not numbers:
        return default, None
    if len(numbers) == 1:
        return numbers[0], None
    lower = min(numbers[0], numbers[1])
    upper = max(numbers[0], numbers[1])
    return lower, upper


def _exercise_map(plan: dict) -> dict[int, Exercise]:
    ids = {
        int(exercise['id'])
        for day in plan.get('days', [])
        for exercise in day.get('exercises', [])
        if str(exercise.get('id', '')).isdigit()
    }
    return Exercise.objects.in_bulk(ids)


def _create_configs(slot_entry: SlotEntry, exercise_data: dict) -> None:
    sets = max(1, min(_parse_first_int(exercise_data.get('sets'), DEFAULT_SETS), 50))
    reps, max_reps = _parse_range(exercise_data.get('reps'), DEFAULT_REPS)
    reps = max(1, min(reps, 3000))

    SetsConfig.objects.create(
        slot_entry=slot_entry,
        iteration=1,
        value=sets,
        operation=OperationChoices.REPLACE,
    )
    RepetitionsConfig.objects.create(
        slot_entry=slot_entry,
        iteration=1,
        value=Decimal(reps),
        operation=OperationChoices.REPLACE,
    )

    if max_reps and max_reps > reps:
        MaxRepetitionsConfig.objects.create(
            slot_entry=slot_entry,
            iteration=1,
            value=Decimal(min(max_reps, 3000)),
            operation=OperationChoices.REPLACE,
        )

    rest = _parse_first_int(exercise_data.get('rest'), 0)
    if rest:
        RestConfig.objects.create(
            slot_entry=slot_entry,
            iteration=1,
            value=max(1, min(rest, 1800)),
            operation=OperationChoices.REPLACE,
        )


@transaction.atomic
def save_generated_workout_plan(user: User, plan: dict) -> SavedWorkoutPlan:
    """
    Convert a generated plan preview into a regular routine.

    Unknown exercise IDs are skipped so stale browser data cannot break the
    save flow after exercises have changed in the database.
    """
    start = timezone.localdate()
    title = _truncate(plan.get('title'), 25, 'Generated routine')
    summary = str(plan.get('summary') or '').strip()
    days = plan.get('days') or []
    exercise_lookup = _exercise_map(plan)
    warnings = []
    saved_exercises = 0

    routine = Routine.objects.create(
        user=user,
        name=title,
        description=summary[:1000],
        start=start,
        end=start + datetime.timedelta(weeks=ROUTINE_WEEKS),
        fit_in_week=True,
    )

    for day_index, day_data in enumerate(days, start=1):
        day = Day.objects.create(
            routine=routine,
            order=day_index,
            name=_truncate(day_data.get('name'), 20, f'Day {day_index}'),
            description=str(day_data.get('focus') or '')[:1000],
        )

        for slot_index, exercise_data in enumerate(day_data.get('exercises', []), start=1):
            exercise_id = exercise_data.get('id')
            try:
                exercise_id = int(exercise_id)
            except (TypeError, ValueError):
                warnings.append(f'Skipped exercise with invalid ID: {exercise_id}')
                continue

            exercise = exercise_lookup.get(exercise_id)
            if exercise is None:
                warnings.append(f'Skipped missing exercise ID: {exercise_id}')
                continue

            slot = Slot.objects.create(day=day, order=slot_index)
            slot_entry = SlotEntry.objects.create(
                slot=slot,
                exercise=exercise,
                order=1,
                repetition_rounding=Decimal('1'),
                weight_rounding=Decimal('1'),
            )
            _create_configs(slot_entry, exercise_data)
            saved_exercises += 1

    return SavedWorkoutPlan(
        routine=routine,
        warnings=warnings,
        saved_exercises=saved_exercises,
    )
