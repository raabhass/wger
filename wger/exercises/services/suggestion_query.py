# This file is part of wger Workout Manager.
#
# wger Workout Manager is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wger Workout Manager is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Workout Manager.  If not, see <http://www.gnu.org/licenses/>.

"""
Query, ranking, and variety service for the exercise suggestion feature.

Entry point: query_exercises(filters, language) -> list[dict]

Strategy:
  1. Build relaxation passes (full → drop category → muscles only → equipment only).
  2. For each pass, fetch up to MAX_FETCH_PER_PASS candidates.
  3. Score each candidate against the original filters.
  4. Sort by score descending (best matches first).
  5. Fill the result list respecting:
       - global deduplication (no exercise appears twice)
       - per-category variety cap (MAX_PER_CATEGORY per category)
       - total cap (MAX_RESULTS)
  6. Move to the next (looser) pass only if results < MAX_RESULTS.
"""

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
from wger.exercises.services.prompt_parser import ParsedFilters
from wger.utils.constants import ENGLISH_SHORT_NAME

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

MAX_RESULTS = 20          # hard cap on result list length
MAX_PER_CATEGORY = 5      # variety cap: max exercises from one category
MAX_FETCH_PER_PASS = 100  # how many DB rows to pull per relaxation pass

# Scoring weights
_W_PRIMARY_MUSCLE = 3
_W_EQUIPMENT      = 2
_W_CATEGORY       = 2
_W_SECONDARY_MUSCLE = 1


# ---------------------------------------------------------------------------
# Internal queryset helpers
# ---------------------------------------------------------------------------

def _base_qs() -> QuerySet:
    """
    Exercises with at least one translation, all related data prefetched
    in a single round-trip so scoring and serialization are N+1-free.
    """
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


def _apply_filters(
    qs: QuerySet,
    equipment_ids: list[int],
    muscle_ids: list[int],
    category_ids: list[int],
    exclude_muscle_ids: list[int] = (),
    exclude_equipment_ids: list[int] = (),
) -> QuerySet:
    if equipment_ids:
        qs = qs.filter(equipment__in=equipment_ids)
    if category_ids:
        qs = qs.filter(category__in=category_ids)
    if muscle_ids:
        qs = qs.filter(
            Q(muscles__in=muscle_ids) | Q(muscles_secondary__in=muscle_ids)
        )
    # Exclusions — applied across all passes
    if exclude_muscle_ids:
        qs = qs.exclude(muscles__in=exclude_muscle_ids)
        qs = qs.exclude(muscles_secondary__in=exclude_muscle_ids)
    if exclude_equipment_ids:
        qs = qs.exclude(equipment__in=exclude_equipment_ids)
    return qs.distinct()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(exercise: Exercise, filters: ParsedFilters) -> int:
    """
    Score an exercise against the original parsed filters.
    Higher score = better match. Uses prefetched relations — no extra queries.

    Weights:
      Primary muscle match   +3 per muscle
      Equipment match        +2 per equipment item
      Category match         +2
      Secondary muscle match +1 per muscle
    """
    eq_wanted  = set(filters.equipment_ids)
    mu_wanted  = set(filters.muscle_ids)
    cat_wanted = set(filters.category_ids)

    score = 0

    # Primary muscles (prefetched)
    primary_ids = {m.id for m in exercise.muscles.all()}
    score += _W_PRIMARY_MUSCLE * len(mu_wanted & primary_ids)

    # Equipment (prefetched)
    equipment_ids = {e.id for e in exercise.equipment.all()}
    score += _W_EQUIPMENT * len(eq_wanted & equipment_ids)

    # Category
    if exercise.category_id in cat_wanted:
        score += _W_CATEGORY

    # Secondary muscles (prefetched)
    secondary_ids = {m.id for m in exercise.muscles_secondary.all()}
    score += _W_SECONDARY_MUSCLE * len(mu_wanted & secondary_ids)

    return score


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _get_name(exercise: Exercise, language: str) -> str:
    translation = exercise.get_translation(language)
    if translation:
        return translation.name
    return f'Exercise {exercise.id}'


def _serialize(exercise: Exercise, language: str) -> dict:
    return {
        'id':                exercise.id,
        'name':              _get_name(exercise, language),
        'url':               exercise.get_absolute_url(),
        'category':          exercise.category.name,
        'equipment':         [e.name for e in exercise.equipment.all()],
        'muscles':           [m.name_en or m.name for m in exercise.muscles.all()],
        'muscles_secondary': [m.name_en or m.name for m in exercise.muscles_secondary.all()],
    }


# ---------------------------------------------------------------------------
# Relaxation pass builder
# ---------------------------------------------------------------------------

def _build_passes(filters: ParsedFilters) -> list[tuple[list, list, list]]:
    """
    Returns a list of (equipment_ids, muscle_ids, category_ids) tuples,
    ordered from strictest to most relaxed.
    Later passes only run when earlier passes produce insufficient results.
    """
    e = filters.equipment_ids
    m = filters.muscle_ids
    c = filters.category_ids

    passes = []

    if e or m or c:
        passes.append((e, m, c))          # full match

    if c and (e or m):
        passes.append((e, m, []))         # drop category

    if m:
        passes.append(([], m, []))        # muscles only

    if e:
        passes.append((e, [], []))        # equipment only

    return passes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_exercises(
    filters: ParsedFilters,
    language: Optional[str] = None,
) -> list[dict]:
    """
    Return a ranked, curated list of up to MAX_RESULTS exercise dicts.

    Ranking: exercises that match more of the requested filters score higher
    and appear first.

    Variety: no more than MAX_PER_CATEGORY exercises from any single category,
    so broad queries don't collapse into 20 variations of the same movement.

    Args:
        filters:  Output of parse_prompt().
        language: BCP-47 short code (e.g. 'en', 'de'). Defaults to English.

    Returns:
        List of serialized exercise dicts, best matches first.
    """
    lang = language or ENGLISH_SHORT_NAME

    if filters.is_empty():
        return []

    seen_ids: set[int] = set()
    category_counts: dict[str, int] = {}
    results: list[dict] = []

    for eq_ids, mu_ids, cat_ids in _build_passes(filters):
        if len(results) >= MAX_RESULTS:
            break

        qs = _apply_filters(
            _base_qs(), eq_ids, mu_ids, cat_ids,
            exclude_muscle_ids=filters.exclude_muscle_ids,
            exclude_equipment_ids=filters.exclude_equipment_ids,
        )

        # Fetch a batch large enough for scoring + variety filtering to work
        candidates = [ex for ex in qs[:MAX_FETCH_PER_PASS] if ex.id not in seen_ids]

        # Sort by relevance score descending — best matches come first
        candidates.sort(key=lambda ex: _score(ex, filters), reverse=True)

        for exercise in candidates:
            if len(results) >= MAX_RESULTS:
                break

            # Always mark as seen so this exercise doesn't reappear in
            # looser passes, even if we skip it due to the variety cap
            seen_ids.add(exercise.id)

            cat = exercise.category.name
            if category_counts.get(cat, 0) >= MAX_PER_CATEGORY:
                continue  # variety cap hit for this category

            category_counts[cat] = category_counts.get(cat, 0) + 1
            results.append(_serialize(exercise, lang))

    return results
