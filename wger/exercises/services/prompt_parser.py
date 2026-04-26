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
Prompt parser for the exercise suggestion feature.

Entry point: parse_prompt(text) -> ParsedFilters

Three-tier strategy
-------------------
1. Rule-based pass (always free, always runs):
     Extracts equipment/muscle/category by synonym matching.

2. Gemini enhancement (only when complex intent is detected):
     Handles exclusions ("bad shoulder"), scope ("full body"), and
     goal/level hints ("beginner", "weight loss") that rules cannot.
     Called even when rule-based found matches.

3. Gemini fallback (only when rule-based returned nothing):
     Handles unusual phrasing, non-English terms, or abstract goals
     that the synonym dictionary doesn't cover.

Gemini is NEVER called when:
  - GEMINI_API_KEY is blank
  - The daily call counter (stored in Django cache) has reached
    GEMINI_MAX_DAILY_CALLS (default 100)
  - No complex intent is detected AND the rule-based pass found filters
"""

# Standard Library
import json
import logging
import re
from dataclasses import (
    dataclass,
    field,
)
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lookup tables — PKs match wger/exercises/fixtures/ exactly
# ---------------------------------------------------------------------------

EQUIPMENT_SYNONYMS: dict[int, list[str]] = {
    1:  ['barbell', 'olympic bar', 'straight bar'],
    2:  ['sz-bar', 'sz bar', 'curl bar', 'ez bar', 'ez-bar', 'ez curl'],
    3:  ['dumbbell', 'dumbbells'],
    4:  ['gym mat', 'exercise mat', 'yoga mat', 'floor mat'],
    5:  ['swiss ball', 'stability ball', 'exercise ball', 'balance ball'],
    6:  ['pull-up bar', 'pullup bar', 'chin-up bar', 'chinup bar', 'pull up bar'],
    7:  ['bodyweight', 'body weight', 'no equipment', 'no gear',
         'calisthenics', 'bodyweight only', 'at home', 'home workout'],
    8:  ['bench', 'flat bench', 'weight bench'],
    9:  ['incline bench', 'incline', 'decline bench', 'decline'],
    10: ['kettlebell', 'kettlebells'],
    11: ['resistance band', 'resistance bands', 'elastic band', 'theraband'],
}

MUSCLE_SYNONYMS: dict[int, list[str]] = {
    1:  ['bicep', 'biceps', 'biceps brachii'],
    2:  ['anterior deltoid', 'front delt', 'front shoulder', 'front deltoid'],
    3:  ['serratus', 'serratus anterior'],
    4:  ['chest', 'pec', 'pecs', 'pectoral', 'pectorals', 'pectoralis'],
    5:  ['tricep', 'triceps', 'triceps brachii'],
    6:  ['abs', 'abdominals', 'abdominal', 'six pack', 'rectus abdominis'],
    7:  ['calf', 'calves', 'gastrocnemius'],
    8:  ['glute', 'glutes', 'gluteus', 'gluteal', 'butt', 'bum'],
    9:  ['trap', 'traps', 'trapezius'],
    10: ['quad', 'quads', 'quadricep', 'quadriceps', 'front thigh'],
    11: ['hamstring', 'hamstrings', 'biceps femoris', 'back of thigh'],
    12: ['lat', 'lats', 'latissimus', 'latissimus dorsi'],
    13: ['brachialis'],
    14: ['oblique', 'obliques', 'side abs', 'love handles'],
    15: ['soleus', 'deep calf'],
    16: ['lower back', 'erector', 'erector spinae', 'lumbar', 'spinal erector'],
}

CATEGORY_SYNONYMS: dict[int, list[str]] = {
    8:  ['arms', 'arm', 'upper arms'],
    9:  ['legs', 'leg', 'lower body'],
    10: ['abs', 'core', 'abdominals'],
    11: ['chest'],
    12: ['back', 'back muscles'],
    13: ['shoulders', 'shoulder', 'delts', 'delt', 'deltoids', 'deltoid'],
    14: ['calves', 'calf'],
    15: ['cardio', 'aerobic', 'running', 'jogging'],
}

_ARMS_MUSCLE_IDS: list[int] = [1, 5, 13]  # biceps, triceps, brachialis

_SOFT_HINTS: dict[str, str] = {
    'light':          'light',
    'easy':           'light',
    'beginner':       'light',
    'gentle':         'light',
    'simple':         'light',
    'low intensity':  'light',
    'heavy':          'heavy',
    'hard':           'heavy',
    'advanced':       'heavy',
    'intense':        'heavy',
    'high intensity': 'heavy',
    'compound':       'compound',
    'isolation':      'isolation',
}

# ---------------------------------------------------------------------------
# Intent triggers — phrases that Gemini handles but rules cannot
# ---------------------------------------------------------------------------

# When any of these appear in a normalised prompt, Gemini is called even if
# the rule-based pass found some filters (enhancement mode, not just fallback).
_INTENT_TRIGGERS: frozenset[str] = frozenset([
    # Exclusion / injury
    'injur', 'injury', 'pain', 'avoid', 'hurt', 'sore',
    'recovering', 'recovery', 'rehab', 'bad knee', 'bad back',
    'bad shoulder', 'bad hip', 'bad wrist', 'bad elbow',
    # Scope
    'full body', 'whole body', 'total body', 'full-body',
    # Level / goal
    'beginner', 'intermediate', 'advanced',
    'lose weight', 'weight loss', 'burn fat',
    'tone', 'toning', 'build muscle', 'bulk', 'cutting',
    # Context rules can't resolve
    'warm up', 'warm-up', 'cool down', 'cool-down',
    'posture', 'balance', 'coordination', 'flexibility', 'stretch',
    'functional', 'athletic', 'explosive', 'mobility',
])

# ---------------------------------------------------------------------------
# Compact Gemini system prompt (built once at import time)
# Kept short deliberately — every token saved is money saved.
# ---------------------------------------------------------------------------

_GEMINI_SYSTEM = (
    "You are a fitness assistant. Map a user's exercise request to IDs.\n"
    "Equipment: 1=Barbell,2=SZ-Bar,3=Dumbbell,4=Gym mat,5=Swiss Ball,"
    "6=Pull-up bar,7=Bodyweight,8=Bench,9=Incline bench,10=Kettlebell,11=Band.\n"
    "Muscles: 1=Biceps,2=Front delt,3=Serratus,4=Chest,5=Triceps,6=Abs,"
    "7=Calves,8=Glutes,9=Traps,10=Quads,11=Hamstrings,12=Lats,13=Brachialis,"
    "14=Obliques,15=Soleus,16=Lower back.\n"
    "Categories: 8=Arms,9=Legs,10=Abs,11=Chest,12=Back,13=Shoulders,14=Calves,15=Cardio.\n"
    "Rules:\n"
    "- If user mentions injury/pain/avoid for a body part, add that muscle to xm (exclude).\n"
    "- If user says full body, set c=[8,9,10,11,12,13].\n"
    "- If user says beginner, prefer e=[3,7] (dumbbell/bodyweight).\n"
    "- xm = muscle IDs to exclude, xe = equipment IDs to exclude.\n"
    'Reply ONLY with JSON: {"e":[...],"m":[...],"c":[...],"xm":[...],"xe":[...]}'
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParsedFilters:
    equipment_ids:         list[int] = field(default_factory=list)
    muscle_ids:            list[int] = field(default_factory=list)
    category_ids:          list[int] = field(default_factory=list)
    exclude_muscle_ids:    list[int] = field(default_factory=list)
    exclude_equipment_ids: list[int] = field(default_factory=list)
    name_search:           Optional[str] = None
    notes:                 list[str] = field(default_factory=list)
    explanation:           str = ''
    used_llm:              bool = False

    def is_empty(self) -> bool:
        return not (
            self.equipment_ids
            or self.muscle_ids
            or self.category_ids
            or self.name_search
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _build_explanation(
    equipment_ids: list[int],
    muscle_ids: list[int],
    category_ids: list[int],
    exclude_muscle_ids: list[int] = (),
    exclude_equipment_ids: list[int] = (),
) -> str:
    parts: list[str] = []
    if equipment_ids:
        names = [EQUIPMENT_SYNONYMS[i][0] for i in equipment_ids if i in EQUIPMENT_SYNONYMS]
        parts.append('equipment: ' + ', '.join(names))
    if muscle_ids:
        names = [MUSCLE_SYNONYMS[i][0] for i in muscle_ids if i in MUSCLE_SYNONYMS]
        parts.append('muscles: ' + ', '.join(names))
    if category_ids:
        names = [CATEGORY_SYNONYMS[i][0] for i in category_ids if i in CATEGORY_SYNONYMS]
        parts.append('category: ' + ', '.join(names))
    if exclude_muscle_ids:
        names = [MUSCLE_SYNONYMS[i][0] for i in exclude_muscle_ids if i in MUSCLE_SYNONYMS]
        parts.append('excluding muscles: ' + ', '.join(names))
    if exclude_equipment_ids:
        names = [EQUIPMENT_SYNONYMS[i][0] for i in exclude_equipment_ids if i in EQUIPMENT_SYNONYMS]
        parts.append('excluding equipment: ' + ', '.join(names))
    return ('Matched ' + '; '.join(parts) + '.') if parts else 'No filters matched.'


def _has_complex_intent(norm: str) -> bool:
    """Return True if the prompt contains intent that rules cannot resolve."""
    return any(trigger in norm for trigger in _INTENT_TRIGGERS)


# ---------------------------------------------------------------------------
# Rate limiter — uses Django cache (Redis in dev, memory in tests)
# ---------------------------------------------------------------------------

def _within_rate_limit() -> bool:
    """
    Check and increment the daily Gemini call counter.
    Returns True if we are within the limit, False if the cap is reached.
    """
    try:
        from django.conf import settings       # noqa: PLC0415
        from django.core.cache import cache    # noqa: PLC0415
        import datetime                        # noqa: PLC0415

        max_calls = getattr(settings, 'GEMINI_MAX_DAILY_CALLS', 100)
        today = datetime.date.today().isoformat()
        key = f'gemini_daily_calls_{today}'

        # Use add() + incr() to be atomic-safe with Redis
        cache.add(key, 0, timeout=86400)
        count = cache.incr(key)

        if count > max_calls:
            logger.warning(
                'Gemini daily limit (%d) reached (%d calls today). '
                'Using rule-based fallback.',
                max_calls, count,
            )
            return False
        return True

    except Exception as exc:  # noqa: BLE001
        # If cache is unavailable, allow the call rather than silently blocking
        logger.warning('Rate-limit cache check failed: %s', exc)
        return True


# ---------------------------------------------------------------------------
# Rule-based parser (free, deterministic, no network)
# ---------------------------------------------------------------------------

def _rule_based_parse(text: str) -> ParsedFilters:
    norm = _normalise(text)

    equipment_ids = _scan(norm, EQUIPMENT_SYNONYMS)
    muscle_ids    = _scan(norm, MUSCLE_SYNONYMS)
    category_ids  = _scan(norm, CATEGORY_SYNONYMS)

    if 8 in category_ids:
        for mid in _ARMS_MUSCLE_IDS:
            if mid not in muscle_ids:
                muscle_ids.append(mid)

    notes: list[str] = []
    for phrase, hint in sorted(_SOFT_HINTS.items(), key=lambda x: -len(x[0])):
        if phrase in norm and hint not in notes:
            notes.append(hint)

    explanation = _build_explanation(equipment_ids, muscle_ids, category_ids)

    return ParsedFilters(
        equipment_ids=equipment_ids,
        muscle_ids=muscle_ids,
        category_ids=category_ids,
        notes=notes,
        explanation=explanation,
        used_llm=False,
    )


# ---------------------------------------------------------------------------
# Gemini call (paid — guarded by rate limiter)
# ---------------------------------------------------------------------------

def _call_gemini(text: str) -> Optional[dict]:
    """
    Send text to Gemini and return the parsed JSON dict, or None on failure.
    Caller is responsible for checking the rate limit before calling this.
    """
    try:
        from google import genai                # noqa: PLC0415
        from google.genai import types          # noqa: PLC0415
        from django.conf import settings        # noqa: PLC0415

        api_key    = getattr(settings, 'GEMINI_API_KEY', '')
        model_name = getattr(settings, 'GEMINI_MODEL', 'gemini-2.5-flash-lite')

        if not api_key:
            return None

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=_GEMINI_SYSTEM,
                max_output_tokens=100,
                temperature=0.0,
            ),
        )

        raw = response.text.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        return json.loads(raw)

    except json.JSONDecodeError as exc:
        logger.warning('Gemini returned non-JSON: %s', exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning('Gemini call failed: %s', exc)

    return None


def _validated_ids(raw_list: list, lookup: dict) -> list[int]:
    """Coerce to int and keep only IDs that exist in the lookup table."""
    result = []
    for item in raw_list:
        try:
            pk = int(item)
            if pk in lookup:
                result.append(pk)
        except (TypeError, ValueError):
            pass
    return result


def _gemini_parse(text: str) -> ParsedFilters:
    """Full parse: Gemini extracts everything from scratch (fallback path)."""
    if not _within_rate_limit():
        return ParsedFilters(explanation='No filters matched.')

    data = _call_gemini(text)
    if data is None:
        return ParsedFilters(explanation='No filters matched.')

    equipment_ids         = _validated_ids(data.get('e',  []), EQUIPMENT_SYNONYMS)
    muscle_ids            = _validated_ids(data.get('m',  []), MUSCLE_SYNONYMS)
    category_ids          = _validated_ids(data.get('c',  []), CATEGORY_SYNONYMS)
    exclude_muscle_ids    = _validated_ids(data.get('xm', []), MUSCLE_SYNONYMS)
    exclude_equipment_ids = _validated_ids(data.get('xe', []), EQUIPMENT_SYNONYMS)

    if 8 in category_ids:
        for mid in _ARMS_MUSCLE_IDS:
            if mid not in muscle_ids:
                muscle_ids.append(mid)

    explanation = _build_explanation(
        equipment_ids, muscle_ids, category_ids,
        exclude_muscle_ids, exclude_equipment_ids,
    )

    return ParsedFilters(
        equipment_ids=equipment_ids,
        muscle_ids=muscle_ids,
        category_ids=category_ids,
        exclude_muscle_ids=exclude_muscle_ids,
        exclude_equipment_ids=exclude_equipment_ids,
        explanation=explanation or 'Gemini could not extract filters.',
        used_llm=True,
    )


def _gemini_enhance(base: ParsedFilters, text: str) -> ParsedFilters:
    """
    Enhancement pass: Gemini adds exclusions and category expansions
    on top of what the rule-based parser already found.

    Conservative strategy:
    - Equipment/muscle inclusions: keep ONLY the rule-based results.
      Gemini often over-includes in enhancement mode (e.g. returns all 16
      muscles for a "bad shoulder" prompt) which bloats and weakens the query.
    - Category inclusions: take the union (Gemini is reliable for "full body"
      expansions like c=[8,9,10,11,12,13]).
    - Exclusions: come entirely from Gemini.
    - Safety: never exclude an ID that the user explicitly included.
    """
    if not _within_rate_limit():
        return base

    data = _call_gemini(text)
    if data is None:
        return base

    # Categories: rule-based union Gemini (reliable for "full body" expansion)
    gemini_cat = _validated_ids(data.get('c', []), CATEGORY_SYNONYMS)
    merged_cat = list(dict.fromkeys(base.category_ids + gemini_cat))

    # Muscle inclusions: keep rule-based only
    merged_mu = list(base.muscle_ids)
    if 8 in merged_cat:
        for mid in _ARMS_MUSCLE_IDS:
            if mid not in merged_mu:
                merged_mu.append(mid)

    # Exclusions come entirely from Gemini
    exclude_muscle_ids    = _validated_ids(data.get('xm', []), MUSCLE_SYNONYMS)
    exclude_equipment_ids = _validated_ids(data.get('xe', []), EQUIPMENT_SYNONYMS)

    # Safety: Gemini sometimes excludes what the user explicitly asked for.
    # Never exclude an ID that is already in the inclusion list.
    include_eq_set = set(base.equipment_ids)
    include_mu_set = set(merged_mu)
    exclude_equipment_ids = [i for i in exclude_equipment_ids if i not in include_eq_set]
    exclude_muscle_ids    = [i for i in exclude_muscle_ids    if i not in include_mu_set]

    explanation = _build_explanation(
        base.equipment_ids, merged_mu, merged_cat,
        exclude_muscle_ids, exclude_equipment_ids,
    )

    return ParsedFilters(
        equipment_ids=base.equipment_ids,
        muscle_ids=merged_mu,
        category_ids=merged_cat,
        exclude_muscle_ids=exclude_muscle_ids,
        exclude_equipment_ids=exclude_equipment_ids,
        notes=base.notes,
        explanation=explanation,
        used_llm=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_prompt(text: str) -> ParsedFilters:
    """
    Convert a free-text exercise prompt into structured filter IDs.

    Gemini is called only when necessary and only within the daily rate limit:
      - enhancement: rule-based found filters but complex intent detected
      - fallback:    rule-based found nothing at all

    Args:
        text: Raw user input.

    Returns:
        ParsedFilters with include/exclude IDs, notes, explanation, used_llm flag.
    """
    if not text or not text.strip():
        return ParsedFilters(explanation='Empty prompt.')

    result = _rule_based_parse(text)
    norm   = _normalise(text)

    if result.is_empty():
        # Fallback: Gemini handles the whole parse
        logger.debug('Rule-based empty — Gemini fallback.')
        return _gemini_parse(text)

    if _has_complex_intent(norm):
        # Enhancement: Gemini adds exclusions / intent adjustments
        logger.debug('Complex intent detected — Gemini enhancement.')
        return _gemini_enhance(result, text)

    return result
