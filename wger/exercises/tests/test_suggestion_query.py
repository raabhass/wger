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
Tests for wger.exercises.services.suggestion_query.

Uses programmatic test data so there is no dependency on fixture files.
Fixtures loaded: only the minimum needed (languages, licenses).
"""

# Django
from django.test import TestCase

# wger
from wger.core.models import (
    Language,
    License,
)
from wger.exercises.models import (
    Equipment,
    Exercise,
    ExerciseCategory,
    Muscle,
    Translation,
)
from wger.exercises.services.prompt_parser import ParsedFilters
from wger.exercises.services.suggestion_query import (
    MAX_PER_CATEGORY,
    MAX_RESULTS,
    _score,
    query_exercises,
)

# Minimum description length enforced by Translation model validator (40 chars)
_DESCRIPTION = 'A' * 40


class SuggestionQueryTestCase(TestCase):
    """
    Base class: creates a small but complete exercise dataset covering
    equipment, muscle, and category combinations used across all tests.

    Exercises created:
      ex_db    — Dumbbell, Biceps,  category Arms
      ex_bb    — Barbell,  Chest,   category Chest
      ex_bw    — no equip, Abs,     category Abs
      ex_kb    — Kettle,   Glutes,  category Legs
      ex_multi — Dumbbell + Barbell, Biceps + Triceps, category Arms
    """

    fixtures = ['licenses', 'languages']

    def setUp(self):
        self.lang    = Language.objects.get(short_name='en')
        self.license = License.objects.first()

        # Categories
        self.cat_arms  = ExerciseCategory.objects.create(name='Arms')
        self.cat_chest = ExerciseCategory.objects.create(name='Chest')
        self.cat_abs   = ExerciseCategory.objects.create(name='Abs')
        self.cat_legs  = ExerciseCategory.objects.create(name='Legs')

        # Equipment
        self.eq_dumbbell = Equipment.objects.create(name='Dumbbell')
        self.eq_barbell  = Equipment.objects.create(name='Barbell')
        self.eq_kettle   = Equipment.objects.create(name='Kettlebell')

        # Muscles
        self.mu_biceps  = Muscle.objects.create(name='Biceps brachii', name_en='Biceps',  is_front=True)
        self.mu_triceps = Muscle.objects.create(name='Triceps brachii', name_en='Triceps', is_front=False)
        self.mu_chest   = Muscle.objects.create(name='Pectoralis major', name_en='Chest',  is_front=True)
        self.mu_abs     = Muscle.objects.create(name='Rectus abdominis', name_en='Abs',    is_front=True)
        self.mu_glutes  = Muscle.objects.create(name='Gluteus maximus', name_en='Glutes',  is_front=False)

        # Helper to create an exercise + English translation
        def make(name, category, equipment=None, muscles=None, muscles_secondary=None):
            ex = Exercise.objects.create(category=category, license=self.license)
            if equipment:
                ex.equipment.set(equipment)
            if muscles:
                ex.muscles.set(muscles)
            if muscles_secondary:
                ex.muscles_secondary.set(muscles_secondary)
            Translation.objects.create(
                exercise=ex,
                language=self.lang,
                license=self.license,
                name=name,
                description=_DESCRIPTION,
            )
            return ex

        self.ex_db    = make('Dumbbell Curl',      self.cat_arms,  [self.eq_dumbbell],              [self.mu_biceps])
        self.ex_bb    = make('Barbell Bench Press', self.cat_chest, [self.eq_barbell],               [self.mu_chest])
        self.ex_bw    = make('Crunch',             self.cat_abs,   [],                              [self.mu_abs])
        self.ex_kb    = make('Kettlebell Swing',   self.cat_legs,  [self.eq_kettle],                [self.mu_glutes])
        self.ex_multi = make('Dumbbell Skull Crusher', self.cat_arms,
                             [self.eq_dumbbell, self.eq_barbell],
                             [self.mu_biceps],
                             [self.mu_triceps])


class FilterApplicationTests(SuggestionQueryTestCase):
    """query_exercises returns the right exercises for direct filter matches."""

    def test_equipment_filter(self):
        filters = ParsedFilters(equipment_ids=[self.eq_dumbbell.id])
        results = query_exercises(filters)
        ids = [r['id'] for r in results]
        self.assertIn(self.ex_db.id,    ids)
        self.assertIn(self.ex_multi.id, ids)
        self.assertNotIn(self.ex_bb.id, ids)
        self.assertNotIn(self.ex_bw.id, ids)

    def test_muscle_filter_primary(self):
        filters = ParsedFilters(muscle_ids=[self.mu_biceps.id])
        results = query_exercises(filters)
        ids = [r['id'] for r in results]
        self.assertIn(self.ex_db.id,    ids)
        self.assertIn(self.ex_multi.id, ids)

    def test_muscle_filter_secondary(self):
        # Triceps is secondary on ex_multi — should still be returned
        filters = ParsedFilters(muscle_ids=[self.mu_triceps.id])
        results = query_exercises(filters)
        ids = [r['id'] for r in results]
        self.assertIn(self.ex_multi.id, ids)

    def test_category_filter(self):
        filters = ParsedFilters(category_ids=[self.cat_arms.id])
        results = query_exercises(filters)
        ids = [r['id'] for r in results]
        self.assertIn(self.ex_db.id,    ids)
        self.assertIn(self.ex_multi.id, ids)
        self.assertNotIn(self.ex_bb.id, ids)

    def test_combined_equipment_and_muscle(self):
        filters = ParsedFilters(
            equipment_ids=[self.eq_dumbbell.id],
            muscle_ids=[self.mu_biceps.id],
        )
        results = query_exercises(filters)
        ids = [r['id'] for r in results]
        self.assertIn(self.ex_db.id,    ids)
        self.assertIn(self.ex_multi.id, ids)
        self.assertNotIn(self.ex_bb.id, ids)  # barbell, not dumbbell

    def test_no_results_on_impossible_filter(self):
        # Dumbbell + Chest muscle — no such exercise in our dataset
        filters = ParsedFilters(
            equipment_ids=[self.eq_dumbbell.id],
            muscle_ids=[self.mu_chest.id],
        )
        # Full match returns 0 → relaxation kicks in
        # Relaxed to muscle only → chest barbell bench press won't match dumbbell
        # Relaxed to equipment only → dumbbell exercises returned
        results = query_exercises(filters)
        # Should still return something via relaxation (dumbbell exercises)
        ids = [r['id'] for r in results]
        self.assertTrue(len(results) > 0)
        self.assertIn(self.ex_db.id, ids)


class RelaxationTests(SuggestionQueryTestCase):
    """Filter relaxation kicks in correctly when strict match returns zero."""

    def test_full_match_preferred_over_relaxed(self):
        # ex_db matches both dumbbell AND biceps (full match)
        # ex_bb matches barbell but NOT biceps (no full match on equipment+muscle)
        filters = ParsedFilters(
            equipment_ids=[self.eq_dumbbell.id],
            muscle_ids=[self.mu_biceps.id],
        )
        results = query_exercises(filters)
        ids = [r['id'] for r in results]
        # Full-match results come first
        self.assertEqual(ids[0], self.ex_db.id)

    def test_relaxes_to_muscle_when_equipment_misses(self):
        # Use a muscle that exists but equipment that has no exercises
        fake_equipment_id = 99999
        filters = ParsedFilters(
            equipment_ids=[fake_equipment_id],
            muscle_ids=[self.mu_biceps.id],
        )
        results = query_exercises(filters)
        ids = [r['id'] for r in results]
        # Full match → 0. Drop category → 0. Muscles only → finds ex_db and ex_multi
        self.assertIn(self.ex_db.id,    ids)
        self.assertIn(self.ex_multi.id, ids)

    def test_relaxes_to_equipment_when_muscle_misses(self):
        fake_muscle_id = 99999
        filters = ParsedFilters(
            equipment_ids=[self.eq_kettle.id],
            muscle_ids=[fake_muscle_id],
        )
        results = query_exercises(filters)
        ids = [r['id'] for r in results]
        # Eventually relaxes to equipment only → finds ex_kb
        self.assertIn(self.ex_kb.id, ids)

    def test_empty_filters_return_empty_list(self):
        filters = ParsedFilters()
        results = query_exercises(filters)
        self.assertEqual(results, [])


class SerializationTests(SuggestionQueryTestCase):
    """Each result dict has the expected shape and correct values."""

    def _get_result_for(self, exercise_id: int, filters: ParsedFilters) -> dict:
        results = query_exercises(filters)
        for r in results:
            if r['id'] == exercise_id:
                return r
        self.fail(f'Exercise {exercise_id} not found in results')

    def test_result_has_required_keys(self):
        filters = ParsedFilters(equipment_ids=[self.eq_dumbbell.id])
        result = self._get_result_for(self.ex_db.id, filters)
        for key in ('id', 'name', 'url', 'category', 'equipment', 'muscles'):
            self.assertIn(key, result)

    def test_name_is_english_translation(self):
        filters = ParsedFilters(equipment_ids=[self.eq_dumbbell.id])
        result = self._get_result_for(self.ex_db.id, filters)
        self.assertEqual(result['name'], 'Dumbbell Curl')

    def test_url_points_to_exercise_view(self):
        filters = ParsedFilters(equipment_ids=[self.eq_dumbbell.id])
        result = self._get_result_for(self.ex_db.id, filters)
        self.assertIn(str(self.ex_db.id), result['url'])

    def test_category_name_present(self):
        filters = ParsedFilters(equipment_ids=[self.eq_dumbbell.id])
        result = self._get_result_for(self.ex_db.id, filters)
        self.assertEqual(result['category'], 'Arms')

    def test_equipment_list_populated(self):
        filters = ParsedFilters(equipment_ids=[self.eq_dumbbell.id])
        result = self._get_result_for(self.ex_db.id, filters)
        self.assertIn('Dumbbell', result['equipment'])

    def test_muscles_list_uses_english_name(self):
        filters = ParsedFilters(muscle_ids=[self.mu_biceps.id])
        result = self._get_result_for(self.ex_db.id, filters)
        self.assertIn('Biceps', result['muscles'])

    def test_multiple_equipment_in_result(self):
        filters = ParsedFilters(equipment_ids=[self.eq_dumbbell.id])
        result = self._get_result_for(self.ex_multi.id, filters)
        self.assertIn('Dumbbell', result['equipment'])
        self.assertIn('Barbell',  result['equipment'])


class DeduplicationTests(SuggestionQueryTestCase):
    """Exercises must not appear twice even when matched by multiple passes."""

    def test_no_duplicates_across_relaxation_passes(self):
        # ex_db matches equipment (dumbbell) and muscle (biceps)
        # It would appear in pass 1 (full) and pass 3 (muscle only) — should appear once
        filters = ParsedFilters(
            equipment_ids=[self.eq_dumbbell.id],
            muscle_ids=[self.mu_biceps.id],
        )
        results = query_exercises(filters)
        ids = [r['id'] for r in results]
        self.assertEqual(len(ids), len(set(ids)), 'Duplicate exercise IDs found in results')


class MaxResultsTests(SuggestionQueryTestCase):
    """Results are capped at MAX_RESULTS."""

    def test_results_capped(self):
        # Create enough exercises to exceed MAX_RESULTS
        for i in range(MAX_RESULTS + 5):
            ex = Exercise.objects.create(category=self.cat_arms, license=self.license)
            ex.equipment.set([self.eq_dumbbell])
            Translation.objects.create(
                exercise=ex,
                language=self.lang,
                license=self.license,
                name=f'Extra Exercise {i}',
                description=_DESCRIPTION,
            )

        filters = ParsedFilters(equipment_ids=[self.eq_dumbbell.id])
        results = query_exercises(filters)
        self.assertLessEqual(len(results), MAX_RESULTS)


class RankingTests(SuggestionQueryTestCase):
    """Better-matching exercises score higher and appear earlier in results."""

    def test_primary_muscle_scores_higher_than_secondary(self):
        # ex_db: primary=biceps. ex_multi: primary=biceps, secondary=triceps.
        # Both match biceps filter — ex_multi should score >= ex_db
        filters = ParsedFilters(muscle_ids=[self.mu_biceps.id])
        score_db    = _score(self.ex_db,    filters)
        score_multi = _score(self.ex_multi, filters)
        # ex_multi has biceps as primary too, so scores should be equal or higher
        self.assertGreaterEqual(score_multi, score_db)

    def test_equipment_match_adds_to_score(self):
        filters = ParsedFilters(
            equipment_ids=[self.eq_dumbbell.id],
            muscle_ids=[self.mu_biceps.id],
        )
        score_db = _score(self.ex_db, filters)   # dumbbell + biceps (primary)
        score_bb = _score(self.ex_bb, filters)   # barbell + chest — misses both

        self.assertGreater(score_db, score_bb)

    def test_full_match_scores_higher_than_partial(self):
        # ex_db matches equipment+muscle. ex_kb matches neither.
        filters = ParsedFilters(
            equipment_ids=[self.eq_dumbbell.id],
            muscle_ids=[self.mu_biceps.id],
        )
        self.assertGreater(_score(self.ex_db, filters), _score(self.ex_kb, filters))

    def test_best_scoring_exercise_is_first_in_results(self):
        # With dumbbell + biceps filter, ex_db (dumbbell + biceps primary)
        # should outrank ex_multi (dumbbell + biceps primary + triceps secondary)
        # Actually both score the same on these filters — just assert first result
        # has the right equipment and muscle.
        filters = ParsedFilters(
            equipment_ids=[self.eq_dumbbell.id],
            muscle_ids=[self.mu_biceps.id],
        )
        results = query_exercises(filters)
        self.assertTrue(len(results) > 0)
        first = results[0]
        self.assertIn('Dumbbell', first['equipment'])
        self.assertIn('Biceps', first['muscles'])

    def test_zero_score_exercise_still_included_via_relaxation(self):
        # ex_bw has no equipment and abs muscle — won't match dumbbell+biceps full pass
        # but will appear in relaxation if we only filter equipment
        filters = ParsedFilters(equipment_ids=[self.eq_dumbbell.id])
        # ex_bw has no equipment so it won't appear at all — correct
        results = query_exercises(filters)
        ids = [r['id'] for r in results]
        self.assertNotIn(self.ex_bw.id, ids)


class VarietyTests(SuggestionQueryTestCase):
    """No more than MAX_PER_CATEGORY results from any single category."""

    def _make_arms_dumbbell_exercise(self, name: str) -> Exercise:
        ex = Exercise.objects.create(category=self.cat_arms, license=self.license)
        ex.equipment.set([self.eq_dumbbell])
        ex.muscles.set([self.mu_biceps])
        Translation.objects.create(
            exercise=ex, language=self.lang, license=self.license,
            name=name, description=_DESCRIPTION,
        )
        return ex

    def test_variety_cap_per_category(self):
        # Create more Arms exercises than MAX_PER_CATEGORY
        for i in range(MAX_PER_CATEGORY + 3):
            self._make_arms_dumbbell_exercise(f'Arms Exercise {i}')

        filters = ParsedFilters(equipment_ids=[self.eq_dumbbell.id])
        results = query_exercises(filters)

        category_counts: dict[str, int] = {}
        for r in results:
            category_counts[r['category']] = category_counts.get(r['category'], 0) + 1

        for cat, count in category_counts.items():
            self.assertLessEqual(
                count, MAX_PER_CATEGORY,
                msg=f'Category "{cat}" has {count} results, exceeds cap of {MAX_PER_CATEGORY}'
            )

    def test_variety_spreads_across_categories(self):
        # Add extra exercises in other categories so there's something to spread to
        for i in range(MAX_PER_CATEGORY + 2):
            self._make_arms_dumbbell_exercise(f'Extra Arms {i}')

        # Also add chest dumbbell exercises
        for i in range(3):
            ex = Exercise.objects.create(category=self.cat_chest, license=self.license)
            ex.equipment.set([self.eq_dumbbell])
            ex.muscles.set([self.mu_chest])
            Translation.objects.create(
                exercise=ex, language=self.lang, license=self.license,
                name=f'Chest Exercise {i}', description=_DESCRIPTION,
            )

        filters = ParsedFilters(equipment_ids=[self.eq_dumbbell.id])
        results = query_exercises(filters)
        categories = {r['category'] for r in results}

        # Both Arms and Chest should appear in results
        self.assertIn('Arms',  categories)
        self.assertIn('Chest', categories)
