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
Tests for wger.exercises.services.prompt_parser.

All tests run against the rule-based path only — no network calls,
no API keys required.  Gemini fallback is tested via a mock.
"""

# Standard Library
from unittest.mock import MagicMock, patch

# Django
from django.test import SimpleTestCase

# wger
from wger.exercises.services.prompt_parser import (
    ParsedFilters,
    _rule_based_parse,
    parse_prompt,
)


class EquipmentExtractionTests(SimpleTestCase):
    """Rule-based equipment detection."""

    def test_dumbbell_singular(self):
        result = _rule_based_parse('I want a dumbbell workout')
        self.assertIn(3, result.equipment_ids)

    def test_dumbbell_plural(self):
        result = _rule_based_parse('I only have dumbbells at home')
        self.assertIn(3, result.equipment_ids)

    def test_barbell(self):
        result = _rule_based_parse('barbell squats please')
        self.assertIn(1, result.equipment_ids)

    def test_kettlebell(self):
        result = _rule_based_parse('kettlebell swings for cardio')
        self.assertIn(10, result.equipment_ids)

    def test_bodyweight_variants(self):
        for phrase in ['bodyweight', 'no equipment', 'at home', 'calisthenics']:
            with self.subTest(phrase=phrase):
                result = _rule_based_parse(phrase)
                self.assertIn(7, result.equipment_ids, msg=f'Failed for: {phrase}')

    def test_resistance_band(self):
        result = _rule_based_parse('I have a resistance band')
        self.assertIn(11, result.equipment_ids)

    def test_pullup_bar_variants(self):
        for phrase in ['pull-up bar', 'pullup bar', 'chin-up bar']:
            with self.subTest(phrase=phrase):
                result = _rule_based_parse(phrase)
                self.assertIn(6, result.equipment_ids)

    def test_multiple_equipment(self):
        result = _rule_based_parse('I have a barbell and a bench')
        self.assertIn(1, result.equipment_ids)
        self.assertIn(8, result.equipment_ids)


class MuscleExtractionTests(SimpleTestCase):
    """Rule-based muscle detection."""

    def test_biceps(self):
        result = _rule_based_parse('want to train my biceps')
        self.assertIn(1, result.muscle_ids)

    def test_bicep_singular(self):
        result = _rule_based_parse('bicep curl')
        self.assertIn(1, result.muscle_ids)

    def test_triceps(self):
        result = _rule_based_parse('tricep dips')
        self.assertIn(5, result.muscle_ids)

    def test_chest(self):
        result = _rule_based_parse('chest exercises please')
        self.assertIn(4, result.muscle_ids)

    def test_pecs_synonym(self):
        result = _rule_based_parse('want bigger pecs')
        self.assertIn(4, result.muscle_ids)

    def test_lats(self):
        result = _rule_based_parse('work on my lats')
        self.assertIn(12, result.muscle_ids)

    def test_latissimus(self):
        result = _rule_based_parse('latissimus dorsi exercises')
        self.assertIn(12, result.muscle_ids)

    def test_glutes(self):
        result = _rule_based_parse('glute exercises')
        self.assertIn(8, result.muscle_ids)

    def test_hamstrings(self):
        result = _rule_based_parse('hamstring stretches')
        self.assertIn(11, result.muscle_ids)

    def test_quads(self):
        result = _rule_based_parse('quad strengthening')
        self.assertIn(10, result.muscle_ids)

    def test_lower_back(self):
        result = _rule_based_parse('lower back pain relief exercises')
        self.assertIn(16, result.muscle_ids)

    def test_obliques(self):
        result = _rule_based_parse('oblique exercises')
        self.assertIn(14, result.muscle_ids)

    def test_traps(self):
        result = _rule_based_parse('trap exercises')
        self.assertIn(9, result.muscle_ids)

    def test_calves(self):
        result = _rule_based_parse('calf raises')
        self.assertIn(7, result.muscle_ids)

    def test_abs_variants(self):
        for phrase in ['abs', 'abdominals', 'six pack']:
            with self.subTest(phrase=phrase):
                result = _rule_based_parse(f'{phrase} workout')
                self.assertIn(6, result.muscle_ids)


class CategoryExtractionTests(SimpleTestCase):
    """Rule-based category detection."""

    def test_arms_category(self):
        result = _rule_based_parse('arm exercises')
        self.assertIn(8, result.category_ids)

    def test_legs_category(self):
        result = _rule_based_parse('leg day workout')
        self.assertIn(9, result.category_ids)

    def test_shoulders_category(self):
        result = _rule_based_parse('shoulder exercises')
        self.assertIn(13, result.category_ids)

    def test_back_category(self):
        result = _rule_based_parse('back workout')
        self.assertIn(12, result.category_ids)

    def test_cardio_category(self):
        result = _rule_based_parse('cardio workout')
        self.assertIn(15, result.category_ids)

    def test_delts_synonym(self):
        result = _rule_based_parse('deltoid exercises')
        self.assertIn(13, result.category_ids)


class ArmsExpansionTests(SimpleTestCase):
    """When 'arms' category is matched, arm muscles should also be included."""

    def test_arms_expands_to_biceps_triceps_brachialis(self):
        result = _rule_based_parse('arm exercises')
        self.assertIn(8, result.category_ids)
        self.assertIn(1, result.muscle_ids)   # biceps
        self.assertIn(5, result.muscle_ids)   # triceps
        self.assertIn(13, result.muscle_ids)  # brachialis

    def test_no_duplicate_muscles_when_arms_and_biceps_both_mentioned(self):
        result = _rule_based_parse('arm and bicep workout')
        count = result.muscle_ids.count(1)
        self.assertEqual(count, 1, 'Biceps ID should appear only once')


class SoftHintTests(SimpleTestCase):
    """Soft hints should be captured in notes, not as filter IDs."""

    def test_light_hint(self):
        result = _rule_based_parse('light dumbbell bicep workout')
        self.assertIn('light', result.notes)
        # 'light' must not pollute filter IDs
        self.assertNotIn('light', result.equipment_ids)
        self.assertNotIn('light', result.muscle_ids)

    def test_easy_maps_to_light(self):
        result = _rule_based_parse('easy exercises for beginners')
        self.assertIn('light', result.notes)

    def test_heavy_hint(self):
        result = _rule_based_parse('heavy barbell squats')
        self.assertIn('heavy', result.notes)

    def test_compound_hint(self):
        result = _rule_based_parse('compound chest exercises')
        self.assertIn('compound', result.notes)

    def test_no_notes_on_plain_prompt(self):
        result = _rule_based_parse('dumbbell bicep exercises')
        self.assertEqual(result.notes, [])


class EdgeCaseTests(SimpleTestCase):
    """Boundary conditions and robustness."""

    def test_empty_string(self):
        result = parse_prompt('')
        self.assertTrue(result.is_empty())

    def test_whitespace_only(self):
        result = parse_prompt('   ')
        self.assertTrue(result.is_empty())

    def test_unknown_prompt_returns_empty(self):
        result = _rule_based_parse('xyzzy foobar baz')
        self.assertTrue(result.is_empty())

    def test_case_insensitive(self):
        result = _rule_based_parse('DUMBBELL BICEPS WORKOUT')
        self.assertIn(3, result.equipment_ids)
        self.assertIn(1, result.muscle_ids)

    def test_partial_word_not_matched(self):
        # "bar" inside "barbell" should not match Pull-up bar separately;
        # and "lat" inside "flat" should not match Lats.
        result = _rule_based_parse('flat bench press')
        self.assertNotIn(12, result.muscle_ids)  # 'lat' inside 'flat' must not match

    def test_explanation_populated(self):
        result = _rule_based_parse('dumbbell bicep curls')
        self.assertNotEqual(result.explanation, '')
        self.assertNotEqual(result.explanation, 'No filters matched.')

    def test_no_llm_flag_on_rule_based(self):
        result = _rule_based_parse('dumbbell bicep workout')
        self.assertFalse(result.used_llm)

    def test_is_empty_false_when_filters_present(self):
        result = _rule_based_parse('dumbbell bicep workout')
        self.assertFalse(result.is_empty())


class GeminiFallbackTests(SimpleTestCase):
    """Gemini is only called when rule-based returns nothing."""

    def test_gemini_not_called_when_rules_match(self):
        with patch('wger.exercises.services.prompt_parser._gemini_parse') as mock_gemini:
            parse_prompt('dumbbell bicep workout')
            mock_gemini.assert_not_called()

    def test_gemini_called_when_rules_return_nothing(self):
        with patch('wger.exercises.services.prompt_parser._gemini_parse') as mock_gemini:
            mock_gemini.return_value = ParsedFilters(explanation='Gemini result')
            parse_prompt('xyzzy foobar baz')
            mock_gemini.assert_called_once_with('xyzzy foobar baz')

    def test_gemini_result_used_when_rules_empty(self):
        fake_result = ParsedFilters(
            equipment_ids=[3],
            muscle_ids=[1],
            explanation='Gemini extracted dumbbell + biceps.',
            used_llm=True,
        )
        with patch('wger.exercises.services.prompt_parser._gemini_parse', return_value=fake_result):
            result = parse_prompt('xyzzy foobar baz')
        self.assertIn(3, result.equipment_ids)
        self.assertTrue(result.used_llm)

    def test_gemini_skipped_when_no_api_key(self):
        """When GEMINI_API_KEY is blank, _gemini_parse returns empty filters gracefully."""
        with self.settings(GEMINI_API_KEY=''):
            # Patch google.generativeai to simulate it being importable but key is blank
            mock_genai = MagicMock()
            with patch.dict('sys.modules', {'google.generativeai': mock_genai}):
                result = parse_prompt('xyzzy foobar baz')
        # Should not raise, should return empty ParsedFilters
        self.assertIsInstance(result, ParsedFilters)


class RealisticPromptTests(SimpleTestCase):
    """End-to-end realistic prompts — rule-based path only."""

    def test_dumbbell_biceps_light(self):
        result = parse_prompt('I only have dumbbells and want light biceps exercises')
        self.assertIn(3, result.equipment_ids)   # Dumbbell
        self.assertIn(1, result.muscle_ids)       # Biceps
        self.assertIn('light', result.notes)
        self.assertFalse(result.used_llm)

    def test_bodyweight_chest(self):
        result = parse_prompt('bodyweight chest exercises at home')
        self.assertIn(7, result.equipment_ids)   # Bodyweight
        self.assertIn(4, result.muscle_ids)       # Chest

    def test_kettlebell_glutes_legs(self):
        result = parse_prompt('kettlebell exercises for glutes and legs')
        self.assertIn(10, result.equipment_ids)  # Kettlebell
        self.assertIn(8, result.muscle_ids)       # Glutes
        self.assertIn(9, result.category_ids)     # Legs

    def test_pullup_bar_back_lats(self):
        result = parse_prompt('pull-up bar exercises for back and lats')
        self.assertIn(6, result.equipment_ids)   # Pull-up bar
        self.assertIn(12, result.muscle_ids)      # Lats
        self.assertIn(12, result.category_ids)    # Back category

    def test_barbell_heavy_compound(self):
        result = parse_prompt('heavy barbell compound leg exercises')
        self.assertIn(1, result.equipment_ids)   # Barbell
        self.assertIn(9, result.category_ids)     # Legs
        self.assertIn('heavy', result.notes)
        self.assertIn('compound', result.notes)
