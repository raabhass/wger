# This file is part of wger Workout Manager.
#
# wger Workout Manager is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the rule-based workout plan prompt parser."""

# Django
from django.test import SimpleTestCase

# wger
from wger.manager.services.workout_plan_prompt_parser import parse_workout_plan_prompt


class WorkoutPlanPromptParserTests(SimpleTestCase):
    def test_extracts_days_from_numeric_prompt(self):
        result = parse_workout_plan_prompt('Create a 4-day beginner strength plan')
        self.assertEqual(result.days, 4)

    def test_extracts_days_from_word_prompt(self):
        result = parse_workout_plan_prompt('I need a three day plan for fat loss')
        self.assertEqual(result.days, 3)

    def test_defaults_to_three_days(self):
        result = parse_workout_plan_prompt('Build me a dumbbell workout')
        self.assertEqual(result.days, 3)

    def test_extracts_goal_level_equipment_and_targets(self):
        result = parse_workout_plan_prompt(
            'I want a 4-day beginner strength plan using dumbbells and bodyweight, '
            'focused on chest and legs'
        )

        self.assertEqual(result.goal, 'strength')
        self.assertEqual(result.level, 'beginner')
        self.assertIn(3, result.equipment_ids)
        self.assertIn(7, result.equipment_ids)
        self.assertIn(4, result.muscle_ids)
        self.assertIn(9, result.category_ids)

    def test_home_no_equipment_maps_to_bodyweight(self):
        result = parse_workout_plan_prompt(
            'Create a 3-day home workout plan for fat loss with no gym equipment'
        )

        self.assertEqual(result.goal, 'fat_loss')
        self.assertIn(7, result.equipment_ids)

    def test_full_body_expands_categories(self):
        result = parse_workout_plan_prompt('full body beginner workout')

        for category_id in (8, 9, 10, 11, 12, 13):
            self.assertIn(category_id, result.category_ids)
        self.assertIn('full_body', result.notes)

    def test_empty_prompt_returns_explanation(self):
        result = parse_workout_plan_prompt('')

        self.assertEqual(result.explanation, 'Empty prompt.')
