# This file is part of wger Workout Manager.
#
# wger Workout Manager is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for optional Gemini-assisted workout plan prompt parsing."""

# Standard Library
import json
from unittest.mock import patch

# Django
from django.urls import reverse

# wger
from wger.core.tests.base_testcase import WgerTestCase
from wger.exercises.models import Exercise
from wger.manager.models import (
    Routine,
    SlotEntry,
)
from wger.manager.services.workout_plan_prompt_parser import parse_workout_plan_prompt


class WorkoutPlanLlmParserTests(WgerTestCase):
    def test_rule_based_generation_still_works_when_llm_disabled(self):
        with self.settings(ENABLE_WORKOUT_PLAN_LLM=False, GEMINI_API_KEY=''):
            result = parse_workout_plan_prompt(
                'Create a 4-day beginner strength plan with dumbbells for chest'
            )

        self.assertFalse(result.used_llm)
        self.assertEqual(result.days, 4)
        self.assertEqual(result.goal, 'strength')
        self.assertIn(3, result.equipment_ids)

    def test_missing_gemini_api_key_does_not_break_generation(self):
        with self.settings(ENABLE_WORKOUT_PLAN_LLM=True, GEMINI_API_KEY=''):
            result = parse_workout_plan_prompt('lean body home friendly plan')

        self.assertFalse(result.used_llm)
        self.assertEqual(result.days, 3)

    def test_gemini_fallback_is_called_only_for_ambiguous_prompts(self):
        with patch(
            'wger.manager.services.workout_plan_llm._call_gemini',
            return_value=None,
        ) as mock_call:
            parse_workout_plan_prompt(
                'Create a 4-day beginner strength dumbbell plan for chest and legs'
            )
            mock_call.assert_not_called()

            parse_workout_plan_prompt('lean body not too hard home friendly plan')
            mock_call.assert_called_once()

    def test_invalid_gemini_json_falls_back_safely(self):
        with patch(
            'wger.manager.services.workout_plan_llm._call_gemini',
            return_value=None,
        ):
            result = parse_workout_plan_prompt('lean body not too hard')

        self.assertFalse(result.used_llm)
        self.assertEqual(result.goal, 'general')

    def test_valid_gemini_json_improves_parsed_fields(self):
        gemini_data = {
            'days_per_week': 4,
            'goal': 'hypertrophy',
            'intensity': 'beginner',
            'equipment': ['dumbbell', 'bodyweight'],
            'target_muscles': ['chest', 'legs'],
            'workout_style': 'upper_lower',
            'notes': 'Interpreted as a beginner upper/lower muscle-building plan.',
        }

        with patch(
            'wger.manager.services.workout_plan_llm._call_gemini',
            return_value=gemini_data,
        ):
            result = parse_workout_plan_prompt('lean body home friendly plan')

        self.assertTrue(result.used_llm)
        self.assertEqual(result.days, 4)
        self.assertEqual(result.goal, 'muscle_gain')
        self.assertEqual(result.level, 'beginner')
        self.assertIn(3, result.equipment_ids)
        self.assertIn(7, result.equipment_ids)
        self.assertIn(4, result.muscle_ids)
        self.assertIn(9, result.category_ids)
        self.assertIn('upper/lower', result.llm_note)


class WorkoutPlanLlmSaveFlowTests(WgerTestCase):
    def test_saving_generated_plan_still_works_after_llm_assisted_parsing(self):
        self.user_login('admin')
        count_before = Routine.objects.count()
        gemini_data = {
            'days_per_week': 1,
            'goal': 'strength',
            'intensity': 'beginner',
            'equipment': ['bodyweight'],
            'target_muscles': ['chest'],
            'workout_style': 'full_body',
            'notes': 'Interpreted as a short beginner plan.',
        }

        with patch(
            'wger.manager.services.workout_plan_llm._call_gemini',
            return_value=gemini_data,
        ):
            response = self.client.post(
                reverse('manager:routine:generate'),
                data=json.dumps({'prompt': 'not too hard home friendly plan'}),
                content_type='application/json',
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['used_llm'])
        plan = response.json()['options'][0]
        plan['days'][0]['exercises'] = [
            {
                'id': Exercise.objects.first().id,
                'sets': '2',
                'reps': '8-12',
                'rest': '60 sec',
            }
        ]

        save_response = self.client.post(
            reverse('manager:routine:generate-save'),
            data=json.dumps({'plan': plan}),
            content_type='application/json',
        )

        self.assertEqual(save_response.status_code, 200)
        self.assertEqual(Routine.objects.count(), count_before + 1)
        routine = Routine.objects.latest('id')
        self.assertEqual(SlotEntry.objects.filter(slot__day__routine=routine).count(), 1)
