# This file is part of wger Workout Manager.
#
# wger Workout Manager is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for saving generated workout plan previews."""

# Standard Library
import json

# Django
from django.urls import reverse

# wger
from wger.core.tests.base_testcase import WgerTestCase
from wger.exercises.models import Exercise
from wger.manager.models import (
    Day,
    RepetitionsConfig,
    RestConfig,
    Routine,
    SetsConfig,
    SlotEntry,
)


class WorkoutPlanSaveViewTests(WgerTestCase):
    def _plan_payload(self, exercise_id: int) -> dict:
        return {
            'title': 'Generated Strength Plan',
            'goal': 'strength',
            'level': 'beginner',
            'days_count': 1,
            'summary': 'Beginner strength preview with 1 training day.',
            'days': [
                {
                    'day': 1,
                    'name': 'Day 1: Chest',
                    'focus': 'Chest',
                    'exercises': [
                        {
                            'id': exercise_id,
                            'name': 'Saved Exercise',
                            'url': '/exercise/1/view',
                            'category': 'Chest',
                            'equipment': ['Bodyweight'],
                            'muscles': ['Chest'],
                            'sets': '3',
                            'reps': '8-12',
                            'rest': '60 sec',
                        }
                    ],
                }
            ],
        }

    def _post_plan(self, plan: dict):
        return self.client.post(
            reverse('manager:routine:generate-save'),
            data=json.dumps({'plan': plan}),
            content_type='application/json',
        )

    def test_generated_plan_can_be_saved_for_authenticated_user(self):
        self.user_login('admin')
        exercise = Exercise.objects.first()
        count_before = Routine.objects.count()

        response = self._post_plan(self._plan_payload(exercise.id))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(Routine.objects.count(), count_before + 1)

        routine = Routine.objects.latest('id')
        self.assertEqual(routine.user.username, 'admin')
        self.assertEqual(routine.days.count(), 1)
        self.assertEqual(Day.objects.filter(routine=routine).first().slots.count(), 1)
        self.assertEqual(SlotEntry.objects.filter(slot__day__routine=routine).count(), 1)
        self.assertEqual(
            SetsConfig.objects.filter(slot_entry__slot__day__routine=routine).count(),
            1,
        )
        self.assertEqual(
            RepetitionsConfig.objects.filter(slot_entry__slot__day__routine=routine).count(),
            1,
        )
        self.assertEqual(
            RestConfig.objects.filter(slot_entry__slot__day__routine=routine).count(),
            1,
        )
        self.assertEqual(response.json()['redirect_url'], routine.get_absolute_url())

    def test_unauthenticated_user_cannot_save(self):
        exercise = Exercise.objects.first()
        count_before = Routine.objects.count()

        response = self._post_plan(self._plan_payload(exercise.id))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(Routine.objects.count(), count_before)

    def test_invalid_generated_exercise_ids_are_skipped(self):
        self.user_login('admin')
        count_before = Routine.objects.count()

        response = self._post_plan(self._plan_payload(999999))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(Routine.objects.count(), count_before + 1)

        routine = Routine.objects.latest('id')
        self.assertEqual(SlotEntry.objects.filter(slot__day__routine=routine).count(), 0)
        self.assertEqual(response.json()['redirect_url'], routine.get_absolute_url())

    def test_existing_preview_generation_still_works(self):
        self.user_login('admin')

        response = self.client.post(
            reverse('manager:routine:generate'),
            data=json.dumps({'prompt': 'Create a 3-day beginner strength plan with dumbbells'}),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn('options', response.json())
        self.assertTrue(response.json()['options'])

    def test_routine_overview_links_to_generator(self):
        self.user_login('admin')

        response = self.client.get(reverse('manager:routine:overview'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse('manager:routine:generate'))
        self.assertContains(response, 'Workout plan generator')
