# This file is part of wger Workout Manager.
#
# wger Workout Manager is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for preview-only workout plan generation."""

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
from wger.manager.services.workout_plan_generator import (
    EXERCISES_PER_DAY,
    generate_workout_plan_options,
)
from wger.manager.services.workout_plan_prompt_parser import ParsedWorkoutPlanPrompt


_DESCRIPTION = 'A' * 40


class WorkoutPlanGeneratorTestCase(TestCase):
    fixtures = ['licenses', 'languages']

    def setUp(self):
        self.lang = Language.objects.get(short_name='en')
        self.license = License.objects.first()

        self.cat_arms = ExerciseCategory.objects.create(id=8, name='Arms')
        self.cat_legs = ExerciseCategory.objects.create(id=9, name='Legs')
        self.cat_abs = ExerciseCategory.objects.create(id=10, name='Abs')
        self.cat_chest = ExerciseCategory.objects.create(id=11, name='Chest')
        self.cat_back = ExerciseCategory.objects.create(id=12, name='Back')
        self.cat_shoulders = ExerciseCategory.objects.create(id=13, name='Shoulders')
        self.cat_cardio = ExerciseCategory.objects.create(id=15, name='Cardio')

        self.eq_dumbbell = Equipment.objects.create(id=3, name='Dumbbell')
        self.eq_bodyweight = Equipment.objects.create(id=7, name='Bodyweight')

        self.mu_chest = Muscle.objects.create(
            id=4,
            name='Pectoralis major',
            name_en='Chest',
            is_front=True,
        )
        self.mu_abs = Muscle.objects.create(
            id=6,
            name='Rectus abdominis',
            name_en='Abs',
            is_front=True,
        )
        self.mu_quads = Muscle.objects.create(
            id=10,
            name='Quadriceps',
            name_en='Quads',
            is_front=True,
        )
        self.mu_lats = Muscle.objects.create(
            id=12,
            name='Latissimus dorsi',
            name_en='Lats',
            is_front=False,
        )

        self._make('Dumbbell Press', self.cat_chest, [self.eq_dumbbell], [self.mu_chest])
        self._make('Push Up', self.cat_chest, [self.eq_bodyweight], [self.mu_chest])
        self._make('Bodyweight Squat', self.cat_legs, [self.eq_bodyweight], [self.mu_quads])
        self._make('Dumbbell Lunge', self.cat_legs, [self.eq_dumbbell], [self.mu_quads])
        self._make('Crunch', self.cat_abs, [self.eq_bodyweight], [self.mu_abs])
        self._make('Dumbbell Row', self.cat_back, [self.eq_dumbbell], [self.mu_lats])
        self._make('Mountain Climber', self.cat_cardio, [self.eq_bodyweight], [self.mu_abs])
        self._make('Shoulder Press', self.cat_shoulders, [self.eq_dumbbell], [self.mu_chest])

    def _make(self, name, category, equipment, muscles):
        exercise = Exercise.objects.create(category=category, license=self.license)
        exercise.equipment.set(equipment)
        exercise.muscles.set(muscles)
        Translation.objects.create(
            exercise=exercise,
            language=self.lang,
            license=self.license,
            name=name,
            description=_DESCRIPTION,
        )
        return exercise


class WorkoutPlanGeneratorTests(WorkoutPlanGeneratorTestCase):
    def test_generates_three_options_for_three_day_plan(self):
        parsed = ParsedWorkoutPlanPrompt(days=3, goal='fat_loss', equipment_ids=[7])

        options = generate_workout_plan_options(parsed, language='en')

        self.assertEqual(len(options), 3)
        self.assertEqual(len(options[0]['days']), 3)

    def test_generated_days_have_exercise_preview_data(self):
        parsed = ParsedWorkoutPlanPrompt(
            days=2,
            goal='strength',
            equipment_ids=[3],
            category_ids=[11, 9],
            level='beginner',
        )

        options = generate_workout_plan_options(parsed, language='en')
        first_day = options[0]['days'][0]

        self.assertTrue(first_day['exercises'])
        exercise = first_day['exercises'][0]
        for key in ('id', 'name', 'url', 'category', 'equipment', 'sets', 'reps', 'rest'):
            self.assertIn(key, exercise)
        self.assertEqual(exercise['sets'], '2')

    def test_equipment_filter_prefers_requested_equipment(self):
        parsed = ParsedWorkoutPlanPrompt(days=1, equipment_ids=[7], category_ids=[11])

        options = generate_workout_plan_options(parsed, language='en')
        exercises = options[0]['days'][0]['exercises']

        self.assertTrue(exercises)
        self.assertIn('Bodyweight', exercises[0]['equipment'])

    def test_limits_exercises_per_day(self):
        parsed = ParsedWorkoutPlanPrompt(days=1, equipment_ids=[3])

        options = generate_workout_plan_options(parsed, language='en')

        self.assertLessEqual(len(options[0]['days'][0]['exercises']), EXERCISES_PER_DAY)

    def test_does_not_save_routine_models(self):
        parsed = ParsedWorkoutPlanPrompt(days=3, equipment_ids=[3])

        generate_workout_plan_options(parsed, language='en')

        from wger.manager.models import Routine

        self.assertEqual(Routine.objects.count(), 0)
