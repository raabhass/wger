# -*- coding: utf-8 -*-

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

# Standard Library
import copy
import datetime
import json
import logging
from typing import List

# Django
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import (
    HttpResponseForbidden,
    HttpResponseRedirect,
    JsonResponse,
)
from django.shortcuts import (
    get_object_or_404,
    render,
)
from django.urls import reverse
from django.utils.translation import get_language
from django.views import View

# wger
from wger.manager.models import (
    AbstractChangeConfig,
    Routine,
    SlotEntry,
)
from wger.manager.services.workout_plan_generator import generate_workout_plan_options
from wger.manager.services.workout_plan_prompt_parser import parse_workout_plan_prompt
from wger.manager.services.workout_plan_saver import save_generated_workout_plan


logger = logging.getLogger(__name__)


class WorkoutPlanGeneratorView(LoginRequiredMixin, View):
    """
    GET  /routine/generate  renders the prompt page.
    POST /routine/generate  returns preview-only workout plan options.
    """

    template_name = 'routines/generate.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        try:
            body = json.loads(request.body)
            prompt = body.get('prompt', '').strip()
        except (json.JSONDecodeError, AttributeError):
            return JsonResponse({'error': 'Invalid request body.'}, status=400)

        if not prompt:
            return JsonResponse({'error': 'Prompt is required.'}, status=400)

        if len(prompt) > 500:
            return JsonResponse({'error': 'Prompt is too long (max 500 characters).'}, status=400)

        parsed = parse_workout_plan_prompt(prompt)
        options = generate_workout_plan_options(
            parsed,
            language=get_language() or 'en',
        )
        exercise_count = sum(
            len(day.get('exercises', []))
            for option in options
            for day in option.get('days', [])
        )
        if exercise_count < parsed.days and not parsed.used_llm:
            parsed = parse_workout_plan_prompt(prompt, force_llm=True)
            options = generate_workout_plan_options(
                parsed,
                language=get_language() or 'en',
            )

        return JsonResponse(
            {
                'options': options,
                'parsed': {
                    'days': parsed.days,
                    'goal': parsed.goal,
                    'equipment_ids': parsed.equipment_ids,
                    'muscle_ids': parsed.muscle_ids,
                    'category_ids': parsed.category_ids,
                    'level': parsed.level,
                    'notes': parsed.notes,
                    'confidence': parsed.confidence,
                    'missing_fields': parsed.missing_fields,
                    'used_llm': parsed.used_llm,
                    'llm_note': parsed.llm_note,
                },
                'explanation': parsed.explanation,
                'used_llm': parsed.used_llm,
                'llm_note': parsed.llm_note,
            }
        )


class WorkoutPlanSaveView(LoginRequiredMixin, View):
    """Save a selected generated workout plan as a regular routine."""

    def post(self, request):
        try:
            body = json.loads(request.body)
            plan = body.get('plan')
        except (json.JSONDecodeError, AttributeError):
            return JsonResponse({'error': 'Invalid request body.'}, status=400)

        if not isinstance(plan, dict):
            return JsonResponse({'error': 'A generated plan is required.'}, status=400)

        plan_hash = str(hash(json.dumps(plan, sort_keys=True)))
        saved_plans = request.session.get('saved_generated_workout_plans', {})
        existing_routine_id = saved_plans.get(plan_hash)
        if existing_routine_id:
            messages.info(request, 'This generated plan was already saved.')
            return JsonResponse(
                {
                    'redirect_url': reverse(
                        'manager:routine:view',
                        kwargs={'pk': existing_routine_id},
                    )
                }
            )

        saved_plan = save_generated_workout_plan(request.user, plan)
        if saved_plan.saved_exercises:
            messages.success(request, 'Generated workout plan saved.')
        else:
            messages.warning(request, 'Routine saved, but no valid exercises were found.')

        for warning in saved_plan.warnings:
            messages.warning(request, warning)

        saved_plans[plan_hash] = saved_plan.routine.id
        request.session['saved_generated_workout_plans'] = saved_plans
        request.session.modified = True

        return JsonResponse({'redirect_url': saved_plan.routine.get_absolute_url()})


@login_required
def copy_routine(request, pk):
    """
    Makes a copy of a routine
    """
    routine = get_object_or_404(Routine, pk=pk)

    if request.user != routine.user and not routine.is_public:
        # Check if the user is a trainer and the routine belongs to a client, only if it does not
        # belong to the user.
        trainer_identity_pk = request.session.get('trainer.identity', None)
        if not trainer_identity_pk or routine.user.pk != trainer_identity_pk:
            return HttpResponseForbidden()

    def copy_config(configs: List[AbstractChangeConfig], slot_entry: SlotEntry):
        for config in configs:
            config_copy = copy.copy(config)
            config_copy.pk = None
            config_copy.slot_entry = slot_entry
            config_copy.save()

    # Process request
    # Copy workout
    routine_copy: Routine = copy.copy(routine)
    routine_copy.pk = None
    routine_copy.created = None
    routine_copy.user = request.user
    routine_copy.is_template = False
    routine_copy.is_public = False

    # Update the start and end date
    routine_copy.start = datetime.date.today()
    routine_copy.end = routine_copy.start + routine.duration

    routine_copy.save()

    # Copy the days
    for day in routine.days.all():
        day_copy = copy.copy(day)
        day_copy.pk = None
        day_copy.routine = routine_copy
        day_copy.save()

        # Copy the slots
        for current_slot in day.slots.all():
            slot_copy = copy.copy(current_slot)
            slot_copy.pk = None
            slot_copy.day = day_copy
            slot_copy.save()

            # Copy the slot entries
            for current_entry in current_slot.entries.all():
                slot_entry_copy = copy.copy(current_entry)
                slot_entry_copy.pk = None
                slot_entry_copy.slot = slot_copy
                slot_entry_copy.save()

                copy_config(current_entry.weightconfig_set.all(), slot_entry_copy)
                copy_config(current_entry.maxweightconfig_set.all(), slot_entry_copy)

                copy_config(current_entry.repetitionsconfig_set.all(), slot_entry_copy)
                copy_config(current_entry.maxrepetitionsconfig_set.all(), slot_entry_copy)

                copy_config(current_entry.rirconfig_set.all(), slot_entry_copy)
                copy_config(current_entry.maxrirconfig_set.all(), slot_entry_copy)

                copy_config(current_entry.restconfig_set.all(), slot_entry_copy)
                copy_config(current_entry.maxrestconfig_set.all(), slot_entry_copy)

                copy_config(current_entry.setsconfig_set.all(), slot_entry_copy)
                copy_config(current_entry.maxsetsconfig_set.all(), slot_entry_copy)

    return HttpResponseRedirect(routine_copy.get_absolute_url())
