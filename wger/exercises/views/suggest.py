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

# Standard Library
import json
import logging

# Django
from django.http import JsonResponse
from django.utils.translation import get_language
from django.views import View
from django.shortcuts import render

# wger
from wger.exercises.services.prompt_parser import parse_prompt
from wger.exercises.services.suggestion_query import query_exercises

logger = logging.getLogger(__name__)


class ExerciseSuggestView(View):
    """
    GET  /exercise/suggest/  — renders the suggestion page
    POST /exercise/suggest/  — AJAX endpoint, returns JSON exercise list
    """

    template_name = 'exercise/suggest.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        # Parse JSON body
        try:
            body = json.loads(request.body)
            prompt = body.get('prompt', '').strip()
        except (json.JSONDecodeError, AttributeError):
            return JsonResponse({'error': 'Invalid request body.'}, status=400)

        if not prompt:
            return JsonResponse({'error': 'Prompt is required.'}, status=400)

        if len(prompt) > 300:
            return JsonResponse({'error': 'Prompt is too long (max 300 characters).'}, status=400)

        # Parse prompt into structured filters
        filters = parse_prompt(prompt)

        # Query and rank exercises
        language = get_language() or 'en'
        results = query_exercises(filters, language=language)

        return JsonResponse({
            'results':      results,
            'filters_used': {
                'equipment_ids':         filters.equipment_ids,
                'muscle_ids':            filters.muscle_ids,
                'category_ids':          filters.category_ids,
                'exclude_muscle_ids':    filters.exclude_muscle_ids,
                'exclude_equipment_ids': filters.exclude_equipment_ids,
                'name_search':           filters.name_search,
            },
            'notes':       filters.notes,
            'explanation': filters.explanation,
            'used_llm':    filters.used_llm,
        })
