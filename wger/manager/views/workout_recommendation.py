from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_GET

from wger.manager.models import Routine, WorkoutSession
from wger.manager.services.progression import ProgressionRecommendationService


@login_required
@require_GET
def routine_recommendations_json(request, routine_pk):
    routine = get_object_or_404(
        Routine,
        pk=routine_pk,
        user=request.user,
    )

    latest_session = (
        WorkoutSession.objects.filter(routine=routine)
        .order_by('-date', '-id')
        .first()
    )

    if not latest_session:
        return JsonResponse({
            'session_id': None,
            'recommendations': [],
            'message': 'No completed workout session found for this routine yet.'
        })

    recommendations = ProgressionRecommendationService.get_session_recommendations(latest_session)

    return JsonResponse({
        'session_id': latest_session.id,
        'recommendations': recommendations,
        'message': 'ok',
    })