from collections import defaultdict
from decimal import Decimal

from wger.manager.models import WorkoutLog, WorkoutSession


def clean_exercise_name(exercise) -> str:
    """
    Return only the readable exercise name.
    Example:
    'base <uuid> (Crunches)' -> 'Crunches'
    """
    raw_name = str(exercise)

    if "(" in raw_name and ")" in raw_name:
        start = raw_name.rfind("(")
        end = raw_name.rfind(")")
        if start != -1 and end != -1 and start < end:
            return raw_name[start + 1:end].strip()

    return raw_name


class ProgressionRecommendationService:
    SMALL_WEIGHT_INCREASE = Decimal("2.5")
    LARGE_WEIGHT_INCREASE = Decimal("5.0")

    @classmethod
    def get_session_recommendations(cls, session: WorkoutSession) -> list[dict]:
        logs = (
            WorkoutLog.objects.filter(session=session)
            .select_related("exercise", "session")
            .order_by("exercise_id", "id")
        )

        grouped_logs = defaultdict(list)
        for log in logs:
            grouped_logs[log.exercise_id].append(log)

        recommendations = []
        for _, exercise_logs in grouped_logs.items():
            recommendations.append(
                cls._build_recommendation_for_exercise(session, exercise_logs)
            )

        return recommendations

    @classmethod
    def _build_recommendation_for_exercise(cls, session: WorkoutSession, logs: list[WorkoutLog]) -> dict:
        exercise = logs[0].exercise
        total_sets = len(logs)
        successful_sets = 0
        almost_successful_sets = 0
        latest_weight = None

        for log in logs:
            if log.weight is not None:
                latest_weight = log.weight

            if log.repetitions is not None and log.repetitions_target is not None:
                if log.repetitions >= log.repetitions_target:
                    successful_sets += 1
                elif log.repetitions == log.repetitions_target - 1:
                    almost_successful_sets += 1

        duration_minutes = cls._get_duration_minutes(session)

        # Case 3:
        # If the workout impression was bad, recommend or slightly reducing weight.
        if str(session.impression) == "1":
            return {
                "exercise_id": exercise.id,
                "exercise_name": clean_exercise_name(exercise),
                "type": "hold_or_reduce",
                "message": "If you felt that the excercise was challenging/difficult: Keep the same weight or reduce slightly next session.",
                "duration_minutes": duration_minutes,
            }
        # Case 1:
        # If all sets were successful, the workout felt GOOD,
        # and a weight value exists, recommend increasing the weight.
        if (
            total_sets > 0
            and successful_sets == total_sets
            and str(session.impression) == "3"
            and latest_weight is not None
        ):
            increase = cls.LARGE_WEIGHT_INCREASE if latest_weight >= Decimal("20") else cls.SMALL_WEIGHT_INCREASE
            next_weight = latest_weight + increase

            return {
                "exercise_id": exercise.id,
                "exercise_name": clean_exercise_name(exercise),
                "type": "increase_weight",
                "message": f"You hit all target reps. Increase to {next_weight} next session.",
                "duration_minutes": duration_minutes,
            }

        # Case 2:
        # If every set was either successful or only one rep short,
        # recommend keeping the same weight and aiming for one more rep.
        if total_sets > 0 and successful_sets + almost_successful_sets == total_sets:
            increase = cls.LARGE_WEIGHT_INCREASE if latest_weight >= Decimal("20") else cls.SMALL_WEIGHT_INCREASE
            next_weight = latest_weight + increase
            return {
                "exercise_id": exercise.id,
                "exercise_name": clean_exercise_name(exercise),
                "type": "add_rep",
                "message": f"Keep the same weight and aim for 1 more rep next session. You can also try increasing the weight to {next_weight} next session.",
                "duration_minutes": duration_minutes,
            }

        # Default case:
        # If performance was not strong enough to progress, and the workout
        # was not explicitly bad, recommend maintaining the current load.
        return {
            "exercise_id": exercise.id,
            "exercise_name": clean_exercise_name(exercise),
            "type": "hold",
            "message": "Maintain current load and focus on consistent form next session.",
            "duration_minutes": duration_minutes,
        }

    @staticmethod
    def _get_duration_minutes(session: WorkoutSession):
        if not session.time_start or not session.time_end:
            return None

        start_minutes = session.time_start.hour * 60 + session.time_start.minute
        end_minutes = session.time_end.hour * 60 + session.time_end.minute
        return end_minutes - start_minutes