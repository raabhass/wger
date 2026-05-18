from collections import defaultdict
from decimal import Decimal

from wger.manager.models import WorkoutLog, WorkoutSession


def clean_exercise_name(exercise) -> str:
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
    DEFAULT_TARGET_REPS = 12

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
    def _build_recommendation_for_exercise(
        cls, session: WorkoutSession, logs: list[WorkoutLog]
    ) -> dict:
        exercise = logs[0].exercise
        total_sets = len(logs)

        successful_sets = 0
        almost_successful_sets = 0
        failed_sets = 0
        latest_weight = None

        for log in logs:
            if log.weight is not None:
                latest_weight = log.weight

            target_reps = log.repetitions_target or cls.DEFAULT_TARGET_REPS

            if log.repetitions is not None:
                if log.repetitions >= target_reps:
                    successful_sets += 1
                elif log.repetitions == target_reps - 1:
                    almost_successful_sets += 1
                elif Decimal(log.repetitions) < Decimal(target_reps) * Decimal("0.5"):
                    failed_sets += 1

        duration_minutes = cls._get_duration_minutes(session)

        # Case 1: Bad impression overrides everything
        if str(session.impression) == "1":
            return {
                "exercise_id": exercise.id,
                "exercise_name": clean_exercise_name(exercise),
                "type": "hold_or_reduce",
                "message": "Your workout impression was low. Keep the same weight or reduce slightly next session to focus on recovery and form.",
                "duration_minutes": duration_minutes,
            }

        # Case 2: Very low completed reps also counts as a bad workout
        if failed_sets > 0:
            return {
                "exercise_id": exercise.id,
                "exercise_name": clean_exercise_name(exercise),
                "type": "hold_or_reduce",
                "message": "You were far below the target reps. Keep the same weight or reduce slightly next session to focus on recovery and form.",
                "duration_minutes": duration_minutes,
            }

        # Case 3: All sets successful + good impression = increase weight
        if (
            total_sets > 0
            and successful_sets == total_sets
            and str(session.impression) == "3"
            and latest_weight is not None
        ):
            increase = (
                cls.LARGE_WEIGHT_INCREASE
                if latest_weight >= Decimal("20")
                else cls.SMALL_WEIGHT_INCREASE
            )
            next_weight = latest_weight + increase

            return {
                "exercise_id": exercise.id,
                "exercise_name": clean_exercise_name(exercise),
                "type": "increase_weight",
                "message": f"You hit all target reps. Increase to {next_weight} next session.",
                "duration_minutes": duration_minutes,
            }

        # Case 4: Every set was successful or one rep short = add 1 rep
        if total_sets > 0 and successful_sets + almost_successful_sets == total_sets:
            return {
                "exercise_id": exercise.id,
                "exercise_name": clean_exercise_name(exercise),
                "type": "add_rep",
                "message": "Keep the same weight and aim for 1 more rep next session.",
                "duration_minutes": duration_minutes,
            }

        # Default case
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