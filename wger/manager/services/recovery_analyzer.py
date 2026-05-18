from datetime import timedelta
from django.utils import timezone
from wger.manager.models import WorkoutSession

def analyze_recovery(log):

    #returns recommendation and possible_actions
    recommendations = []
    possible_actions = []

    yesterday = log.date - timedelta(days=1)
    recovery_flag = False

    # Checks for heavy training yesterday and if the general impression is 3 or below. Needs future improvement based on if weights are completed.
    heavy_training = WorkoutSession.objects.filter(
        user=log.user,
        date= yesterday,
        impression__lte=3,
    ).exists()

    # Sleep warning
    if log.sleep_hours < 6:
        recommendations.append(
            "🚩Low sleep detected. "
        )
        possible_actions.append("Prioritize recovery or lighter training.")

    # Heavy Training detected
    if heavy_training:
        recommendations.append(
            "🚩Heavy workout detected."
        )
        possible_actions.append("Avoid excessive/intense training.")

    # Hydration warning
    if log.water_intake < 2000:
        recommendations.append(
            "🚩Hydration intake is below recommended levels."
        )
        possible_actions.append("Increase water intake throughout the day.")

    # Energy warning
    if log.energy_level <= 2:
        recommendations.append(
            "🚩Low energy reported. Recovery risk may be elevated."
        )
        possible_actions.append("Consider rest or lighter excercise.")

    #Displays weekly hydration on yesterdays log to provide recommendation
    if log.date == (timezone.localdate() - timedelta(days=1)):
        weekly_hydration = analyze_weekly_hydration(log)
        if weekly_hydration:
            recommendations.append(weekly_hydration)

    if possible_actions:
        recommendations.append("Recommendation(s): " + " ".join(possible_actions))

    # Default healthy response
    if not recommendations:
        recommendations.append(
            f"Recovery status looks good."
        )

    return recommendations


def analyze_weekly_hydration(log):
    start_date = log.date - timedelta(days=6)

    weekly_logs = log.user.recoveryhydrationlog_set.filter(
        date__gte=start_date,
        date__lte=log.date,
    )

    total_water = sum(entry.water_intake for entry in weekly_logs)
    log_count = weekly_logs.count()

    if log_count == 0:
        return None

    average_water = total_water / log_count
    daily_target = 2000

    if average_water < daily_target:
        return (
            f"💧Weekly hydration is low. Your average is "
            f"{average_water:.0f} ml/day, below the {daily_target} ml target."
        )

    return (
        f"💧Weekly hydration looks good. Your average is "
        f"{average_water:.0f} ml/day."
    )