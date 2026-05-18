from django.shortcuts import redirect, get_object_or_404

from wger.core.views.react import ReactView
from wger.manager.forms.recovery import RecoveryHydrationLogForm
from wger.manager.models.recovery import RecoveryHydrationLog
from wger.manager.services.recovery_analyzer import analyze_recovery


class RecoveryHydrationView(ReactView):
    template_name = "recovery/recovery_hydration.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        edit_log = None

        if "pk" in self.kwargs:
            edit_log = get_object_or_404(
                RecoveryHydrationLog,
                pk=self.kwargs["pk"],
                user=self.request.user,
            )
        context["form"] = RecoveryHydrationLogForm(instance=edit_log)

        logs = RecoveryHydrationLog.objects.filter(
            user=self.request.user
        )

        log_data = []

        for log in logs:
            log_data.append({
                "log": log,
                "recommendations": analyze_recovery(log),
            })

        context["log_data"] = log_data
        context["edit_log"] = edit_log

        return context

    def post(self, request, *args, **kwargs):
        edit_log = None

        if "pk" in self.kwargs:
            edit_log = get_object_or_404(
                RecoveryHydrationLog,
                pk=self.kwargs["pk"],
                user=self.request.user,
            )
        form = RecoveryHydrationLogForm(
            request.POST,
            instance=edit_log,
        )

        if form.is_valid():
            log = form.save(commit=False)
            log.user = request.user
            log.save()

        return redirect("manager:recovery-hydration")