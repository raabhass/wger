from django import forms

from wger.manager.models.recovery import RecoveryHydrationLog


class RecoveryHydrationLogForm(forms.ModelForm):
    class Meta:
        model = RecoveryHydrationLog

        fields = [
            "date",
            "water_intake",
            "sleep_hours",
            "energy_level",
        ]

        widgets = {
            "date": forms.DateInput(
                attrs={
                    "type": "date",
                    "class": "form-control",
                }
            ),

            "water_intake": forms.NumberInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "Water intake in ml. i.e 1 cup = 250 ml",
                }
            ),

            "sleep_hours": forms.NumberInput(
                attrs={
                    "class": "form-control",
                    "step": "0.1",
                    "placeholder": "Hours of sleep",
                }
            ),

            "energy_level": forms.NumberInput(
                attrs={
                    "class": "form-control",
                    "min": 0,
                    "max": 5,
                    "placeholder": "0-5",
                }
            ),
        }