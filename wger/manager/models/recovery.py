from django.conf import settings

from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator

class RecoveryHydrationLog(models.Model):

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE
    )

    date = models.DateField()
    water_intake = models.PositiveIntegerField(help_text="Water intake in ml (1 cup of water = 250ml)...")
    sleep_hours = models.DecimalField(max_digits=3, decimal_places=2)
    energy_level = models.PositiveSmallIntegerField(validators= [MinValueValidator(0), MaxValueValidator(5)])

    created_at = models.DateTimeField(auto_now_add=True)

    #Returns the recent date it was made
    class Meta:
        ordering = ["-date"]

    def __str__(self):
        return f"{self.user} - {self.date}"
