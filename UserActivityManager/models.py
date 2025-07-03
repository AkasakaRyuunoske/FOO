from django.db import models

from UserManager.models import CustomUser


class ClusterTag(models.Model):
    id = models.IntegerField(primary_key=True)
    created_by_user_id = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name="clusters")
    name = models.CharField(max_length=255, unique=True, null=False)
    description = models.TextField(max_length=4000, null=False)

    class Meta:
        verbose_name = "Tag Cluster"
        verbose_name_plural = "Tag Clusters"

    def __str__(self): return self.name
