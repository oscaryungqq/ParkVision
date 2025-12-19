from celery import Celery

redis_url = "redis://localhost:6379/0"

celery_app = Celery("tasks", broker=redis_url, backend=redis_url)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)

celery_app.conf.imports = ("worker.tasks",)