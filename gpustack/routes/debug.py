import asyncio
import logging
import tracemalloc
from fastapi import APIRouter, Request

from gpustack.api.exceptions import (
    BadRequestException,
    InvalidException,
)

from gpustack.server.bus import event_bus

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/log_level")
async def get_log_level():
    current_level = logging.getLogger().level
    return logging.getLevelName(current_level)


@router.put("/log_level")
async def set_log_level(request: Request):
    level = await request.body()
    level_str = level.decode("utf-8").upper().strip()
    numeric_level = logging._nameToLevel.get(level_str)
    if not isinstance(numeric_level, int):
        raise InvalidException(message="Invalid log level")

    logging.getLogger().setLevel(numeric_level)
    logger.info(f"Set log level to {level_str}")
    return "ok"


@router.get("/memory")
def get_memory_profile():
    if not tracemalloc.is_tracing():
        raise BadRequestException(
            message="tracemalloc is not enabled. Please run GPUStack server in debug mode."
        )

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    result = [str(stat) for stat in top_stats[:20]]
    return {"top_memory_lines": result}


@router.get("/bus")
async def get_event_bus():
    response = {
        "modelinstance_subscribers": [],
        "model_subscribers": [],
        "summary": {},
        "tasks_info": {},
    }

    # ModelInstance 订阅者信息
    modelinstance_subs = event_bus.subscribers.get('modelinstance', [])
    for i, sub in enumerate(modelinstance_subs):
        response["modelinstance_subscribers"].append(
            {
                "id": i,
                "queue_contents": sub.queue.get_queue_contents_info(),
                "queue_stats": sub.queue.get_queue_stats(),
            }
        )

    # Model 订阅者信息
    model_subs = event_bus.subscribers.get('model', [])
    for i, sub in enumerate(model_subs):
        response["model_subscribers"].append(
            {
                "id": i,
                "queue_contents": sub.queue.get_queue_contents_info(),
                "queue_stats": sub.queue.get_queue_stats(),
            }
        )

    # 异步任务信息
    tasks = asyncio.all_tasks()
    response["tasks_info"] = {
        "total_tasks": len(tasks),
        "tasks": [str(task) for task in tasks],  # 可以自定义任务信息格式
    }

    # 汇总信息
    response["summary"] = {
        "total_modelinstance_subscribers": len(modelinstance_subs),
        "total_model_subscribers": len(model_subs),
        "total_tasks": len(tasks),
    }

    return response
