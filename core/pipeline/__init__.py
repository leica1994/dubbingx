"""
队列+监听模式的多阶段任务处理系统

该模块实现了基于事件驱动的任务调度系统，支持真正的流水线并行处理。
核心特性：
- 每个处理步骤独立队列
- 事件驱动的任务流转
- 智能资源调度 (GPU/CPU/IO)
- 实时状态监控
- 容错和重试机制
"""

from .task import Task, TaskStatus, ProcessResult, ResourceType
from .task_queue import TaskQueue, TaskEvent, TaskEventType, QueueManager
from .resource_manager import ResourceManager
from .step_processor import StepProcessor
from .task_listener import (
    TaskListener,
    LoggingTaskListener,
    StatisticsTaskListener,
    TaskFlowListener,
    CallbackTaskListener,
    CompositeTaskListener
)
from .task_scheduler import TaskScheduler

__all__ = [
    'Task',
    'TaskStatus', 
    'ProcessResult',
    'ResourceType',
    'TaskQueue',
    'TaskEvent',
    'TaskEventType',
    'QueueManager',
    'ResourceManager',
    'StepProcessor',
    'TaskListener',
    'LoggingTaskListener',
    'StatisticsTaskListener',
    'TaskFlowListener',
    'CallbackTaskListener',
    'CompositeTaskListener',
    'TaskScheduler',
]