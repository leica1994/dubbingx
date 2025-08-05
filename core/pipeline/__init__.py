"""
队列+监听模式的多阶段任务处理系统

该模块实现了基于事件驱动的任务调度系统，支持真正的流水线并行处理。
核心特性：
- 每个处理步骤独立队列
- 事件驱动的任务流转
- 真正的异步并发处理
- 实时状态监控
- 容错和重试机制
"""

from .processors import (AlignAudioProcessor, GenerateAlignedSrtProcessor,
                         GenerateReferenceAudioProcessor, GenerateTTSProcessor,
                         MergeAudioVideoProcessor, PreprocessSubtitleProcessor,
                         ProcessVideoSpeedProcessor, SeparateMediaProcessor)
from .step_processor import StepProcessor
from .task import ProcessResult, ResourceType, Task, TaskStatus
from .task_listener import (CallbackTaskListener, CompositeTaskListener,
                            LoggingTaskListener, StatisticsTaskListener,
                            TaskFlowListener, TaskListener)
from .task_queue import QueueManager, TaskEvent, TaskEventType, TaskQueue
from .task_scheduler import TaskScheduler

__all__ = [
    "Task",
    "TaskStatus",
    "ProcessResult",
    "ResourceType",
    "TaskQueue",
    "TaskEvent",
    "TaskEventType",
    "QueueManager",
    "StepProcessor",
    "TaskListener",
    "LoggingTaskListener",
    "StatisticsTaskListener",
    "TaskFlowListener",
    "CallbackTaskListener",
    "CompositeTaskListener",
    "TaskScheduler",
    "PreprocessSubtitleProcessor",
    "SeparateMediaProcessor",
    "GenerateReferenceAudioProcessor",
    "GenerateTTSProcessor",
    "AlignAudioProcessor",
    "GenerateAlignedSrtProcessor",
    "ProcessVideoSpeedProcessor",
    "MergeAudioVideoProcessor",
]
