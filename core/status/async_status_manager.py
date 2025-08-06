"""
异步状态管理系统

提供完全异步、非阻塞的状态更新机制，防止GUI卡住
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from PySide6.QtCore import QObject, QTimer, Signal


class StatusEventType(Enum):
    """状态事件类型"""
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    STEP_STARTED = "step_started"
    STEP_PROGRESS = "step_progress"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    BATCH_UPDATE = "batch_update"


@dataclass
class StatusEvent:
    """状态事件数据类"""
    event_type: StatusEventType
    task_id: str
    step_id: Optional[int] = None
    status: str = ""
    message: str = ""
    progress: float = 0.0
    current_item: int = 0
    total_items: int = 1
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0

    def __post_init__(self):
        """后处理初始化"""
        if self.sequence == 0:
            self.sequence = int(self.timestamp * 1000000)  # 微秒级序列号


class StatusCache:
    """内存状态缓存系统"""

    def __init__(self):
        """初始化状态缓存"""
        self.logger = logging.getLogger(f"{__name__}.StatusCache")
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 任务状态缓存 {task_id: task_info}
        self._tasks = {}
        
        # 步骤状态缓存 {(task_id, step_id): step_info}
        self._steps = {}
        
        # 最近更新时间
        self._last_update = {}
        
        # 状态变更监听器
        self._listeners: List[Callable[[StatusEvent], None]] = []

    def add_listener(self, listener: Callable[[StatusEvent], None]) -> None:
        """添加状态变更监听器"""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[StatusEvent], None]) -> None:
        """移除状态变更监听器"""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def update_task_status(self, event: StatusEvent) -> bool:
        """更新任务状态"""
        with self._lock:
            try:
                task_id = event.task_id
                now = time.time()
                
                # 获取或创建任务信息
                if task_id not in self._tasks:
                    self._tasks[task_id] = {
                        "task_id": task_id,
                        "status": "pending",
                        "progress": 0.0,
                        "current_step": 0,
                        "total_steps": 8,
                        "started_at": None,
                        "updated_at": now,
                        "completed_at": None,
                        "error_message": None,
                        "metadata": {}
                    }
                
                task_info = self._tasks[task_id]
                
                # 更新任务信息
                if event.event_type == StatusEventType.TASK_STARTED:
                    task_info["status"] = "processing"
                    task_info["started_at"] = now
                elif event.event_type == StatusEventType.TASK_COMPLETED:
                    task_info["status"] = "completed"
                    task_info["progress"] = 100.0
                    task_info["completed_at"] = now
                elif event.event_type == StatusEventType.TASK_FAILED:
                    task_info["status"] = "failed"
                    task_info["error_message"] = event.error_message
                    task_info["completed_at"] = now
                
                task_info["updated_at"] = now
                task_info["metadata"].update(event.metadata)
                
                # 记录更新时间
                self._last_update[task_id] = now
                
                # 通知监听器（异步）
                self._notify_listeners_async(event)
                
                return True
                
            except Exception as e:
                self.logger.error(f"更新任务状态失败: {e}")
                return False

    def update_step_status(self, event: StatusEvent) -> bool:
        """更新步骤状态"""
        with self._lock:
            try:
                task_id = event.task_id
                step_id = event.step_id
                now = time.time()
                
                if step_id is None:
                    return False
                
                step_key = (task_id, step_id)
                
                # 获取或创建步骤信息
                if step_key not in self._steps:
                    self._steps[step_key] = {
                        "task_id": task_id,
                        "step_id": step_id,
                        "status": "pending",
                        "progress": 0.0,
                        "current_item": 0,
                        "total_items": 1,
                        "started_at": None,
                        "updated_at": now,
                        "completed_at": None,
                        "error_message": None,
                        "metadata": {}
                    }
                
                step_info = self._steps[step_key]
                
                # 更新步骤信息
                if event.event_type == StatusEventType.STEP_STARTED:
                    step_info["status"] = "processing"
                    step_info["started_at"] = now
                    step_info["total_items"] = event.total_items
                elif event.event_type == StatusEventType.STEP_PROGRESS:
                    step_info["status"] = "processing"
                    step_info["progress"] = event.progress
                    step_info["current_item"] = event.current_item
                    step_info["total_items"] = event.total_items
                elif event.event_type == StatusEventType.STEP_COMPLETED:
                    step_info["status"] = "completed"
                    step_info["progress"] = 100.0
                    step_info["current_item"] = step_info["total_items"]
                    step_info["completed_at"] = now
                elif event.event_type == StatusEventType.STEP_FAILED:
                    step_info["status"] = "failed"
                    step_info["error_message"] = event.error_message
                    step_info["completed_at"] = now
                
                step_info["updated_at"] = now
                step_info["metadata"].update(event.metadata)
                
                # 更新任务当前步骤
                if task_id in self._tasks:
                    task_info = self._tasks[task_id]
                    if step_info["status"] == "completed":
                        # 计算已完成步骤数
                        completed_steps = sum(1 for key, info in self._steps.items() 
                                            if key[0] == task_id and info["status"] == "completed")
                        task_info["current_step"] = completed_steps
                        task_info["progress"] = (completed_steps / 8) * 100.0
                
                # 记录更新时间
                self._last_update[step_key] = now
                
                # 通知监听器（异步）
                self._notify_listeners_async(event)
                
                return True
                
            except Exception as e:
                self.logger.error(f"更新步骤状态失败: {e}")
                return False

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        with self._lock:
            return self._tasks.get(task_id, {}).copy() if task_id in self._tasks else None

    def get_step_status(self, task_id: str, step_id: int) -> Optional[Dict[str, Any]]:
        """获取步骤状态"""
        with self._lock:
            step_key = (task_id, step_id)
            return self._steps.get(step_key, {}).copy() if step_key in self._steps else None

    def get_task_steps(self, task_id: str) -> Dict[int, Dict[str, Any]]:
        """获取任务的所有步骤状态"""
        with self._lock:
            steps = {}
            for (tid, step_id), step_info in self._steps.items():
                if tid == task_id:
                    steps[step_id] = step_info.copy()
            return steps

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """获取所有任务状态"""
        with self._lock:
            return {task_id: task_info.copy() for task_id, task_info in self._tasks.items()}

    def clear_task(self, task_id: str) -> bool:
        """清理任务状态"""
        with self._lock:
            try:
                # 删除任务
                if task_id in self._tasks:
                    del self._tasks[task_id]
                
                # 删除步骤
                step_keys_to_remove = [key for key in self._steps.keys() if key[0] == task_id]
                for key in step_keys_to_remove:
                    del self._steps[key]
                
                # 清理更新时间
                keys_to_remove = [key for key in self._last_update.keys() 
                                 if (isinstance(key, str) and key == task_id) or
                                    (isinstance(key, tuple) and key[0] == task_id)]
                for key in keys_to_remove:
                    del self._last_update[key]
                
                return True
                
            except Exception as e:
                self.logger.error(f"清理任务状态失败: {e}")
                return False

    def _notify_listeners_async(self, event: StatusEvent) -> None:
        """异步通知监听器"""
        def notify():
            for listener in self._listeners.copy():  # 复制列表避免并发修改
                try:
                    listener(event)
                except Exception as e:
                    self.logger.debug(f"监听器通知失败: {e}")
        
        # 在新线程中执行通知
        threading.Thread(target=notify, daemon=True).start()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            return {
                "tasks_count": len(self._tasks),
                "steps_count": len(self._steps),
                "last_updates": len(self._last_update),
                "listeners_count": len(self._listeners)
            }