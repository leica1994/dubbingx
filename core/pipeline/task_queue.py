"""
线程安全任务队列模块

实现了支持监听器模式的任务队列，用于步骤间的任务流转
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

from .task import Task


class TaskEventType(Enum):
    """任务事件类型"""
    TASK_ADDED = "task_added"           # 任务添加到队列
    TASK_REMOVED = "task_removed"       # 任务从队列移除
    TASK_COMPLETED = "task_completed"   # 任务处理完成
    TASK_FAILED = "task_failed"         # 任务处理失败
    QUEUE_EMPTY = "queue_empty"         # 队列为空
    QUEUE_FULL = "queue_full"           # 队列已满


@dataclass
class TaskEvent:
    """任务事件"""
    event_type: TaskEventType
    task: Optional[Task] = None
    queue_name: str = ""
    message: str = ""
    data: Dict[str, Any] = None


class TaskListener:
    """任务监听器接口"""
    
    def on_task_event(self, event: TaskEvent) -> None:
        """处理任务事件"""
        pass


class TaskQueue:
    """线程安全的任务队列"""
    
    def __init__(self, name: str, maxsize: int = 0):
        """
        初始化任务队列
        
        Args:
            name: 队列名称
            maxsize: 最大队列大小，0表示无限制
        """
        self.name = name
        self.maxsize = maxsize
        self._queue: Queue[Task] = Queue(maxsize=maxsize)
        self._listeners: List[TaskListener] = []
        self._lock = threading.Lock()
        self._stats = {
            "total_added": 0,
            "total_removed": 0,
            "total_completed": 0,
            "total_failed": 0,
        }
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def put(self, task: Task, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        向队列添加任务
        
        Args:
            task: 要添加的任务
            block: 是否阻塞等待
            timeout: 超时时间
            
        Returns:
            是否成功添加
        """
        try:
            self._queue.put(task, block=block, timeout=timeout)
            
            with self._lock:
                self._stats["total_added"] += 1
            
            # 触发任务添加事件
            event = TaskEvent(
                event_type=TaskEventType.TASK_ADDED,
                task=task,
                queue_name=self.name,
                message=f"任务 {task.task_id} 已添加到队列 {self.name}"
            )
            self._notify_listeners(event)
            
            self.logger.debug(f"任务 {task.task_id} 已添加到队列 {self.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加任务到队列 {self.name} 失败: {e}")
            return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Task]:
        """
        从队列获取任务
        
        Args:
            block: 是否阻塞等待
            timeout: 超时时间
            
        Returns:
            获取的任务，如果超时或队列为空则返回None
        """
        try:
            task = self._queue.get(block=block, timeout=timeout)
            
            with self._lock:
                self._stats["total_removed"] += 1
            
            # 触发任务移除事件
            event = TaskEvent(
                event_type=TaskEventType.TASK_REMOVED,
                task=task,
                queue_name=self.name,
                message=f"任务 {task.task_id} 已从队列 {self.name} 移除"
            )
            self._notify_listeners(event)
            
            self.logger.debug(f"任务 {task.task_id} 已从队列 {self.name} 移除")
            return task
            
        except Empty:
            # 队列为空时触发事件
            if not block:
                event = TaskEvent(
                    event_type=TaskEventType.QUEUE_EMPTY,
                    queue_name=self.name,
                    message=f"队列 {self.name} 为空"
                )
                self._notify_listeners(event)
            
            return None
        except Exception as e:
            self.logger.error(f"从队列 {self.name} 获取任务失败: {e}")
            return None
    
    def task_done(self, task: Task, success: bool = True, message: str = "") -> None:
        """
        标记任务处理完成
        
        Args:
            task: 已处理的任务
            success: 是否成功处理
            message: 处理消息
        """
        try:
            self._queue.task_done()
            
            with self._lock:
                if success:
                    self._stats["total_completed"] += 1
                else:
                    self._stats["total_failed"] += 1
            
            # 触发任务完成事件
            event_type = TaskEventType.TASK_COMPLETED if success else TaskEventType.TASK_FAILED
            event = TaskEvent(
                event_type=event_type,
                task=task,
                queue_name=self.name,
                message=message or f"任务 {task.task_id} 在队列 {self.name} 处理{'成功' if success else '失败'}"
            )
            self._notify_listeners(event)
            
            self.logger.debug(f"任务 {task.task_id} 在队列 {self.name} 处理{'成功' if success else '失败'}")
            
        except Exception as e:
            self.logger.error(f"标记任务完成失败: {e}")
    
    def add_listener(self, listener: TaskListener) -> None:
        """添加任务监听器"""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)
                self.logger.debug(f"已添加监听器到队列 {self.name}")
    
    def remove_listener(self, listener: TaskListener) -> None:
        """移除任务监听器"""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
                self.logger.debug(f"已从队列 {self.name} 移除监听器")
    
    def _notify_listeners(self, event: TaskEvent) -> None:
        """通知所有监听器"""
        with self._lock:
            listeners = self._listeners.copy()
        
        for listener in listeners:
            try:
                listener.on_task_event(event)
            except Exception as e:
                self.logger.error(f"监听器处理事件失败: {e}")
    
    def qsize(self) -> int:
        """获取队列大小"""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """检查队列是否为空"""
        return self._queue.empty()
    
    def full(self) -> bool:
        """检查队列是否已满"""
        return self._queue.full()
    
    def join(self) -> None:
        """等待队列中所有任务完成"""
        self._queue.join()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        with self._lock:
            return {
                "name": self.name,
                "size": self.qsize(),
                "maxsize": self.maxsize,
                "empty": self.empty(),
                "full": self.full(),
                "stats": self._stats.copy(),
                "listener_count": len(self._listeners),
            }
    
    def clear(self) -> int:
        """清空队列并返回清理的任务数量"""
        count = 0
        with self._lock:
            try:
                while not self._queue.empty():
                    self._queue.get_nowait()
                    count += 1
            except Empty:
                pass
        
        self.logger.info(f"队列 {self.name} 已清空，移除了 {count} 个任务")
        return count


class QueueManager:
    """队列管理器"""
    
    def __init__(self):
        self._queues: Dict[str, TaskQueue] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def create_queue(self, name: str, maxsize: int = 0) -> TaskQueue:
        """创建任务队列"""
        with self._lock:
            if name in self._queues:
                self.logger.warning(f"队列 {name} 已存在")
                return self._queues[name]
            
            queue = TaskQueue(name, maxsize)
            self._queues[name] = queue
            self.logger.info(f"已创建队列 {name}")
            return queue
    
    def get_queue(self, name: str) -> Optional[TaskQueue]:
        """获取指定名称的队列"""
        with self._lock:
            return self._queues.get(name)
    
    def remove_queue(self, name: str) -> bool:
        """移除指定名称的队列"""
        with self._lock:
            if name in self._queues:
                queue = self._queues[name]
                # 清空队列
                cleared_count = queue.clear()
                del self._queues[name]
                self.logger.info(f"已移除队列 {name}，清理了 {cleared_count} 个任务")
                return True
            return False
    
    def get_all_queues(self) -> Dict[str, TaskQueue]:
        """获取所有队列"""
        with self._lock:
            return self._queues.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取所有队列的统计信息"""
        with self._lock:
            return {
                "total_queues": len(self._queues),
                "queues": {name: queue.get_stats() for name, queue in self._queues.items()}
            }
    
    def clear_all(self) -> Dict[str, int]:
        """清空所有队列"""
        with self._lock:
            result = {}
            for name, queue in self._queues.items():
                count = queue.clear()
                result[name] = count
            
            self.logger.info(f"已清空所有队列: {result}")
            return result