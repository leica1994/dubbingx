"""
任务监听器模块

实现任务事件的监听和处理，支持任务状态变化通知和自动流转
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .task import TaskStatus
from .task_queue import TaskEvent, TaskEventType
from .task_queue import TaskListener as BaseTaskListener


class TaskListener(BaseTaskListener):
    """任务监听器基类"""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self._enabled = True

    def enable(self) -> None:
        """启用监听器"""
        self._enabled = True
        self.logger.debug(f"监听器 {self.name} 已启用")

    def disable(self) -> None:
        """禁用监听器"""
        self._enabled = False
        self.logger.debug(f"监听器 {self.name} 已禁用")

    def is_enabled(self) -> bool:
        """检查监听器是否启用"""
        return self._enabled

    def on_task_event(self, event: TaskEvent) -> None:
        """处理任务事件（基类实现）"""
        if not self._enabled:
            return

        try:
            # 根据事件类型分发处理
            if event.event_type == TaskEventType.TASK_ADDED:
                self.on_task_added(event)
            elif event.event_type == TaskEventType.TASK_REMOVED:
                self.on_task_removed(event)
            elif event.event_type == TaskEventType.TASK_COMPLETED:
                self.on_task_completed(event)
            elif event.event_type == TaskEventType.TASK_FAILED:
                self.on_task_failed(event)
            elif event.event_type == TaskEventType.QUEUE_EMPTY:
                self.on_queue_empty(event)
            elif event.event_type == TaskEventType.QUEUE_FULL:
                self.on_queue_full(event)

        except Exception as e:
            self.logger.error(f"监听器 {self.name} 处理事件失败: {e}", exc_info=True)

    def on_task_added(self, event: TaskEvent) -> None:
        """任务添加到队列时触发"""
        pass

    def on_task_removed(self, event: TaskEvent) -> None:
        """任务从队列移除时触发"""
        pass

    def on_task_completed(self, event: TaskEvent) -> None:
        """任务处理完成时触发"""
        pass

    def on_task_failed(self, event: TaskEvent) -> None:
        """任务处理失败时触发"""
        pass

    def on_queue_empty(self, event: TaskEvent) -> None:
        """队列为空时触发"""
        pass

    def on_queue_full(self, event: TaskEvent) -> None:
        """队列已满时触发"""
        pass


class LoggingTaskListener(TaskListener):
    """日志记录监听器"""

    def __init__(self, log_level: int = logging.INFO):
        super().__init__("LoggingListener")
        self.log_level = log_level

    def on_task_added(self, event: TaskEvent) -> None:
        """记录任务添加日志"""
        if event.task:
            self.logger.log(
                self.log_level,
                f"任务 {event.task.task_id} 已添加到队列 {event.queue_name}",
            )

    def on_task_removed(self, event: TaskEvent) -> None:
        """记录任务移除日志"""
        if event.task:
            self.logger.log(
                self.log_level,
                f"任务 {event.task.task_id} 已从队列 {event.queue_name} 移除",
            )

    def on_task_completed(self, event: TaskEvent) -> None:
        """记录任务完成日志"""
        if event.task:
            self.logger.log(
                self.log_level,
                f"任务 {event.task.task_id} 在队列 {event.queue_name} 处理成功",
            )

    def on_task_failed(self, event: TaskEvent) -> None:
        """记录任务失败日志"""
        if event.task:
            self.logger.warning(
                f"任务 {event.task.task_id} 在队列 {event.queue_name} 处理失败: {event.message}"
            )

    def on_queue_empty(self, event: TaskEvent) -> None:
        """记录队列为空日志"""
        self.logger.log(self.log_level, f"队列 {event.queue_name} 为空")

    def on_queue_full(self, event: TaskEvent) -> None:
        """记录队列已满日志"""
        self.logger.warning(f"队列 {event.queue_name} 已满")


class StatisticsTaskListener(TaskListener):
    """统计信息监听器"""

    def __init__(self):
        super().__init__("StatisticsListener")
        self._stats = {
            "total_added": 0,
            "total_removed": 0,
            "total_completed": 0,
            "total_failed": 0,
            "queue_empty_events": 0,
            "queue_full_events": 0,
        }
        self._queue_stats = {}  # 按队列统计

    def on_task_added(self, event: TaskEvent) -> None:
        """更新添加统计"""
        self._stats["total_added"] += 1
        self._update_queue_stats(event.queue_name, "added", 1)

    def on_task_removed(self, event: TaskEvent) -> None:
        """更新移除统计"""
        self._stats["total_removed"] += 1
        self._update_queue_stats(event.queue_name, "removed", 1)

    def on_task_completed(self, event: TaskEvent) -> None:
        """更新完成统计"""
        self._stats["total_completed"] += 1
        self._update_queue_stats(event.queue_name, "completed", 1)

    def on_task_failed(self, event: TaskEvent) -> None:
        """更新失败统计"""
        self._stats["total_failed"] += 1
        self._update_queue_stats(event.queue_name, "failed", 1)

    def on_queue_empty(self, event: TaskEvent) -> None:
        """更新队列为空统计"""
        self._stats["queue_empty_events"] += 1
        self._update_queue_stats(event.queue_name, "empty_events", 1)

    def on_queue_full(self, event: TaskEvent) -> None:
        """更新队列已满统计"""
        self._stats["queue_full_events"] += 1
        self._update_queue_stats(event.queue_name, "full_events", 1)

    def _update_queue_stats(
        self, queue_name: str, stat_name: str, increment: int
    ) -> None:
        """更新指定队列的统计信息"""
        if queue_name not in self._queue_stats:
            self._queue_stats[queue_name] = {
                "added": 0,
                "removed": 0,
                "completed": 0,
                "failed": 0,
                "empty_events": 0,
                "full_events": 0,
            }

        self._queue_stats[queue_name][stat_name] += increment

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "global_stats": self._stats.copy(),
            "queue_stats": self._queue_stats.copy(),
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = {
            "total_added": 0,
            "total_removed": 0,
            "total_completed": 0,
            "total_failed": 0,
            "queue_empty_events": 0,
            "queue_full_events": 0,
        }
        self._queue_stats = {}
        self.logger.info("统计信息已重置")


class TaskFlowListener(TaskListener):
    """任务流控制监听器"""

    def __init__(self, task_scheduler):
        super().__init__("TaskFlowListener")
        self.task_scheduler = task_scheduler

    def on_task_completed(self, event: TaskEvent) -> None:
        """任务完成时，将其推送到下一个步骤"""
        self.logger.info(f"🔄 TaskFlowListener: 收到任务完成事件 - 队列: {event.queue_name}")
        
        if not event.task:
            self.logger.error("❌ TaskFlowListener: 事件中没有任务对象")
            return

        task = event.task
        self.logger.info(f"📋 TaskFlowListener: 处理任务 {task.task_id} 的完成事件")

        # 从队列名称中解析当前步骤ID
        # 队列名称格式: step_{step_id}_{step_name}
        current_step = self._extract_step_id_from_queue_name(event.queue_name)
        if current_step is None:
            self.logger.error(f"❌ 无法从队列名称 {event.queue_name} 解析步骤ID")
            return
        
        self.logger.info(f"📍 TaskFlowListener: 当前步骤 {current_step}, 任务 {task.task_id}")

        # 计算下一个步骤
        next_step = current_step + 1
        if next_step < 8:  # 总共8个步骤 (0-7)
            self.logger.info(f"⏭️ TaskFlowListener: 准备推送任务 {task.task_id} 到步骤 {next_step}")
            
            # 手动更新任务进度到下一步
            task.update_progress(next_step)

            # 将任务提交到下一个步骤的队列
            try:
                success = self.task_scheduler.submit_task_to_step(task, next_step)
                if success:
                    self.logger.info(f"✅ 任务 {task.task_id} 成功从步骤 {current_step} 推送到步骤 {next_step}")
                else:
                    self.logger.error(f"❌ 任务 {task.task_id} 推送到步骤 {next_step} 失败: 提交返回False")
            except Exception as e:
                self.logger.error(
                    f"❌ 推送任务 {task.task_id} 到步骤 {next_step} 失败: {e}"
                )
                task.update_status(TaskStatus.FAILED, f"任务流转失败: {str(e)}")
        else:
            # 所有步骤完成
            task.update_status(TaskStatus.COMPLETED, "所有处理步骤已完成")
            self.logger.info(f"🎉 任务 {task.task_id} 已完成所有处理步骤")

    def _extract_step_id_from_queue_name(self, queue_name: str) -> Optional[int]:
        """从队列名称中提取步骤ID"""
        try:
            # 队列名称格式: step_{step_id}_{step_name}
            parts = queue_name.split("_")
            if len(parts) >= 2 and parts[0] == "step":
                return int(parts[1])
        except (ValueError, IndexError):
            pass
        return None

    def on_task_failed(self, event: TaskEvent) -> None:
        """任务失败时的处理"""
        if not event.task:
            return

        task = event.task

        # 从队列名称中解析当前步骤ID
        current_step = self._extract_step_id_from_queue_name(event.queue_name)
        if current_step is None:
            self.logger.error(f"无法从队列名称 {event.queue_name} 解析步骤ID")
            return

        # 检查是否可以重试
        if task.can_retry():
            task.increment_retry()

            try:
                # 重置任务进度到当前步骤（因为可能已经被更新）
                task.update_progress(current_step)

                # 重新提交到当前步骤
                self.task_scheduler.submit_task_to_step(task, current_step)
                self.logger.info(
                    f"任务 {task.task_id} 在步骤 {current_step} 开始第 {task.retry_count} 次重试"
                )
            except Exception as e:
                self.logger.error(f"重试任务 {task.task_id} 失败: {e}")
                task.update_status(TaskStatus.FAILED, f"重试失败: {str(e)}")
        else:
            # 无法重试，标记为最终失败
            task.update_status(TaskStatus.FAILED, "已达到最大重试次数")
            self.logger.error(
                f"任务 {task.task_id} 在步骤 {current_step} 已达到最大重试次数，标记为失败"
            )


class CallbackTaskListener(TaskListener):
    """回调函数监听器"""

    def __init__(self, callbacks: Dict[TaskEventType, Callable[[TaskEvent], None]]):
        super().__init__("CallbackListener")
        self.callbacks = callbacks or {}

    def add_callback(
        self, event_type: TaskEventType, callback: Callable[[TaskEvent], None]
    ) -> None:
        """添加回调函数"""
        self.callbacks[event_type] = callback

    def remove_callback(self, event_type: TaskEventType) -> None:
        """移除回调函数"""
        self.callbacks.pop(event_type, None)

    def on_task_event(self, event: TaskEvent) -> None:
        """处理任务事件"""
        if not self._enabled:
            return

        callback = self.callbacks.get(event.event_type)
        if callback:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"回调函数执行失败: {e}", exc_info=True)


class CompositeTaskListener(TaskListener):
    """组合监听器 - 可以包含多个子监听器"""

    def __init__(self, name: str = "CompositeListener"):
        super().__init__(name)
        self._listeners: List[TaskListener] = []

    def add_listener(self, listener: TaskListener) -> None:
        """添加子监听器"""
        if listener not in self._listeners:
            self._listeners.append(listener)
            self.logger.debug(f"已添加子监听器: {listener.name}")

    def remove_listener(self, listener: TaskListener) -> None:
        """移除子监听器"""
        if listener in self._listeners:
            self._listeners.remove(listener)
            self.logger.debug(f"已移除子监听器: {listener.name}")

    def on_task_event(self, event: TaskEvent) -> None:
        """将事件分发给所有子监听器"""
        if not self._enabled:
            return

        for listener in self._listeners:
            try:
                listener.on_task_event(event)
            except Exception as e:
                self.logger.error(
                    f"子监听器 {listener.name} 处理事件失败: {e}", exc_info=True
                )

    def enable_all(self) -> None:
        """启用所有子监听器"""
        for listener in self._listeners:
            listener.enable()
        self.logger.debug("已启用所有子监听器")

    def disable_all(self) -> None:
        """禁用所有子监听器"""
        for listener in self._listeners:
            listener.disable()
        self.logger.debug("已禁用所有子监听器")

    def get_listeners(self) -> List[TaskListener]:
        """获取所有子监听器"""
        return self._listeners.copy()
