"""
主任务调度器模块

负责协调整个流水线处理系统，管理队列、工作线程和资源分配
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .step_processor import StepProcessor
from .task import ProcessResult, StepStatus, Task, TaskStatus
from .task_listener import (
    CompositeTaskListener,
    LoggingTaskListener,
    StatisticsTaskListener,
    TaskFlowListener,
    TaskListener,
)
from .task_queue import QueueManager


class TaskScheduler:
    """主任务调度器"""

    # 处理步骤定义（移除资源类型限制，实现真正的异步并发）
    STEP_DEFINITIONS = [
        (0, "preprocess_subtitle"),
        (1, "separate_media"),
        (2, "generate_reference_audio"),
        (3, "generate_tts"),
        (4, "align_audio"),
        (5, "generate_aligned_srt"),
        (6, "process_video_speed"),
        (7, "merge_audio_video"),
    ]

    def __init__(
        self,
        max_workers_per_step: Optional[Dict[int, int]] = None,
    ):
        """
        初始化任务调度器

        Args:
            max_workers_per_step: 每个步骤的最大工作线程数
        """
        self.logger = logging.getLogger(__name__)

        # 工作线程数配置（每个步骤独立配置）
        self.max_workers_per_step = max_workers_per_step or {
            0: 8,  # preprocess_subtitle - 增加到8个线程处理大批量任务
            1: 2,  # separate_media - GPU任务
            2: 4,  # generate_reference_audio - 增加线程数
            3: 2,  # generate_tts - GPU任务
            4: 4,  # align_audio - 增加线程数
            5: 4,  # generate_aligned_srt - I/O任务
            6: 4,  # process_video_speed - I/O任务
            7: 4,  # merge_audio_video - I/O任务
        }

        # 队列管理器
        self.queue_manager = QueueManager()

        # 步骤处理器
        self.processors: Dict[int, StepProcessor] = {}

        # 工作线程池
        self.worker_pools: Dict[int, ThreadPoolExecutor] = {}

        # 监听器
        self.master_listener = CompositeTaskListener("MasterListener")

        # 控制状态
        self._running = False
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

        # 任务跟踪
        self._active_tasks: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, Task] = {}
        self._failed_tasks: Dict[str, Task] = {}

        # 初始化组件
        self._initialize_queues()
        self._initialize_listeners()

        self.logger.info("任务调度器初始化完成")

    def _initialize_queues(self) -> None:
        """初始化所有步骤的队列"""
        for step_id, step_name in self.STEP_DEFINITIONS:
            queue = self.queue_manager.create_queue(f"step_{step_id}_{step_name}")
            self.logger.debug(f"已创建队列: {queue.name}")

    def _initialize_listeners(self) -> None:
        """初始化监听器"""
        # 添加日志监听器（DEBUG级别显示更多调试信息）
        log_listener = LoggingTaskListener(logging.DEBUG)
        self.master_listener.add_listener(log_listener)

        # 添加统计监听器
        stats_listener = StatisticsTaskListener()
        self.master_listener.add_listener(stats_listener)

        # 添加任务流控制监听器
        flow_listener = TaskFlowListener(self)
        self.master_listener.add_listener(flow_listener)

        # 将主监听器添加到所有队列
        for queue in self.queue_manager.get_all_queues().values():
            queue.add_listener(self.master_listener)

    def register_processor(self, processor: StepProcessor) -> None:
        """
        注册步骤处理器

        Args:
            processor: 要注册的处理器
        """
        with self._lock:
            self.processors[processor.step_id] = processor
            self.logger.info(f"已注册步骤处理器: {processor.step_name}")

    def start(self) -> None:
        """启动调度器"""
        with self._lock:
            if self._running:
                self.logger.warning("调度器已在运行中")
                return

            self._running = True
            self._shutdown_event.clear()

        # 启动所有步骤的工作线程池（移除资源管理，实现真正的异步并发）
        for step_id, step_name in self.STEP_DEFINITIONS:
            if step_id not in self.processors:
                self.logger.error(f"步骤 {step_id} ({step_name}) 未注册处理器")
                continue

            max_workers = self.max_workers_per_step.get(step_id, 2)
            executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=f"Step-{step_id}-{step_name}",
            )

            self.worker_pools[step_id] = executor

            # 为每个工作线程启动任务处理循环
            for _ in range(max_workers):
                executor.submit(self._worker_loop, step_id)

            self.logger.info(
                f"已启动步骤 {step_id} ({step_name}) 的 {max_workers} 个工作线程"
            )

        self.logger.info("任务调度器已启动")

    def stop(self, timeout: float = 30.0) -> None:
        """
        停止调度器

        Args:
            timeout: 等待超时时间（秒）
        """
        with self._lock:
            if not self._running:
                self.logger.warning("调度器未在运行")
                return

            self._running = False
            self._shutdown_event.set()

        self.logger.info("正在停止任务调度器...")

        # 停止所有工作线程池
        for step_id, executor in self.worker_pools.items():
            self.logger.debug(f"正在停止步骤 {step_id} 的工作线程...")
            executor.shutdown(wait=False)

        # 等待所有线程完成
        start_time = time.time()
        for step_id, executor in self.worker_pools.items():
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                self.logger.warning("停止超时，强制关闭线程池")
                break

            executor.shutdown(wait=True)
            if not executor._threads:  # 所有线程已停止
                self.logger.debug(f"步骤 {step_id} 的工作线程已停止")

        self.worker_pools.clear()
        self.logger.info("任务调度器已停止")

    def _worker_loop(self, step_id: int) -> None:
        """
        工作线程循环（简化版，移除资源管理，实现真正的异步并发）

        Args:
            step_id: 步骤ID
        """
        queue = self.queue_manager.get_queue(
            f"step_{step_id}_{self.STEP_DEFINITIONS[step_id][1]}"
        )
        processor = self.processors.get(step_id)

        if not queue or not processor:
            self.logger.error(f"步骤 {step_id} 的队列或处理器未找到")
            return

        thread_name = threading.current_thread().name
        self.logger.debug(f"工作线程 {thread_name} 开始处理步骤 {step_id}")

        while not self._shutdown_event.is_set():
            try:
                # 从队列获取任务（带超时）
                task = queue.get(block=True, timeout=0.1)
                if task is None:
                    continue

                self.logger.debug(f"工作线程 {thread_name} 获取到任务 {task.task_id}")

                # 直接处理任务（不再等待资源，实现真正的异步并发）
                result = processor.process_task(task)

                # 标记任务完成
                queue.task_done(task, result.success, result.message)

                # 更新任务跟踪
                self._update_task_tracking(task, result)

            except Exception as e:
                self.logger.error(
                    f"工作线程 {thread_name} 处理任务时出错: {e}", exc_info=True
                )
                continue

        self.logger.debug(f"工作线程 {thread_name} 已退出")

    def _update_task_tracking(self, task: Task, result: ProcessResult) -> None:
        """更新任务跟踪状态"""
        with self._lock:
            if result.success:
                # 检查是否完成所有步骤
                # 注意：task.current_step 现在指向下一个要处理的步骤
                # 所以当 current_step >= 8 时，说明所有步骤都已完成
                if task.current_step >= len(self.STEP_DEFINITIONS):  # 8个步骤 (0-7)
                    # 任务完成，更新状态
                    task.status = TaskStatus.COMPLETED
                    task.progress = 1.0
                    self._active_tasks.pop(task.task_id, None)
                    self._completed_tasks[task.task_id] = task
                    self.logger.info(f"任务 {task.task_id} 已完成所有步骤")
            else:
                # 检查是否需要标记为失败
                if not task.can_retry():
                    task.status = TaskStatus.FAILED
                    self._active_tasks.pop(task.task_id, None)
                    self._failed_tasks[task.task_id] = task
                    self.logger.error(f"任务 {task.task_id} 处理失败")

    def submit_task(self, task: Task) -> bool:
        """
        提交新任务到调度器

        Args:
            task: 要提交的任务

        Returns:
            是否成功提交
        """
        if not self._running:
            self.logger.error("调度器未运行，无法提交任务")
            return False

        try:
            # 确定任务应该从哪个步骤开始
            start_step = self._determine_start_step(task)

            # 检查任务是否已完成
            if start_step == -1:
                # 任务已完成，更新状态并将其标记为完成状态
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
                with self._lock:
                    self._active_tasks.pop(task.task_id, None)  # 从活动任务中移除
                    self._completed_tasks[task.task_id] = task  # 添加到完成任务
                return True  # 返回成功，因为任务确实已完成

            # 添加到活动任务跟踪
            with self._lock:
                self._active_tasks[task.task_id] = task

            self.logger.info(f"任务 {task.task_id} 将从步骤 {start_step} 开始处理")

            # 提交到确定的起始步骤
            return self.submit_task_to_step(task, start_step)

        except Exception as e:
            self.logger.error(f"提交任务 {task.task_id} 失败: {e}")
            return False

    def _determine_start_step(self, task: Task) -> int:
        """
        确定任务应该从哪个步骤开始处理

        Args:
            task: 任务对象

        Returns:
            起始步骤ID
        """
        try:
            # 检查已完成的步骤
            max_completed_step = -1

            for step_id in range(len(self.STEP_DEFINITIONS)):
                # 检查步骤结果
                if step_id in task.step_results:
                    result = task.step_results[step_id]
                    if result.success and not result.partial_success:
                        max_completed_step = step_id
                        continue

                # 检查步骤详情状态
                if step_id in task.step_details:
                    detail = task.step_details[step_id]
                    if detail.status == StepStatus.COMPLETED:
                        max_completed_step = step_id
                        continue

                # 如果步骤未完成或失败，从这里开始
                break

            # 从下一个未完成的步骤开始
            start_step = max_completed_step + 1

            # 检查任务是否已经完全完成
            if start_step >= len(self.STEP_DEFINITIONS):
                self.logger.info(
                    f"任务 {task.task_id} 已经完全完成所有 {len(self.STEP_DEFINITIONS)} 个步骤"
                )
                return -1  # 返回-1表示任务已完成，无需处理

            return start_step

        except Exception as e:
            self.logger.error(f"确定起始步骤失败: {e}")
            # 默认从第一步开始
            return 0

    def submit_task_to_step(self, task: Task, step_id: int) -> bool:
        """
        将任务提交到指定步骤

        Args:
            task: 任务对象
            step_id: 步骤ID

        Returns:
            是否成功提交
        """
        if step_id >= len(self.STEP_DEFINITIONS):
            self.logger.error(f"无效的步骤ID: {step_id}")
            return False

        step_name = self.STEP_DEFINITIONS[step_id][1]
        queue = self.queue_manager.get_queue(f"step_{step_id}_{step_name}")

        if not queue:
            self.logger.error(f"步骤 {step_id} 的队列未找到")
            return False

        # 更新任务的当前步骤
        task.current_step = step_id

        # 提交到队列（改为阻塞模式，确保任务能够成功提交）
        success = queue.put(task, block=True, timeout=5.0)  # 5秒超时

        if success:
            self.logger.debug(
                f"任务 {task.task_id} 已提交到步骤 {step_id} ({step_name})"
            )
        else:
            self.logger.error(f"任务 {task.task_id} 提交到步骤 {step_id} 失败")

        return success

    def submit_batch_tasks(self, tasks: List[Task]) -> List[bool]:
        """
        批量提交任务

        Args:
            tasks: 任务列表

        Returns:
            每个任务的提交结果
        """
        results = []
        self.logger.info(f"开始批量提交 {len(tasks)} 个任务到调度器")

        for i, task in enumerate(tasks):
            result = self.submit_task(task)
            results.append(result)
            if not result:
                self.logger.error(f"任务 {task.task_id} (第{i+1}个) 提交失败")
            else:
                self.logger.debug(f"任务 {task.task_id} (第{i+1}个) 提交成功")

        success_count = sum(results)
        failed_count = len(results) - success_count
        self.logger.info(
            f"批量提交任务完成: {success_count}/{len(tasks)} 成功, {failed_count} 失败"
        )

        return results

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        # 在活动任务中查找
        with self._lock:
            task = (
                self._active_tasks.get(task_id)
                or self._completed_tasks.get(task_id)
                or self._failed_tasks.get(task_id)
            )

        if task:
            return task.get_summary()
        return None

    def get_all_tasks_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有任务状态"""
        with self._lock:
            return {
                "active": {
                    tid: task.get_summary() for tid, task in self._active_tasks.items()
                },
                "completed": {
                    tid: task.get_summary()
                    for tid, task in self._completed_tasks.items()
                },
                "failed": {
                    tid: task.get_summary() for tid, task in self._failed_tasks.items()
                },
            }

    def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        with self._lock:
            active_count = len(self._active_tasks)
            completed_count = len(self._completed_tasks)
            failed_count = len(self._failed_tasks)

        # 获取队列统计
        queue_stats = self.queue_manager.get_stats()

        # 获取处理器统计
        processor_stats = {
            step_id: processor.get_stats()
            for step_id, processor in self.processors.items()
        }

        # 获取监听器统计
        listener_stats = {}
        for listener in self.master_listener.get_listeners():
            if hasattr(listener, "get_stats"):
                listener_stats[listener.name] = listener.get_stats()

        return {
            "scheduler_status": {
                "running": self._running,
                "active_tasks": active_count,
                "completed_tasks": completed_count,
                "failed_tasks": failed_count,
                "total_tasks": active_count + completed_count + failed_count,
                "success_rate": (
                    f"{completed_count / (completed_count + failed_count) * 100:.1f}%"
                    if (completed_count + failed_count) > 0
                    else "N/A"
                ),
            },
            "queue_stats": queue_stats,
            "processor_stats": processor_stats,
            "listener_stats": listener_stats,
        }

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        等待所有任务完成

        Args:
            timeout: 超时时间（秒）

        Returns:
            是否所有任务都完成
        """
        start_time = time.time()

        while True:
            with self._lock:
                if not self._active_tasks:
                    return True

            if timeout and (time.time() - start_time) > timeout:
                return False

            time.sleep(0.1)

    def cancel_task(self, task_id: str) -> bool:
        """
        取消指定任务

        Args:
            task_id: 任务ID

        Returns:
            是否成功取消
        """
        with self._lock:
            task = self._active_tasks.get(task_id)
            if task:
                task.update_status(TaskStatus.CANCELLED, "任务已被取消")
                self._active_tasks.pop(task_id, None)
                self.logger.info(f"任务 {task_id} 已取消")
                return True

        return False

    def cancel_all_tasks(self) -> int:
        """
        取消所有活动任务

        Returns:
            取消的任务数量
        """
        with self._lock:
            cancelled_count = 0
            for task in self._active_tasks.values():
                task.update_status(TaskStatus.CANCELLED, "批量取消任务")
                cancelled_count += 1

            self._active_tasks.clear()

        self.logger.info(f"已取消 {cancelled_count} 个任务")
        return cancelled_count

    def clear_completed_tasks(self) -> int:
        """
        清理已完成的任务记录

        Returns:
            清理的任务数量
        """
        with self._lock:
            count = len(self._completed_tasks) + len(self._failed_tasks)
            self._completed_tasks.clear()
            self._failed_tasks.clear()

        self.logger.info(f"已清理 {count} 个已完成的任务记录")
        return count

    def add_listener(self, listener: TaskListener) -> None:
        """添加任务监听器"""
        self.master_listener.add_listener(listener)
        self.logger.info(f"已添加监听器: {listener.name}")

    def remove_listener(self, listener: TaskListener) -> None:
        """移除任务监听器"""
        self.master_listener.remove_listener(listener)
        self.logger.info(f"已移除监听器: {listener.name}")

    def is_running(self) -> bool:
        """检查调度器是否在运行"""
        return self._running

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
