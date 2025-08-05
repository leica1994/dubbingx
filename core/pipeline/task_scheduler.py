"""
主任务调度器模块

负责协调整个流水线处理系统，管理队列、工作线程和资源分配
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .resource_manager import ResourceManager, ResourceType
from .step_processor import StepProcessor
from .task import ProcessResult, Task, TaskStatus
from .task_listener import (
    CompositeTaskListener,
    LoggingTaskListener,
    StatisticsTaskListener,
    TaskFlowListener,
    TaskListener,
)
from .task_queue import QueueManager, TaskQueue


class TaskScheduler:
    """主任务调度器"""
    
    # 处理步骤定义
    STEP_DEFINITIONS = [
        (0, "preprocess_subtitle", ResourceType.CPU_INTENSIVE),    # CPU密集型
        (1, "separate_media", ResourceType.GPU_INTENSIVE),        # GPU密集型
        (2, "generate_reference_audio", ResourceType.CPU_INTENSIVE), # CPU密集型
        (3, "generate_tts", ResourceType.GPU_INTENSIVE),           # GPU密集型
        (4, "align_audio", ResourceType.CPU_INTENSIVE),           # CPU密集型
        (5, "generate_aligned_srt", ResourceType.IO_INTENSIVE),    # I/O密集型
        (6, "process_video_speed", ResourceType.IO_INTENSIVE),      # I/O密集型
        (7, "merge_audio_video", ResourceType.IO_INTENSIVE),      # I/O密集型
    ]
    
    def __init__(
        self,
        max_workers_per_step: Optional[Dict[ResourceType, int]] = None,
        resource_manager: Optional[ResourceManager] = None
    ):
        """
        初始化任务调度器
        
        Args:
            max_workers_per_step: 每种资源类型的最大工作线程数
            resource_manager: 资源管理器实例
        """
        self.logger = logging.getLogger(__name__)
        
        # 初始化资源管理器
        self.resource_manager = resource_manager or ResourceManager()
        
        # 工作线程数配置
        self.max_workers_per_step = max_workers_per_step or {
            ResourceType.GPU_INTENSIVE: 2,
            ResourceType.CPU_INTENSIVE: 4,
            ResourceType.IO_INTENSIVE: 8,
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
        for step_id, step_name, _ in self.STEP_DEFINITIONS:
            queue = self.queue_manager.create_queue(f"step_{step_id}_{step_name}")
            self.logger.debug(f"已创建队列: {queue.name}")
    
    def _initialize_listeners(self) -> None:
        """初始化监听器"""
        # 添加日志监听器
        log_listener = LoggingTaskListener(logging.INFO)
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
        
        # 启动所有步骤的工作线程池
        for step_id, step_name, resource_type in self.STEP_DEFINITIONS:
            if step_id not in self.processors:
                self.logger.error(f"步骤 {step_id} ({step_name}) 未注册处理器")
                continue
            
            max_workers = self.max_workers_per_step.get(resource_type, 1)
            executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=f"Step-{step_id}-{step_name}"
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
        工作线程循环
        
        Args:
            step_id: 步骤ID
        """
        queue = self.queue_manager.get_queue(f"step_{step_id}_{self.STEP_DEFINITIONS[step_id][1]}")
        processor = self.processors.get(step_id)
        
        if not queue or not processor:
            self.logger.error(f"步骤 {step_id} 的队列或处理器未找到")
            return
        
        thread_name = threading.current_thread().name
        self.logger.debug(f"工作线程 {thread_name} 开始处理步骤 {step_id}")
        
        while not self._shutdown_event.is_set():
            try:
                # 从队列获取任务（带超时）
                task = queue.get(block=True, timeout=1.0)
                if task is None:
                    continue
                
                self.logger.debug(f"工作线程 {thread_name} 获取到任务 {task.task_id}")
                
                # 获取资源并处理任务
                with self.resource_manager.acquire_resource(
                    processor.resource_type,
                    timeout=30.0,
                    task_id=task.task_id
                ) as acquired:
                    
                    if not acquired:
                        self.logger.warning(
                            f"任务 {task.task_id} 获取资源超时，重新入队"
                        )
                        # 重新放回队列
                        queue.put(task, block=False)
                        continue
                    
                    # 处理任务
                    result = processor.process_task(task)
                    
                    # 标记任务完成
                    queue.task_done(task, result.success, result.message)
                    
                    # 更新任务跟踪
                    self._update_task_tracking(task, result)
            
            except Exception as e:
                self.logger.error(f"工作线程 {thread_name} 处理任务时出错: {e}", exc_info=True)
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
                    # 任务完成
                    self._active_tasks.pop(task.task_id, None)
                    self._completed_tasks[task.task_id] = task
                    self.logger.info(f"任务 {task.task_id} 已完成所有步骤")
            else:
                # 检查是否需要标记为失败
                if not task.can_retry():
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
            # 添加到活动任务跟踪
            with self._lock:
                self._active_tasks[task.task_id] = task
            
            # 提交到第一个步骤
            return self.submit_task_to_step(task, 0)
            
        except Exception as e:
            self.logger.error(f"提交任务 {task.task_id} 失败: {e}")
            return False
    
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
        
        # 提交到队列
        success = queue.put(task, block=False)
        
        if success:
            self.logger.debug(f"任务 {task.task_id} 已提交到步骤 {step_id} ({step_name})")
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
        
        for task in tasks:
            result = self.submit_task(task)
            results.append(result)
        
        success_count = sum(results)
        self.logger.info(f"批量提交任务完成: {success_count}/{len(tasks)} 成功")
        
        return results
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        # 在活动任务中查找
        with self._lock:
            task = (
                self._active_tasks.get(task_id) or
                self._completed_tasks.get(task_id) or
                self._failed_tasks.get(task_id)
            )
        
        if task:
            return task.get_summary()
        return None
    
    def get_all_tasks_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有任务状态"""
        with self._lock:
            return {
                "active": {tid: task.get_summary() for tid, task in self._active_tasks.items()},
                "completed": {tid: task.get_summary() for tid, task in self._completed_tasks.items()},
                "failed": {tid: task.get_summary() for tid, task in self._failed_tasks.items()},
            }
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        with self._lock:
            active_count = len(self._active_tasks)
            completed_count = len(self._completed_tasks)
            failed_count = len(self._failed_tasks)
        
        # 获取队列统计
        queue_stats = self.queue_manager.get_stats()
        
        # 获取资源统计
        resource_stats = self.resource_manager.get_summary()
        
        # 获取处理器统计
        processor_stats = {
            step_id: processor.get_stats()
            for step_id, processor in self.processors.items()
        }
        
        # 获取监听器统计
        listener_stats = {}
        for listener in self.master_listener.get_listeners():
            if hasattr(listener, 'get_stats'):
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
            "resource_stats": resource_stats,
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