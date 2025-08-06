"""
抽象步骤处理器基类

定义了处理步骤的通用接口和行为，所有具体步骤处理器都继承此类
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from .task import ProcessResult, StepProgressDetail, StepStatus, Task, TaskStatus


class StepProcessor(ABC):
    """抽象步骤处理器基类"""

    def __init__(
        self,
        step_id: int,
        step_name: str,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ):
        """
        初始化步骤处理器

        Args:
            step_id: 步骤ID (0-7)
            step_name: 步骤名称
            timeout: 处理超时时间（秒）
            max_retries: 最大重试次数
        """
        self.step_id = step_id
        self.step_name = step_name
        self.timeout = timeout
        self.max_retries = max_retries

        self.logger = logging.getLogger(f"{__name__}.{step_name}")

        # 统计信息
        self._total_processed = 0
        self._total_success = 0
        self._total_failed = 0
        self._total_retries = 0
        self._processing_times = []

    def process_task(self, task: Task) -> ProcessResult:
        """
        处理任务的主入口方法

        Args:
            task: 要处理的任务

        Returns:
            处理结果
        """
        start_time = time.time()
        self.logger.info(f"准备处理任务 {task.task_id} - 步骤: {self.step_name}")

        try:
            # 验证任务是否可以处理
            validation_result = self._validate_task(task)
            if not validation_result.success:
                self._update_stats(False)
                return validation_result

            # 初始化步骤详细信息
            step_detail = task.init_step_detail(self.step_id, self.step_name)

            # 检查步骤状态
            step_status = task.get_step_status(self.step_id)

            if step_status == StepStatus.COMPLETED:
                # 步骤已完成，跳过处理
                self.logger.info(
                    f"任务 {task.task_id} 步骤 {self.step_name} 已完成，跳过处理"
                )
                result = task.get_step_result(self.step_id)
                if result:
                    return result
            elif step_status == StepStatus.PROCESSING:
                # 步骤之前中断了，需要从中断处继续
                self.logger.info(
                    f"任务 {task.task_id} 步骤 {self.step_name} 从中断处继续执行"
                )
                result = self._resume_from_interruption_with_status_update(task, step_detail)
            else:
                # 步骤未进行，从头开始
                self.logger.info(f"任务 {task.task_id} 步骤 {self.step_name} 开始执行")
                result = self._execute_process_with_status_update(task, step_detail)

            # 记录处理时间
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.step_detail = step_detail

            # 更新统计信息
            self._update_stats(result.success, processing_time)

            # 根据结果更新步骤状态
            if result.success:
                step_detail.status = StepStatus.COMPLETED
                # 发送完成状态
                self._notify_status_change(
                    task, "completed", f"步骤 {self.step_name} 处理成功"
                )
                self.logger.info(
                    f"任务 {task.task_id} 步骤 {self.step_name} 处理成功 "
                    f"(耗时: {processing_time:.2f}s)"
                )
            elif getattr(result, "partial_success", False):
                # 部分成功，保持当前状态（可能是PROCESSING）
                self.logger.warning(
                    f"任务 {task.task_id} 步骤 {self.step_name} 部分成功 "
                    f"(耗时: {processing_time:.2f}s)"
                )
            else:
                step_detail.status = StepStatus.FAILED
                # 发送失败状态
                self._notify_status_change(
                    task, "failed", f"步骤 {self.step_name} 处理失败: {result.error}"
                )
                self.logger.error(
                    f"任务 {task.task_id} 步骤 {self.step_name} 处理失败: {result.error}"
                )

            # 设置任务步骤结果
            task.set_step_result(self.step_id, result)

            # 保存缓存（如果任务有路径信息）
            if task.paths and "pipeline_cache" in task.paths:
                cache_path = Path(task.paths["pipeline_cache"])
                if task.save_to_cache(cache_path):
                    self.logger.debug(f"任务 {task.task_id} 缓存已更新")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"步骤 {self.step_name} 处理异常: {str(e)}"

            self.logger.error(f"任务 {task.task_id} - {error_msg}", exc_info=True)

            result = ProcessResult(
                success=False,
                message=error_msg,
                error=str(e),
                processing_time=processing_time,
            )

            # 更新步骤状态为失败
            if self.step_id in task.step_details:
                task.step_details[self.step_id].status = StepStatus.FAILED

            self._update_stats(False, processing_time)
            return result

    @abstractmethod
    def _execute_process(
        self, task: Task, step_detail: StepProgressDetail
    ) -> ProcessResult:
        """
        执行具体的处理逻辑（子类必须实现）

        Args:
            task: 要处理的任务
            step_detail: 步骤详细信息

        Returns:
            处理结果
        """
        pass

    def _execute_process_with_status_update(
        self, task: Task, step_detail: StepProgressDetail
    ) -> ProcessResult:
        """
        执行处理逻辑并更新状态（包装方法）
        
        Args:
            task: 要处理的任务
            step_detail: 步骤详细信息
            
        Returns:
            处理结果
        """
        # 只有在真正开始执行处理逻辑时才设置PROCESSING状态和发送通知
        self.logger.info(f"任务 {task.task_id} 步骤 {self.step_name} 真正开始处理")
        task.update_status(TaskStatus.PROCESSING, f"正在执行步骤: {self.step_name}")
        step_detail.status = StepStatus.PROCESSING
        self._notify_status_change(
            task, "processing", f"开始执行步骤: {self.step_name}"
        )
        
        # 调用具体的处理逻辑
        return self._execute_process(task, step_detail)
        
    def _resume_from_interruption_with_status_update(
        self, task: Task, step_detail: StepProgressDetail
    ) -> ProcessResult:
        """
        从中断处恢复执行并更新状态（包装方法）
        
        Args:
            task: 要处理的任务
            step_detail: 步骤详细信息
            
        Returns:
            处理结果
        """
        # 只有在真正开始执行恢复逻辑时才设置PROCESSING状态和发送通知
        self.logger.info(f"任务 {task.task_id} 步骤 {self.step_name} 真正开始从中断处恢复")
        task.update_status(TaskStatus.PROCESSING, f"正在执行步骤: {self.step_name}")
        step_detail.status = StepStatus.PROCESSING
        self._notify_status_change(task, "processing", "从中断处继续执行")
        
        # 调用具体的恢复逻辑
        return self._resume_from_interruption(task, step_detail)

    def _resume_from_interruption(
        self, task: Task, step_detail: StepProgressDetail
    ) -> ProcessResult:
        """
        从中断处恢复执行（默认实现，子类可以重写）

        Args:
            task: 要处理的任务
            step_detail: 步骤详细信息

        Returns:
            处理结果
        """
        # 默认行为：重新执行整个步骤
        self.logger.info(f"步骤 {self.step_name} 使用默认恢复策略：重新执行")
        step_detail.status = StepStatus.PROCESSING
        return self._execute_process(task, step_detail)

    def _validate_task(self, task: Task) -> ProcessResult:
        """
        验证任务是否可以处理

        Args:
            task: 要验证的任务

        Returns:
            验证结果
        """
        # 检查任务状态
        if task.status == TaskStatus.CANCELLED:
            return ProcessResult(
                success=False, message="任务已取消", error="任务已取消，无法继续处理"
            )

        # 检查步骤顺序
        if task.current_step != self.step_id:
            return ProcessResult(
                success=False,
                message=f"步骤顺序错误: 期望步骤 {self.step_id}，当前步骤 {task.current_step}",
                error=f"任务 {task.task_id} 步骤顺序不正确",
            )

        # 检查依赖步骤是否完成
        for dep_step in range(self.step_id):
            if not task.is_step_completed(dep_step):
                return ProcessResult(
                    success=False,
                    message=f"依赖步骤 {dep_step} 未完成",
                    error=f"步骤 {dep_step} 必须在步骤 {self.step_id} 之前完成",
                )

        return ProcessResult(success=True, message="任务验证通过")

    def _update_stats(self, success: bool, processing_time: float = 0.0) -> None:
        """更新统计信息"""
        self._total_processed += 1

        if success:
            self._total_success += 1
        else:
            self._total_failed += 1

        if processing_time > 0:
            self._processing_times.append(processing_time)
            # 只保留最近1000条记录
            if len(self._processing_times) > 1000:
                self._processing_times = self._processing_times[-1000:]

    def _notify_status_change(self, task: Task, status: str, message: str = "") -> None:
        """通知流水线状态变化"""
        try:
            if hasattr(task, "pipeline_ref") and task.pipeline_ref:
                task.pipeline_ref.notify_step_status(
                    task.task_id, self.step_id, status, message
                )
        except Exception as e:
            self.logger.debug(
                f"通知状态变化失败: {e}"
            )  # 使用debug级别，避免干扰主要日志

    def get_next_step_id(self) -> Optional[int]:
        """获取下一个步骤ID"""
        next_id = self.step_id + 1
        return next_id if next_id < 8 else None  # 总共8个步骤 (0-7)

    def should_retry(self, task: Task, result: ProcessResult) -> bool:
        """
        判断是否应该重试

        Args:
            task: 任务对象
            result: 处理结果

        Returns:
            是否应该重试
        """
        if result.success:
            return False

        if not task.can_retry():
            return False

        # 可以根据错误类型决定是否重试
        # 例如：网络错误可以重试，文件不存在错误不重试
        return True

    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        success_rate = (
            (self._total_success / self._total_processed * 100)
            if self._total_processed > 0
            else 0.0
        )

        avg_processing_time = (
            sum(self._processing_times) / len(self._processing_times)
            if self._processing_times
            else 0.0
        )

        max_processing_time = (
            max(self._processing_times) if self._processing_times else 0.0
        )
        min_processing_time = (
            min(self._processing_times) if self._processing_times else 0.0
        )

        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "total_processed": self._total_processed,
            "total_success": self._total_success,
            "total_failed": self._total_failed,
            "total_retries": self._total_retries,
            "success_rate": f"{success_rate:.2f}%",
            "average_processing_time": f"{avg_processing_time:.2f}s",
            "max_processing_time": f"{max_processing_time:.2f}s",
            "min_processing_time": f"{min_processing_time:.2f}s",
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._total_processed = 0
        self._total_success = 0
        self._total_failed = 0
        self._total_retries = 0
        self._processing_times = []

        self.logger.info(f"步骤 {self.step_name} 统计信息已重置")

    def can_process(self, task: Task) -> bool:
        """
        检查是否可以处理指定任务

        Args:
            task: 要检查的任务

        Returns:
            是否可以处理
        """
        validation_result = self._validate_task(task)
        return validation_result.success

    def estimate_processing_time(self) -> float:
        """
        估算处理时间

        Returns:
            预估处理时间（秒）
        """
        if not self._processing_times:
            return 60.0  # 默认估算1分钟

        # 返回平均处理时间
        return sum(self._processing_times) / len(self._processing_times)

    def __str__(self) -> str:
        """字符串表示"""
        return f"StepProcessor(id={self.step_id}, name={self.step_name})"

    def __repr__(self) -> str:
        """调试表示"""
        return self.__str__()
