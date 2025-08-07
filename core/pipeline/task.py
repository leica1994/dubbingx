"""
任务实体和状态定义模块

定义了任务的数据结构、状态枚举和处理结果类型
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    """任务状态枚举"""

    PENDING = "pending"  # 待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 处理失败
    CANCELLED = "cancelled"  # 已取消
    RETRYING = "retrying"  # 重试中


class StepStatus(Enum):
    """步骤状态枚举"""

    PENDING = "pending"  # 未进行
    PROCESSING = "processing"  # 正在处理
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 处理失败


class ResourceType(Enum):
    """资源类型枚举"""

    GPU_INTENSIVE = "gpu_intensive"  # GPU密集型
    CPU_INTENSIVE = "cpu_intensive"  # CPU密集型
    IO_INTENSIVE = "io_intensive"  # I/O密集型


@dataclass
class StepProgressDetail:
    """步骤内部进度详情"""

    # 基础信息
    step_id: int
    step_name: str
    status: StepStatus = StepStatus.PENDING

    # 进度信息
    current_item: int = 0  # 当前处理项目（如第几行字幕）
    total_items: int = 0  # 总项目数
    progress_percent: float = 0.0  # 步骤内进度百分比

    # 详细状态数据
    completed_items: List[Dict[str, Any]] = field(default_factory=list)  # 已完成的项目
    pending_items: List[Dict[str, Any]] = field(default_factory=list)  # 待处理的项目
    failed_items: List[Dict[str, Any]] = field(default_factory=list)  # 失败的项目

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外的步骤特定数据

    def update_progress(
        self, current: int, total: int, item_data: Optional[Dict[str, Any]] = None
    ):
        """更新进度"""
        self.current_item = current
        self.total_items = total
        self.progress_percent = (current / total * 100.0) if total > 0 else 0.0

        if item_data:
            # 更新当前处理项目的数据
            self.metadata["current_item_data"] = item_data

    def add_completed_item(self, item_data: Dict[str, Any]):
        """添加已完成项目"""
        self.completed_items.append(
            {**item_data, "completed_at": datetime.now().isoformat()}
        )
        self.current_item = len(self.completed_items)
        if self.total_items > 0:
            self.progress_percent = self.current_item / self.total_items * 100.0

    def add_failed_item(self, item_data: Dict[str, Any], error: str):
        """添加失败项目"""
        self.failed_items.append(
            {**item_data, "error": error, "failed_at": datetime.now().isoformat()}
        )

    def is_completed(self) -> bool:
        """检查是否完全完成"""
        return (
            self.status == StepStatus.COMPLETED
            and self.current_item >= self.total_items
            and len(self.failed_items) == 0
        )

    def is_partially_completed(self) -> bool:
        """检查是否部分完成"""
        return len(self.completed_items) > 0 and not self.is_completed()

    def get_resume_data(self) -> Dict[str, Any]:
        """获取恢复所需的数据"""
        return {
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "current_item": self.current_item,
            "total_items": self.total_items,
            "metadata": self.metadata,
        }


@dataclass
class ProcessResult:
    """处理结果"""

    success: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0

    # 新增：细粒度状态支持
    step_detail: Optional[StepProgressDetail] = None  # 步骤详细信息
    partial_success: bool = False  # 是否部分成功（用于中断恢复）


@dataclass
class Task:
    """任务实体"""

    # 基本信息
    task_id: str
    video_path: str
    subtitle_path: Optional[str] = None

    # 任务状态
    current_step: int = 0  # 当前处理步骤 (0-7)
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 进度百分比 (0-100)

    # 处理结果
    step_results: Dict[int, ProcessResult] = field(default_factory=dict)
    step_details: Dict[int, StepProgressDetail] = field(
        default_factory=dict
    )  # 新增：步骤详细信息
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 文件路径信息
    paths: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """初始化后处理"""
        if not self.task_id:
            # 基于视频文件名和时间戳生成任务ID
            video_name = Path(self.video_path).stem
            timestamp = int(time.time() * 1000)
            self.task_id = f"{video_name}_{timestamp}"

    def update_status(self, status: TaskStatus, message: str = "") -> None:
        """更新任务状态"""
        self.status = status
        self.updated_at = datetime.now()

        if status == TaskStatus.PROCESSING and self.started_at is None:
            self.started_at = datetime.now()
        elif status == TaskStatus.COMPLETED:
            self.completed_at = datetime.now()
            self.progress = 100.0
        elif status == TaskStatus.FAILED:
            self.error_message = message

    def update_progress(self, step: int, step_progress: float = 100.0) -> None:
        """更新任务进度"""
        self.current_step = step
        # 计算总体进度: (已完成步骤数 + 当前步骤进度/100) / 总步骤数 * 100
        total_steps = 8
        completed_steps = step
        current_step_progress = step_progress / 100.0
        self.progress = (
            (completed_steps + current_step_progress) / total_steps
        ) * 100.0
        self.updated_at = datetime.now()

    def set_step_result(self, step: int, result: ProcessResult) -> None:
        """设置步骤处理结果"""
        # 修复数据类型不匹配问题：使用字符串key匹配存储格式
        step_key = str(step)
        self.step_results[step_key] = result
        self.updated_at = datetime.now()

        if result.success:
            # 注意：不再自动更新步骤，由 TaskFlowListener 负责步骤推进
            # 只更新处理时间统计
            pass
        else:
            self.status = TaskStatus.FAILED
            self.error_message = result.error or result.message

    def get_step_result(self, step: int) -> Optional[ProcessResult]:
        """获取步骤处理结果"""
        # 修复数据类型不匹配问题：使用字符串key匹配存储格式
        step_key = str(step)
        return self.step_results.get(step_key)

    def is_step_completed(self, step: int) -> bool:
        """检查指定步骤是否已完成"""
        # 修复数据类型不匹配问题：使用字符串key匹配存储格式
        step_key = str(step)
        result = self.step_results.get(step_key)
        if result is None:
            return False
        
        # 检查结果格式（新的字典格式）
        if isinstance(result, dict):
            success = result.get("success", False)
            # 修复逻辑错误：如果success为True，就是完成的，不管partial_success的值
            return success
        else:
            # 兼容旧格式（ProcessResult对象）
            return result.success

    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> None:
        """增加重试次数"""
        self.retry_count += 1
        self.status = TaskStatus.RETRYING
        self.updated_at = datetime.now()

    def init_step_detail(
        self, step_id: int, step_name: str, total_items: int = 0
    ) -> StepProgressDetail:
        """初始化步骤详细信息"""
        if step_id not in self.step_details:
            self.step_details[step_id] = StepProgressDetail(
                step_id=step_id, step_name=step_name, total_items=total_items
            )
        return self.step_details[step_id]

    def get_step_detail(self, step_id: int) -> Optional[StepProgressDetail]:
        """获取步骤详细信息"""
        return self.step_details.get(step_id)

    def update_step_detail(self, step_id: int, **kwargs) -> None:
        """更新步骤详细信息"""
        if step_id in self.step_details:
            detail = self.step_details[step_id]
            for key, value in kwargs.items():
                if hasattr(detail, key):
                    setattr(detail, key, value)
            self.updated_at = datetime.now()

    def add_completed_item(self, step_id: int, item_data: Dict[str, Any]) -> None:
        """为指定步骤添加已完成项目"""
        if step_id in self.step_details:
            self.step_details[step_id].add_completed_item(item_data)
            self.updated_at = datetime.now()

    def add_failed_item(
        self, step_id: int, item_data: Dict[str, Any], error: str
    ) -> None:
        """为指定步骤添加失败项目"""
        if step_id in self.step_details:
            self.step_details[step_id].add_failed_item(item_data, error)
            self.updated_at = datetime.now()

    def get_step_status(self, step_id: int) -> StepStatus:
        """获取步骤状态"""
        if step_id in self.step_details:
            return self.step_details[step_id].status
        elif self.is_step_completed(step_id):
            return StepStatus.COMPLETED
        else:
            return StepStatus.PENDING

    def is_step_partially_completed(self, step_id: int) -> bool:
        """检查步骤是否部分完成"""
        if step_id in self.step_details:
            return self.step_details[step_id].is_partially_completed()
        return False

    def get_step_resume_data(self, step_id: int) -> Optional[Dict[str, Any]]:
        """获取步骤恢复数据"""
        if step_id in self.step_details:
            return self.step_details[step_id].get_resume_data()
        return None

    def get_processing_time(self) -> float:
        """获取总处理时间（秒）"""
        if self.started_at is None:
            return 0.0

        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    def get_summary(self) -> Dict[str, Any]:
        """获取任务摘要信息"""
        return {
            "task_id": self.task_id,
            "video_path": self.video_path,
            "subtitle_path": self.subtitle_path,
            "status": self.status.value,
            "current_step": self.current_step,
            "progress": round(self.progress, 2),
            "retry_count": self.retry_count,
            "processing_time": round(self.get_processing_time(), 2),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error_message": self.error_message,
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "video_path": self.video_path,
            "subtitle_path": self.subtitle_path,
            "current_step": self.current_step,
            "status": self.status.value,
            "progress": self.progress,
            "step_results": {
                str(k): {
                    "success": v.success,
                    "message": v.message,
                    "data": v.data,
                    "error": v.error,
                    "processing_time": v.processing_time,
                    "partial_success": getattr(v, "partial_success", False),
                    "step_detail": (
                        {
                            "step_id": v.step_detail.step_id,
                            "step_name": v.step_detail.step_name,
                            "status": v.step_detail.status.value,
                            "current_item": v.step_detail.current_item,
                            "total_items": v.step_detail.total_items,
                            "progress_percent": v.step_detail.progress_percent,
                            "completed_items": v.step_detail.completed_items,
                            "pending_items": v.step_detail.pending_items,
                            "failed_items": v.step_detail.failed_items,
                            "metadata": v.step_detail.metadata,
                        }
                        if v.step_detail
                        else None
                    ),
                }
                for k, v in self.step_results.items()
            },
            "step_details": {
                str(k): {
                    "step_id": v.step_id,
                    "step_name": v.step_name,
                    "status": v.status.value,
                    "current_item": v.current_item,
                    "total_items": v.total_items,
                    "progress_percent": v.progress_percent,
                    "completed_items": v.completed_items,
                    "pending_items": v.pending_items,
                    "failed_items": v.failed_items,
                    "metadata": v.metadata,
                }
                for k, v in self.step_details.items()
            },
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "paths": self.paths,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """从字典创建任务对象"""
        # 解析步骤结果
        step_results = {}
        for k, v in data.get("step_results", {}).items():
            # 解析步骤详细信息
            step_detail = None
            if v.get("step_detail"):
                detail_data = v["step_detail"]
                step_detail = StepProgressDetail(
                    step_id=detail_data["step_id"],
                    step_name=detail_data["step_name"],
                    status=StepStatus(detail_data["status"]),
                    current_item=detail_data["current_item"],
                    total_items=detail_data["total_items"],
                    progress_percent=detail_data["progress_percent"],
                    completed_items=detail_data["completed_items"],
                    pending_items=detail_data["pending_items"],
                    failed_items=detail_data["failed_items"],
                    metadata=detail_data["metadata"],
                )

            step_results[int(k)] = ProcessResult(
                success=v["success"],
                message=v["message"],
                data=v["data"],
                error=v["error"],
                processing_time=v["processing_time"],
                step_detail=step_detail,
                partial_success=v.get("partial_success", False),
            )

        # 解析步骤详细信息
        step_details = {}
        for k, v in data.get("step_details", {}).items():
            step_details[int(k)] = StepProgressDetail(
                step_id=v["step_id"],
                step_name=v["step_name"],
                status=StepStatus(v["status"]),
                current_item=v["current_item"],
                total_items=v["total_items"],
                progress_percent=v["progress_percent"],
                completed_items=v["completed_items"],
                pending_items=v["pending_items"],
                failed_items=v["failed_items"],
                metadata=v["metadata"],
            )

        # 解析时间戳
        created_at = datetime.fromisoformat(data["created_at"])
        updated_at = datetime.fromisoformat(data["updated_at"])
        started_at = (
            datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None
        )
        completed_at = (
            datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None
        )

        return cls(
            task_id=data["task_id"],
            video_path=data["video_path"],
            subtitle_path=data.get("subtitle_path"),
            current_step=data["current_step"],
            status=TaskStatus(data["status"]),
            progress=data["progress"],
            step_results=step_results,
            step_details=step_details,
            error_message=data.get("error_message"),
            retry_count=data["retry_count"],
            max_retries=data["max_retries"],
            created_at=created_at,
            updated_at=updated_at,
            started_at=started_at,
            completed_at=completed_at,
            paths=data.get("paths"),
        )

    def save_to_cache(self, cache_file_path: Path) -> bool:
        """保存任务状态到缓存文件"""
        from ..cache import CacheMetadata, TaskCacheManager

        cache_manager = TaskCacheManager()
        metadata = CacheMetadata(
            task_id=self.task_id,
            video_path=self.video_path,
            completed_steps=self.current_step,
        )

        return cache_manager.save_task_cache(self, cache_file_path, metadata)

    @classmethod
    def load_from_cache(
        cls, cache_file_path: Path, video_path: str, subtitle_path: Optional[str] = None
    ) -> Optional["Task"]:
        """从缓存文件加载任务状态"""
        from ..cache import TaskCacheManager

        cache_manager = TaskCacheManager()
        cache_data = cache_manager.load_task_cache(
            cache_file_path, video_path, subtitle_path
        )

        if cache_data is None:
            return None

        # 从新版本缓存格式创建任务
        task_data = cache_data["task"]
        return cls.from_dict(task_data)
