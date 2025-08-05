"""
任务实体和状态定义模块

定义了任务的数据结构、状态枚举和处理结果类型
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"           # 待处理
    PROCESSING = "processing"     # 处理中
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"            # 处理失败
    CANCELLED = "cancelled"       # 已取消
    RETRYING = "retrying"        # 重试中


class ResourceType(Enum):
    """资源类型枚举"""
    GPU_INTENSIVE = "gpu_intensive"     # GPU密集型
    CPU_INTENSIVE = "cpu_intensive"     # CPU密集型
    IO_INTENSIVE = "io_intensive"       # I/O密集型


@dataclass
class ProcessResult:
    """处理结果"""
    success: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0


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
        self.progress = ((completed_steps + current_step_progress) / total_steps) * 100.0
        self.updated_at = datetime.now()
    
    def set_step_result(self, step: int, result: ProcessResult) -> None:
        """设置步骤处理结果"""
        self.step_results[step] = result
        self.updated_at = datetime.now()
        
        if result.success:
            self.update_progress(step + 1)  # 进入下一步骤
        else:
            self.status = TaskStatus.FAILED
            self.error_message = result.error or result.message
    
    def get_step_result(self, step: int) -> Optional[ProcessResult]:
        """获取步骤处理结果"""
        return self.step_results.get(step)
    
    def is_step_completed(self, step: int) -> bool:
        """检查指定步骤是否已完成"""
        result = self.step_results.get(step)
        return result is not None and result.success
    
    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """增加重试次数"""
        self.retry_count += 1
        self.status = TaskStatus.RETRYING
        self.updated_at = datetime.now()
    
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
                }
                for k, v in self.step_results.items()
            },
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "paths": self.paths,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """从字典创建任务对象"""
        # 解析步骤结果
        step_results = {}
        for k, v in data.get("step_results", {}).items():
            step_results[int(k)] = ProcessResult(
                success=v["success"],
                message=v["message"],
                data=v["data"],
                error=v["error"],
                processing_time=v["processing_time"],
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
            error_message=data.get("error_message"),
            retry_count=data["retry_count"],
            max_retries=data["max_retries"],
            created_at=created_at,
            updated_at=updated_at,
            started_at=started_at,
            completed_at=completed_at,
            paths=data.get("paths"),
        )