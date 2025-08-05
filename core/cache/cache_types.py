"""
缓存类型定义

定义缓存相关的数据结构和类型
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class CacheVersion(Enum):
    """缓存版本枚举"""
    
    V1_0 = "1.0"  # 旧版本缓存格式
    V2_0 = "2.0"  # 新版本缓存格式（支持细粒度状态）
    CURRENT = V2_0  # 当前使用的版本


@dataclass
class CacheMetadata:
    """缓存元数据"""
    
    version: str = CacheVersion.CURRENT.value
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    saved_at: str = field(default_factory=lambda: datetime.now().isoformat())
    task_id: Optional[str] = None
    video_path: Optional[str] = None
    total_steps: int = 8
    completed_steps: int = 0
    
    # 额外的元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_save_time(self):
        """更新保存时间"""
        self.saved_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "saved_at": self.saved_at,
            "task_id": self.task_id,
            "video_path": self.video_path,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheMetadata":
        """从字典创建对象"""
        return cls(
            version=data.get("version", CacheVersion.V1_0.value),
            created_at=data.get("created_at", datetime.now().isoformat()),
            saved_at=data.get("saved_at", datetime.now().isoformat()),
            task_id=data.get("task_id"),
            video_path=data.get("video_path"),
            total_steps=data.get("total_steps", 8),
            completed_steps=data.get("completed_steps", 0),
            metadata=data.get("metadata", {}),
        )