"""
缓存管理模块

提供统一的任务缓存功能，支持细粒度的状态保存和恢复
"""

from .cache_types import CacheMetadata, CacheVersion
from .task_cache import TaskCacheManager

__all__ = [
    "TaskCacheManager",
    "CacheVersion",
    "CacheMetadata",
]
