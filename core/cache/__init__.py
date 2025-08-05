"""
缓存管理模块

提供统一的任务缓存功能，支持细粒度的状态保存和恢复
"""

from .task_cache import TaskCacheManager
from .cache_types import CacheVersion, CacheMetadata

__all__ = [
    "TaskCacheManager",
    "CacheVersion",
    "CacheMetadata",
]