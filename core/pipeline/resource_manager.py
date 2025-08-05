"""
资源池管理器模块

管理GPU、CPU和I/O资源的并发访问，确保系统资源合理分配
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional, List

from .task import ResourceType


@dataclass
class ResourceStats:
    """资源统计信息"""
    resource_type: ResourceType
    max_count: int
    available_count: int
    in_use_count: int
    total_acquired: int
    total_released: int
    average_hold_time: float
    max_hold_time: float


@dataclass
class ResourceUsage:
    """资源使用记录"""
    resource_type: ResourceType
    thread_id: int
    acquired_at: float
    task_id: Optional[str] = None


class ResourceManager:
    """资源池管理器"""
    
    def __init__(
        self,
        gpu_count: int = 2,
        cpu_count: int = 4, 
        io_count: int = 8
    ):
        """
        初始化资源管理器
        
        Args:
            gpu_count: GPU资源数量
            cpu_count: CPU资源数量  
            io_count: I/O资源数量
        """
        self.logger = logging.getLogger(__name__)
        
        # 创建信号量
        self._semaphores = {
            ResourceType.GPU_INTENSIVE: threading.Semaphore(gpu_count),
            ResourceType.CPU_INTENSIVE: threading.Semaphore(cpu_count),
            ResourceType.IO_INTENSIVE: threading.Semaphore(io_count),
        }
        
        # 资源配置
        self._max_counts = {
            ResourceType.GPU_INTENSIVE: gpu_count,
            ResourceType.CPU_INTENSIVE: cpu_count,
            ResourceType.IO_INTENSIVE: io_count,
        }
        
        # 统计信息
        self._stats_lock = threading.Lock()
        self._total_acquired = {rt: 0 for rt in ResourceType}
        self._total_released = {rt: 0 for rt in ResourceType}
        self._hold_times = {rt: [] for rt in ResourceType}
        self._current_usage: Dict[int, ResourceUsage] = {}
        
        self.logger.info(f"资源管理器初始化完成: GPU={gpu_count}, CPU={cpu_count}, I/O={io_count}")
    
    @contextmanager
    def acquire_resource(
        self, 
        resource_type: ResourceType,
        timeout: Optional[float] = None,
        task_id: Optional[str] = None
    ) -> Generator[bool, None, None]:
        """
        获取资源上下文管理器
        
        Args:
            resource_type: 资源类型
            timeout: 超时时间（秒）
            task_id: 任务ID（用于统计）
            
        Yields:
            是否成功获取资源
        """
        semaphore = self._semaphores[resource_type]
        thread_id = threading.get_ident()
        acquired_at = time.time()
        acquired = False
        
        try:
            # 尝试获取资源
            acquired = semaphore.acquire(timeout=timeout)
            
            if acquired:
                # 记录使用情况
                with self._stats_lock:
                    self._total_acquired[resource_type] += 1
                    self._current_usage[thread_id] = ResourceUsage(
                        resource_type=resource_type,
                        thread_id=thread_id,
                        acquired_at=acquired_at,
                        task_id=task_id
                    )
                
                self.logger.debug(
                    f"资源获取成功: {resource_type.value} "
                    f"(任务: {task_id or 'unknown'}, 线程: {thread_id})"
                )
            else:
                self.logger.warning(
                    f"资源获取超时: {resource_type.value} "
                    f"(任务: {task_id or 'unknown'}, 超时: {timeout}s)"
                )
            
            yield acquired
            
        finally:
            if acquired:
                # 释放资源
                semaphore.release()
                released_at = time.time()
                hold_time = released_at - acquired_at
                
                # 更新统计信息
                with self._stats_lock:
                    self._total_released[resource_type] += 1
                    self._hold_times[resource_type].append(hold_time)
                    
                    # 保持最近1000条记录
                    if len(self._hold_times[resource_type]) > 1000:
                        self._hold_times[resource_type] = self._hold_times[resource_type][-1000:]
                    
                    # 移除使用记录
                    self._current_usage.pop(thread_id, None)
                
                self.logger.debug(
                    f"资源释放成功: {resource_type.value} "
                    f"(任务: {task_id or 'unknown'}, 持有时间: {hold_time:.2f}s)"
                )
    
    def try_acquire_resource(
        self, 
        resource_type: ResourceType,
        task_id: Optional[str] = None
    ) -> bool:
        """
        尝试立即获取资源（非阻塞）
        
        Args:
            resource_type: 资源类型
            task_id: 任务ID
            
        Returns:
            是否成功获取
        """
        with self.acquire_resource(resource_type, timeout=0, task_id=task_id) as acquired:
            return acquired
    
    def get_resource_stats(self, resource_type: ResourceType) -> ResourceStats:
        """获取指定资源类型的统计信息"""
        semaphore = self._semaphores[resource_type]
        max_count = self._max_counts[resource_type]
        
        with self._stats_lock:
            available_count = semaphore._value
            in_use_count = max_count - available_count
            total_acquired = self._total_acquired[resource_type]
            total_released = self._total_released[resource_type]
            
            hold_times = self._hold_times[resource_type]
            if hold_times:
                average_hold_time = sum(hold_times) / len(hold_times)
                max_hold_time = max(hold_times)
            else:
                average_hold_time = 0.0
                max_hold_time = 0.0
        
        return ResourceStats(
            resource_type=resource_type,
            max_count=max_count,
            available_count=available_count,
            in_use_count=in_use_count,
            total_acquired=total_acquired,
            total_released=total_released,
            average_hold_time=average_hold_time,
            max_hold_time=max_hold_time
        )
    
    def get_all_stats(self) -> Dict[ResourceType, ResourceStats]:
        """获取所有资源的统计信息"""
        return {
            resource_type: self.get_resource_stats(resource_type)
            for resource_type in ResourceType
        }
    
    def get_current_usage(self) -> Dict[ResourceType, List[ResourceUsage]]:
        """获取当前资源使用情况"""
        with self._stats_lock:
            usage_by_type = {rt: [] for rt in ResourceType}
            
            for usage in self._current_usage.values():
                usage_by_type[usage.resource_type].append(usage)
            
            return usage_by_type
    
    def get_summary(self) -> Dict[str, Any]:
        """获取资源管理器摘要信息"""
        all_stats = self.get_all_stats()
        current_usage = self.get_current_usage()
        
        return {
            "resource_stats": {
                rt.value: {
                    "max_count": stats.max_count,
                    "available_count": stats.available_count,
                    "in_use_count": stats.in_use_count,
                    "usage_rate": f"{(stats.in_use_count / stats.max_count * 100):.1f}%",
                    "total_acquired": stats.total_acquired,
                    "total_released": stats.total_released,
                    "average_hold_time": f"{stats.average_hold_time:.2f}s",
                    "max_hold_time": f"{stats.max_hold_time:.2f}s",
                }
                for rt, stats in all_stats.items()
            },
            "current_usage": {
                rt.value: len(usages)
                for rt, usages in current_usage.items()
            },
            "total_threads_using_resources": len(self._current_usage),
        }
    
    def wait_for_available(
        self, 
        resource_type: ResourceType, 
        timeout: Optional[float] = None
    ) -> bool:
        """
        等待资源可用
        
        Args:
            resource_type: 资源类型
            timeout: 超时时间
            
        Returns:
            资源是否可用
        """
        semaphore = self._semaphores[resource_type]
        
        # 临时获取资源来检查可用性
        acquired = semaphore.acquire(timeout=timeout)
        if acquired:
            semaphore.release()
            return True
        return False
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        with self._stats_lock:
            self._total_acquired = {rt: 0 for rt in ResourceType}
            self._total_released = {rt: 0 for rt in ResourceType}
            self._hold_times = {rt: [] for rt in ResourceType}
            # 不清除当前使用记录，因为可能有正在使用的资源
        
        self.logger.info("资源统计信息已重置")
    
    def is_resource_available(self, resource_type: ResourceType) -> bool:
        """检查指定资源是否可用"""
        stats = self.get_resource_stats(resource_type)
        return stats.available_count > 0
    
    def get_queue_length(self, resource_type: ResourceType) -> int:
        """
        获取等待队列长度（近似值）
        
        注意：这是一个近似值，实际等待线程数可能不同
        """
        stats = self.get_resource_stats(resource_type)
        # 如果正在使用的资源数等于最大资源数，说明可能有等待队列
        if stats.in_use_count >= stats.max_count:
            # 这里返回一个近似值，实际实现中可以通过更复杂的方式跟踪等待队列
            return max(0, stats.total_acquired - stats.total_released - stats.max_count)
        return 0