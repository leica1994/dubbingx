"""
状态事件管理器

统一的异步状态管理中心，协调所有状态更新并防止GUI阻塞
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, QTimer

from ...cache import UnifiedCacheManager
from .async_signal_emitter import AsyncSignalEmitter
from .async_status_manager import StatusCache, StatusEvent, StatusEventType


class StatusEventManager(QObject):
    """状态事件管理器 - 异步状态管理的核心"""
    
    def __init__(self, parent=None):
        """初始化状态事件管理器"""
        super().__init__(parent)
        self.logger = logging.getLogger(f"{__name__}.StatusEventManager")
        
        # 核心组件
        self._status_cache = StatusCache()
        
        # 检查线程上下文，确保在主线程中创建信号发送器
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and app.thread() != self.thread():
            # 将对象移动到GUI主线程
            self.moveToThread(app.thread())
        
        # 在正确的线程上下文中创建信号发送器
        self._signal_emitter = AsyncSignalEmitter(self)
        
        # 统一缓存管理器映射 {task_id: UnifiedCacheManager}
        self._cache_managers: Dict[str, UnifiedCacheManager] = {}
        
        # 后台同步定时器
        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self._sync_to_unified_cache)
        self._sync_timer_interval = 500  # 500ms同步间隔（更频繁）
        
        # 状态监控定时器
        self._monitor_timer = QTimer(self)
        self._monitor_timer.timeout.connect(self._monitor_status)
        self._monitor_timer_interval = 5000  # 5秒监控间隔
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 配置参数
        self._enable_cache_sync = True
        self._enable_monitoring = True
        
        # 统计信息
        self._stats = {
            "events_processed": 0,
            "sync_operations": 0,
            "sync_errors": 0,
            "active_tasks": 0,
            "last_sync_time": 0
        }
        
        # 设置状态缓存监听器
        self._status_cache.add_listener(self._on_cache_status_changed)
        
        # 启动定时器
        self._start_timers()

    def _start_timers(self) -> None:
        """启动定时器"""
        if self._enable_cache_sync and not self._sync_timer.isActive():
            self._sync_timer.start(self._sync_timer_interval)
        
        if self._enable_monitoring and not self._monitor_timer.isActive():
            self._monitor_timer.start(self._monitor_timer_interval)

    def _stop_timers(self) -> None:
        """停止定时器"""
        if self._sync_timer.isActive():
            self._sync_timer.stop()
        
        if self._monitor_timer.isActive():
            self._monitor_timer.stop()

    def register_cache_manager(self, task_id: str, cache_manager: UnifiedCacheManager) -> None:
        """注册统一缓存管理器"""
        with self._lock:
            self._cache_managers[task_id] = cache_manager
            self.logger.info(f"注册缓存管理器: {task_id}")

    def unregister_cache_manager(self, task_id: str) -> None:
        """注销统一缓存管理器"""
        with self._lock:
            if task_id in self._cache_managers:
                del self._cache_managers[task_id]
                self.logger.info(f"注销缓存管理器: {task_id}")

    def get_signal_emitter(self) -> AsyncSignalEmitter:
        """获取异步信号发送器"""
        return self._signal_emitter

    def notify_task_started(self, task_id: str, message: str = "") -> None:
        """通知任务开始"""
        event = StatusEvent(
            event_type=StatusEventType.TASK_STARTED,
            task_id=task_id,
            status="processing",
            message=message
        )
        self._process_status_event(event)

    def notify_task_completed(self, task_id: str, message: str = "") -> None:
        """通知任务完成"""
        event = StatusEvent(
            event_type=StatusEventType.TASK_COMPLETED,
            task_id=task_id,
            status="completed",
            message=message
        )
        self._process_status_event(event)

    def notify_task_failed(self, task_id: str, error_message: str, message: str = "") -> None:
        """通知任务失败"""
        event = StatusEvent(
            event_type=StatusEventType.TASK_FAILED,
            task_id=task_id,
            status="failed",
            message=message,
            error_message=error_message
        )
        self._process_status_event(event)

    def notify_step_started(self, task_id: str, step_id: int, total_items: int = 1, message: str = "") -> None:
        """通知步骤开始"""
        event = StatusEvent(
            event_type=StatusEventType.STEP_STARTED,
            task_id=task_id,
            step_id=step_id,
            status="processing",
            message=message,
            total_items=total_items
        )
        self._process_status_event(event)

    def notify_step_progress(self, task_id: str, step_id: int, current_item: int, total_items: int, message: str = "") -> None:
        """通知步骤进度"""
        progress = (current_item / total_items * 100.0) if total_items > 0 else 0.0
        
        event = StatusEvent(
            event_type=StatusEventType.STEP_PROGRESS,
            task_id=task_id,
            step_id=step_id,
            status="processing",
            message=message,
            progress=progress,
            current_item=current_item,
            total_items=total_items
        )
        self._process_status_event(event)

    def notify_step_completed(self, task_id: str, step_id: int, message: str = "") -> None:
        """通知步骤完成"""
        event = StatusEvent(
            event_type=StatusEventType.STEP_COMPLETED,
            task_id=task_id,
            step_id=step_id,
            status="completed",
            message=message,
            progress=100.0
        )
        self._process_status_event(event)

    def notify_step_failed(self, task_id: str, step_id: int, error_message: str, message: str = "") -> None:
        """通知步骤失败"""
        event = StatusEvent(
            event_type=StatusEventType.STEP_FAILED,
            task_id=task_id,
            step_id=step_id,
            status="failed",
            message=message,
            error_message=error_message
        )
        self._process_status_event(event)

    def _process_status_event(self, event: StatusEvent) -> None:
        """处理状态事件"""
        try:
            self._stats["events_processed"] += 1
            
            # 更新状态缓存
            if event.step_id is not None:
                success = self._status_cache.update_step_status(event)
            else:
                success = self._status_cache.update_task_status(event)
            
            if not success:
                return
            
            # 对于关键状态变化，立即同步到统一缓存
            if self._should_immediate_sync(event):
                self._immediate_sync_task(event.task_id)
            
            # 队列信号发送（异步）
            self._signal_emitter.queue_status_event(event)
            
            self.logger.info(f"处理状态事件: {event.event_type.value} - {event.task_id}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.error(f"处理状态事件失败: {e}")

    def _should_immediate_sync(self, event: StatusEvent) -> bool:
        """判断是否需要立即同步到缓存"""
        # 步骤完成或失败时需要立即同步
        critical_events = [
            StatusEventType.STEP_COMPLETED,
            StatusEventType.STEP_FAILED,
            StatusEventType.TASK_COMPLETED,
            StatusEventType.TASK_FAILED
        ]
        return event.event_type in critical_events

    def _immediate_sync_task(self, task_id: str) -> None:
        """立即同步特定任务到统一缓存"""
        try:
            with self._lock:
                if task_id not in self._cache_managers:
                    return
                    
                cache_manager = self._cache_managers[task_id]
                
                # 获取任务的所有步骤状态
                steps = self._status_cache.get_task_steps(task_id)
                
                for step_id, step_info in steps.items():
                    # 立即同步步骤状态到统一缓存
                    status = step_info.get("status", "pending")
                    progress = step_info.get("progress", 0.0)
                    current_item = step_info.get("current_item", 0)
                    total_items = step_info.get("total_items", 1)
                    error_message = step_info.get("error_message")
                    
                    cache_manager.update_step_status(
                        step_id=step_id,
                        status=status,
                        progress_percent=progress,
                        current_item=current_item,
                        total_items=total_items,
                        error_message=error_message,
                        metadata=step_info.get("metadata", {})
                    )
                
                self.logger.debug(f"立即同步任务 {task_id} 到统一缓存")
                    
        except Exception as e:
            self.logger.error(f"立即同步任务 {task_id} 失败: {e}")

    def _on_cache_status_changed(self, event: StatusEvent) -> None:
        """状态缓存变更回调"""
        # 这里可以添加额外的处理逻辑
        pass

    def _sync_to_unified_cache(self) -> None:
        """同步状态到统一缓存系统"""
        if not self._enable_cache_sync:
            return
        
        try:
            with self._lock:
                sync_count = 0
                
                for task_id, cache_manager in self._cache_managers.items():
                    try:
                        # 获取任务的所有步骤状态
                        steps = self._status_cache.get_task_steps(task_id)
                        
                        for step_id, step_info in steps.items():
                            # 同步步骤状态到统一缓存
                            status = step_info.get("status", "pending")
                            progress = step_info.get("progress", 0.0)
                            current_item = step_info.get("current_item", 0)
                            total_items = step_info.get("total_items", 1)
                            error_message = step_info.get("error_message")
                            
                            cache_manager.update_step_status(
                                step_id=step_id,
                                status=status,
                                progress_percent=progress,
                                current_item=current_item,
                                total_items=total_items,
                                error_message=error_message,
                                metadata=step_info.get("metadata", {})
                            )
                        
                        sync_count += 1
                        
                    except Exception as e:
                        self._stats["sync_errors"] += 1
                        self.logger.debug(f"同步任务 {task_id} 失败: {e}")
                
                if sync_count > 0:
                    self._stats["sync_operations"] += 1
                    self._stats["last_sync_time"] = time.time()
                    self.logger.debug(f"同步了 {sync_count} 个任务到统一缓存")
                
        except Exception as e:
            self._stats["sync_errors"] += 1
            self.logger.error(f"缓存同步失败: {e}")

    def _monitor_status(self) -> None:
        """监控状态系统健康度"""
        try:
            # 更新活跃任务数
            all_tasks = self._status_cache.get_all_tasks()
            active_tasks = sum(1 for task_info in all_tasks.values() 
                             if task_info.get("status") not in ["completed", "failed"])
            self._stats["active_tasks"] = active_tasks
            
            # 检查队列健康度
            queue_stats = self._signal_emitter.get_queue_stats()
            total_queue_size = queue_stats.get("total_queue_size", 0)
            
            if total_queue_size > 500:  # 队列积压过多
                self.logger.warning(f"信号队列积压: {total_queue_size} 个事件")
            
            # 检查缓存同步状态
            last_sync = self._stats.get("last_sync_time", 0)
            if last_sync > 0 and time.time() - last_sync > 30:  # 超过30秒未同步
                self.logger.warning("缓存同步可能出现问题")
            
            self.logger.debug(f"状态监控: 活跃任务={active_tasks}, 队列大小={total_queue_size}")
            
        except Exception as e:
            self.logger.error(f"状态监控失败: {e}")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        return self._status_cache.get_task_status(task_id)

    def get_step_status(self, task_id: str, step_id: int) -> Optional[Dict[str, Any]]:
        """获取步骤状态"""
        return self._status_cache.get_step_status(task_id, step_id)

    def get_all_task_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有任务状态"""
        return self._status_cache.get_all_tasks()

    def clear_task_status(self, task_id: str) -> bool:
        """清除任务状态"""
        success = self._status_cache.clear_task(task_id)
        if success:
            self.unregister_cache_manager(task_id)
        return success

    def flush_all_events(self) -> None:
        """立即处理所有待处理事件"""
        self.logger.info("强制刷新所有状态事件")
        
        # 立即同步缓存
        self._sync_to_unified_cache()
        
        # 刷新所有信号
        self._signal_emitter.flush_all_signals()

    def configure(self, **kwargs) -> None:
        """配置状态管理器"""
        if "enable_cache_sync" in kwargs:
            self._enable_cache_sync = bool(kwargs["enable_cache_sync"])
            
        if "enable_monitoring" in kwargs:
            self._enable_monitoring = bool(kwargs["enable_monitoring"])
            
        if "sync_interval" in kwargs:
            self._sync_timer_interval = max(1000, int(kwargs["sync_interval"]))
            
        if "monitor_interval" in kwargs:
            self._monitor_timer_interval = max(1000, int(kwargs["monitor_interval"]))
        
        # 应用信号发送器配置
        signal_config = {}
        for key in ["batch_interval", "dedup_interval", "max_queue_size", "dedup_window_ms", "batch_size"]:
            if key in kwargs:
                signal_config[key] = kwargs[key]
        
        if signal_config:
            self._signal_emitter.configure(**signal_config)
        
        # 重启定时器以应用新配置
        self._stop_timers()
        self._start_timers()
        
        self.logger.info(f"状态管理器配置已更新: {kwargs}")

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        cache_stats = self._status_cache.get_stats()
        queue_stats = self._signal_emitter.get_queue_stats()
        
        return {
            "status_manager": self._stats.copy(),
            "status_cache": cache_stats,
            "signal_emitter": queue_stats,
            "cache_managers_count": len(self._cache_managers),
            "timers_active": {
                "sync_timer": self._sync_timer.isActive(),
                "monitor_timer": self._monitor_timer.isActive()
            }
        }

    def __del__(self):
        """析构函数"""
        try:
            self.flush_all_events()
            self._stop_timers()
        except:
            pass