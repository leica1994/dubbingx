"""
异步信号发送器

使用Qt定时器实现批量、异步的信号发送，防止GUI卡住
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import QObject, QTimer, Signal, QMetaObject, Qt

from .async_status_manager import StatusEvent, StatusEventType


class AsyncSignalEmitter(QObject):
    """异步信号发送器"""
    
    # GUI状态更新信号
    task_status_changed = Signal(str, str, str)  # task_id, status, message
    step_status_changed = Signal(str, int, str, str)  # task_id, step_id, status, message
    step_progress_changed = Signal(str, int, float, int, int, str)  # task_id, step_id, progress, current, total, message
    batch_status_update = Signal(dict)  # 批量状态更新 {task_id: {...}}
    
    # 内部信号，用于跨线程调用
    _internal_queue_event = Signal(object)  # StatusEvent
    _internal_start_batch_timer = Signal()  # 启动批量定时器信号
    
    def __init__(self, parent=None):
        """初始化异步信号发送器"""
        super().__init__(parent)
        self.logger = logging.getLogger(f"{__name__}.AsyncSignalEmitter")
        
        # 事件队列缓冲 {signal_type: deque}
        self._event_queues = defaultdict(deque)
        
        # 信号去重缓存 {signal_key: (event, timestamp)}
        self._signal_cache = {}
        
        # 批量发送定时器
        self._batch_timer = QTimer(self)
        # 使用Qt.QueuedConnection确保在正确线程中执行
        self._batch_timer.timeout.connect(self._process_batch_signals, Qt.QueuedConnection)
        self._batch_timer_interval = 50  # 50ms批量发送间隔
        
        # 去重定时器
        self._dedup_timer = QTimer(self)
        # 使用Qt.QueuedConnection确保在正确线程中执行  
        self._dedup_timer.timeout.connect(self._cleanup_signal_cache, Qt.QueuedConnection)
        self._dedup_timer_interval = 1000  # 1秒清理过期缓存
        
        # 配置参数
        self._max_queue_size = 1000  # 最大队列大小
        self._dedup_window_ms = 100  # 去重时间窗口（毫秒）
        self._batch_size = 20  # 单次批量处理大小
        
        # 统计信息
        self._stats = {
            "total_events": 0,
            "deduped_events": 0,
            "batch_count": 0,
            "queue_overflow": 0
        }
        
        # 检查线程上下文和Qt环境
        from PySide6.QtWidgets import QApplication
        import threading
        
        app = QApplication.instance()
        current_thread = threading.current_thread()
        
        if app:
            app_thread = app.thread()
            self_thread = self.thread()
            
            if app_thread != self_thread:
                self.moveToThread(app_thread)
        
        # 连接内部信号到实际队列方法（确保在主线程中执行）
        self._internal_queue_event.connect(self._do_queue_status_event, Qt.QueuedConnection)
        self._internal_start_batch_timer.connect(self._start_batch_timer_safe, Qt.QueuedConnection)
        
        # 启动去重定时器（批量定时器在有事件时才启动）
        self._start_dedup_timer()

    def _start_batch_timer_safe(self) -> None:
        """在主线程中安全启动批量定时器"""
        try:
            if not self._batch_timer.isActive():
                # 详细诊断QTimer状态
                from PySide6.QtWidgets import QApplication
                import threading
                app = QApplication.instance()
                current_thread = threading.current_thread()
                
                # 启动定时器
                self._batch_timer.start(self._batch_timer_interval)
                is_active = self._batch_timer.isActive()
                
        except Exception as e:
            import traceback
            traceback.print_exc()

    def _start_timers(self) -> None:
        """启动定时器"""
        # 使用信号安全启动批量定时器
        if not self._batch_timer.isActive():
            self._internal_start_batch_timer.emit()
            
        if not self._dedup_timer.isActive():
            self._dedup_timer.start(self._dedup_timer_interval)

    def _start_dedup_timer(self) -> None:
        """启动去重定时器"""
        if not self._dedup_timer.isActive():
            self._dedup_timer.start(self._dedup_timer_interval)

    def _stop_timers(self) -> None:
        """停止定时器"""
        if self._batch_timer.isActive():
            self._batch_timer.stop()
            
        if self._dedup_timer.isActive():
            self._dedup_timer.stop()

    def queue_status_event(self, event: StatusEvent) -> bool:
        """队列状态事件（主要入口）- 线程安全"""
        try:
            # 如果在主线程中调用，直接处理
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app and app.thread() == self.thread():
                return self._do_queue_status_event(event)
            else:
                # 如果在其他线程中调用，使用信号机制（queued连接）
                self._internal_queue_event.emit(event)
                return True
        except Exception as e:
            self.logger.error(f"队列状态事件失败: {e}")
            return False

    def _do_queue_status_event(self, event: StatusEvent) -> bool:
        """实际的队列状态事件处理（在主线程中执行）"""
        try:
            self._stats["total_events"] += 1
            
            # 生成信号键用于去重
            signal_key = self._generate_signal_key(event)
            current_time = time.time() * 1000  # 毫秒
            
            # 检查是否需要去重
            if signal_key in self._signal_cache:
                cached_event, cached_time = self._signal_cache[signal_key]
                if current_time - cached_time < self._dedup_window_ms:
                    # 在去重时间窗口内，跳过
                    self._stats["deduped_events"] += 1
                    return False
            
            # 更新缓存
            self._signal_cache[signal_key] = (event, current_time)
            
            # 确定队列类型
            queue_type = self._get_queue_type(event)
            
            # 检查队列是否已满
            if len(self._event_queues[queue_type]) >= self._max_queue_size:
                # 队列满，移除最老的事件
                self._event_queues[queue_type].popleft()
                self._stats["queue_overflow"] += 1
            
            # 添加到队列
            self._event_queues[queue_type].append(event)
            
            # 确保批量定时器运行（使用信号安全启动）
            if not self._batch_timer.isActive():
                self._internal_start_batch_timer.emit()
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.error(f"队列状态事件失败: {e}")
            return False

    def _generate_signal_key(self, event: StatusEvent) -> str:
        """生成信号键用于去重"""
        base_key = f"{event.task_id}_{event.event_type.value}"
        
        if event.step_id is not None:
            base_key += f"_{event.step_id}"
        
        # 对于进度事件，添加进度值以实现更精细的去重
        if event.event_type == StatusEventType.STEP_PROGRESS:
            base_key += f"_{int(event.progress)}_{event.current_item}"
            
        return base_key

    def _get_queue_type(self, event: StatusEvent) -> str:
        """获取队列类型"""
        if event.event_type in [StatusEventType.TASK_STARTED, 
                               StatusEventType.TASK_COMPLETED, 
                               StatusEventType.TASK_FAILED]:
            return "task_events"
        elif event.event_type == StatusEventType.STEP_PROGRESS:
            return "progress_events"
        elif event.event_type in [StatusEventType.STEP_STARTED, 
                                 StatusEventType.STEP_COMPLETED, 
                                 StatusEventType.STEP_FAILED]:
            return "step_events"
        else:
            return "other_events"

    def _process_batch_signals(self) -> None:
        """批量处理信号发送"""
        try:
            # 添加线程诊断
            from PySide6.QtWidgets import QApplication
            import threading
            app = QApplication.instance()
            current_thread = threading.current_thread()
            
            total_processed = 0
            
            # 统计所有队列的事件数量
            total_events = sum(len(queue) for queue in self._event_queues.values())
            
            if total_events == 0:
                if self._batch_timer.isActive():
                    self._batch_timer.stop()
                return
            
            # 处理各种类型的事件队列
            for queue_type in list(self._event_queues.keys()):
                queue = self._event_queues[queue_type]
                if not queue:
                    continue
                
                # 批量处理
                batch_count = min(len(queue), self._batch_size)
                events_to_process = []
                
                for _ in range(batch_count):
                    if queue:
                        events_to_process.append(queue.popleft())
                
                if events_to_process:
                    self._emit_batch_signals(queue_type, events_to_process)
                    total_processed += len(events_to_process)
            
            if total_processed > 0:
                self._stats["batch_count"] += 1
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.error(f"批量处理信号失败: {e}")

    def _emit_batch_signals(self, queue_type: str, events: List[StatusEvent]) -> None:
        """发送批量信号"""
        try:
            if queue_type == "task_events":
                self._emit_task_events(events)
            elif queue_type == "step_events":
                self._emit_step_events(events)
            elif queue_type == "progress_events":
                self._emit_progress_events(events)
            else:
                self._emit_other_events(events)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.error(f"发送批量信号失败 ({queue_type}): {e}")

    def _emit_task_events(self, events: List[StatusEvent]) -> None:
        """发送任务事件信号"""
        for event in events:
            try:
                self.task_status_changed.emit(
                    event.task_id,
                    event.status,
                    event.message or ""
                )
            except Exception as e:
                self.logger.debug(f"发送任务状态信号失败: {e}")

    def _emit_step_events(self, events: List[StatusEvent]) -> None:
        """发送步骤事件信号"""
        for event in events:
            try:
                if event.step_id is not None:
                    self.step_status_changed.emit(
                        event.task_id,
                        event.step_id,
                        event.status,
                        event.message or ""
                    )
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.logger.debug(f"发送步骤状态信号失败: {e}")

    def _emit_progress_events(self, events: List[StatusEvent]) -> None:
        """发送进度事件信号"""
        for event in events:
            try:
                if event.step_id is not None:
                    self.step_progress_changed.emit(
                        event.task_id,
                        event.step_id,
                        event.progress,
                        event.current_item,
                        event.total_items,
                        event.message or ""
                    )
            except Exception as e:
                self.logger.debug(f"发送进度信号失败: {e}")

    def _emit_other_events(self, events: List[StatusEvent]) -> None:
        """发送其他事件信号"""
        # 可以发送批量更新信号
        if events:
            try:
                batch_data = {}
                for event in events:
                    task_id = event.task_id
                    if task_id not in batch_data:
                        batch_data[task_id] = []
                    batch_data[task_id].append({
                        "event_type": event.event_type.value,
                        "step_id": event.step_id,
                        "status": event.status,
                        "progress": event.progress,
                        "message": event.message,
                        "timestamp": event.timestamp
                    })
                
                if batch_data:
                    self.batch_status_update.emit(batch_data)
            except Exception as e:
                self.logger.debug(f"发送批量更新信号失败: {e}")

    def _cleanup_signal_cache(self) -> None:
        """清理过期的信号缓存"""
        try:
            current_time = time.time() * 1000
            expired_keys = []
            
            for signal_key, (event, cached_time) in self._signal_cache.items():
                if current_time - cached_time > self._dedup_window_ms * 2:  # 过期时间为去重窗口的2倍
                    expired_keys.append(signal_key)
            
            for key in expired_keys:
                del self._signal_cache[key]
            
            if expired_keys:
                self.logger.debug(f"清理 {len(expired_keys)} 个过期信号缓存")
                
        except Exception as e:
            self.logger.error(f"清理信号缓存失败: {e}")

    def flush_all_signals(self) -> None:
        """立即发送所有待处理信号"""
        try:
            self.logger.info("强制刷新所有待处理信号")
            
            # 停止定时器
            self._stop_timers()
            
            # 处理所有队列
            total_flushed = 0
            for queue_type in list(self._event_queues.keys()):
                queue = self._event_queues[queue_type]
                if queue:
                    events = list(queue)
                    queue.clear()
                    self._emit_batch_signals(queue_type, events)
                    total_flushed += len(events)
            
            self.logger.info(f"刷新了 {total_flushed} 个信号")
            
            # 重启定时器
            self._start_timers()
            
        except Exception as e:
            self.logger.error(f"刷新信号失败: {e}")
            # 确保定时器重启
            self._start_timers()

    def clear_queues(self) -> None:
        """清空所有队列"""
        try:
            total_cleared = sum(len(queue) for queue in self._event_queues.values())
            self._event_queues.clear()
            self._signal_cache.clear()
            
            self.logger.info(f"清空了 {total_cleared} 个待处理信号")
            
        except Exception as e:
            self.logger.error(f"清空队列失败: {e}")

    def get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        queue_sizes = {queue_type: len(queue) for queue_type, queue in self._event_queues.items()}
        
        return {
            "queue_sizes": queue_sizes,
            "total_queue_size": sum(queue_sizes.values()),
            "cache_size": len(self._signal_cache),
            "stats": self._stats.copy(),
            "timers_active": {
                "batch_timer": self._batch_timer.isActive(),
                "dedup_timer": self._dedup_timer.isActive()
            }
        }

    def configure(self, **kwargs) -> None:
        """配置信号发送器参数"""
        if "batch_interval" in kwargs:
            self._batch_timer_interval = max(10, int(kwargs["batch_interval"]))
        if "dedup_interval" in kwargs:
            self._dedup_timer_interval = max(500, int(kwargs["dedup_interval"]))
        if "max_queue_size" in kwargs:
            self._max_queue_size = max(100, int(kwargs["max_queue_size"]))
        if "dedup_window_ms" in kwargs:
            self._dedup_window_ms = max(50, int(kwargs["dedup_window_ms"]))
        if "batch_size" in kwargs:
            self._batch_size = max(1, int(kwargs["batch_size"]))
        
        # 重启定时器以应用新配置
        self._stop_timers()
        self._start_timers()
        
        self.logger.info(f"信号发送器配置已更新: {kwargs}")

    def __del__(self):
        """析构函数"""
        try:
            self.flush_all_signals()
            self._stop_timers()
        except:
            pass