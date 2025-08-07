"""GUI专用的流水线处理器"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Signal

from ..dubbing_pipeline import StreamlinePipeline


class GUIStreamlinePipeline(StreamlinePipeline):
    """GUI专用的流水线处理器"""

    log_message = Signal(str)

    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(output_dir)
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        """设置日志处理器"""
        handler = self._create_signal_handler()
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self._delayed_setup_logging(handler)

    def _create_signal_handler(self):
        """创建信号日志处理器"""

        class SignalLogHandler(logging.Handler):
            def __init__(self, signal_emitter):
                super().__init__()
                self.signal_emitter = signal_emitter

            def emit(self, record):
                msg = self.format(record)
                self.signal_emitter.log_message.emit(msg)

        handler = SignalLogHandler(self)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        return handler

    def _delayed_setup_logging(self, handler):
        """延迟设置流水线组件的日志处理器"""
        if hasattr(self, "task_scheduler") and self.task_scheduler:
            self._setup_scheduler_logging(handler)
            self.logger.info("流水线日志处理器设置完成")

    def _setup_scheduler_logging(self, handler):
        """设置调度器相关的日志处理器"""
        self.task_scheduler.logger.addHandler(handler)
        self.task_scheduler.logger.setLevel(logging.INFO)

        for processor in self.task_scheduler.processors.values():
            processor.logger.addHandler(handler)
            processor.logger.setLevel(logging.INFO)

        if hasattr(self.task_scheduler, "queue_manager"):
            self.task_scheduler.queue_manager.logger.addHandler(handler)
            self.task_scheduler.queue_manager.logger.setLevel(logging.INFO)

    def process_batch_streamline(
        self,
        video_subtitle_pairs: List[Tuple[str, Optional[str]]],
        resume_from_cache: bool = True,
        enable_vocal_separation: bool = False,
    ) -> Dict[str, Any]:
        """使用流水线模式批量处理视频（带GUI日志支持）"""
        self._ensure_logging_setup()
        return super().process_batch_streamline(video_subtitle_pairs, resume_from_cache, enable_vocal_separation)

    def _ensure_logging_setup(self):
        """确保日志处理器已设置"""
        if hasattr(self, "task_scheduler") and self.task_scheduler:
            handler = self._find_signal_handler()
            if handler:
                self._delayed_setup_logging(handler)

    def _find_signal_handler(self):
        """查找信号处理器"""
        for h in self.logger.handlers:
            if hasattr(h, "signal_emitter"):
                return h
        return None