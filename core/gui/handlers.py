"""GUI日志和事件处理器"""

import logging

from PySide6.QtCore import Signal


class LogHandler(logging.Handler):
    """自定义日志处理器，用于将日志输出到GUI"""

    def __init__(self, signal_emitter):
        super().__init__()
        self.signal_emitter = signal_emitter

    def emit(self, record):
        msg = self.format(record)
        self.signal_emitter.log_message.emit(msg)