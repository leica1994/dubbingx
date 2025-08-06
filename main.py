"""DubbingX GUI - 智能视频配音系统图形界面"""

import ctypes
import logging
import os
import re
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.cache import TaskCacheManager
from core.dubbing_pipeline import StreamlinePipeline
from core.tts_processor import initialize_tts_processor
from core.util import sanitize_filename


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
    ) -> Dict[str, Any]:
        """使用流水线模式批量处理视频（带GUI日志支持）"""
        self._ensure_logging_setup()
        return super().process_batch_streamline(video_subtitle_pairs, resume_from_cache)

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


class VideoSubtitleMatcher:
    """视频字幕匹配器"""

    @staticmethod
    def find_video_subtitle_pairs(folder_path: str) -> List[Tuple[str, str]]:
        """在文件夹中查找视频和对应的字幕文件"""
        folder = Path(folder_path)
        if not folder.exists():
            return []

        video_files = VideoSubtitleMatcher._find_video_files(folder)
        return VideoSubtitleMatcher._match_subtitles(video_files)

    @staticmethod
    def _find_video_files(folder: Path) -> List[Path]:
        """查找所有视频文件"""
        video_extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
            ".m4v",
            ".webm",
            ".ts",
        }

        video_files_set = set()
        for ext in video_extensions:
            video_files_set.update(folder.rglob(f"*{ext}"))
            video_files_set.update(folder.rglob(f"*{ext.upper()}"))

        return sorted(list(video_files_set))

    @staticmethod
    def _match_subtitles(video_files: List[Path]) -> List[Tuple[str, str]]:
        """为视频文件匹配字幕"""
        subtitle_extensions = {".srt", ".ass", ".ssa", ".sub", ".vtt"}
        pairs = []
        matched_subtitles = set()

        for video_file in video_files:
            subtitle_file = VideoSubtitleMatcher._find_matching_subtitle(
                video_file, subtitle_extensions, matched_subtitles
            )
            if subtitle_file:
                pairs.append((str(video_file), str(subtitle_file)))
                matched_subtitles.add(str(subtitle_file))

        return pairs

    @staticmethod
    def _find_matching_subtitle(
        video_file: Path, subtitle_extensions: set, matched_subtitles: set
    ) -> Optional[Path]:
        """为单个视频文件查找匹配的字幕"""
        video_name = video_file.stem
        video_folder = video_file.parent

        for ext in subtitle_extensions:
            for case_ext in [ext, ext.upper()]:
                potential_subtitle = video_folder / f"{video_name}{case_ext}"
                if (
                    potential_subtitle.exists()
                    and str(potential_subtitle) not in matched_subtitles
                ):
                    return potential_subtitle
        return None


class LogHandler(logging.Handler):
    """自定义日志处理器，用于将日志输出到GUI"""

    def __init__(self, signal_emitter):
        super().__init__()
        self.signal_emitter = signal_emitter

    def emit(self, record):
        msg = self.format(record)
        self.signal_emitter.log_message.emit(msg)


class StreamlineBatchDubbingWorkerThread(QThread):
    """流水线批量配音处理工作线程 - 使用 process_batch_streamline 方法"""

    # 信号定义
    progress_update = Signal(int, int, str)  # 当前进度, 总数, 当前处理文件名
    batch_finished = Signal(bool, str, dict)  # 是否成功, 消息, 结果详情
    log_message = Signal(str)  # 日志消息

    def __init__(
        self,
        video_subtitle_pairs: List[Tuple[str, str]],
        resume_from_cache: bool = True,
    ):
        super().__init__()
        self.pairs = video_subtitle_pairs
        self.resume_from_cache = resume_from_cache
        self.is_cancelled = False

        # 创建流水线管道
        self.pipeline = GUIStreamlinePipeline()

        # 连接流水线日志信号到工作线程信号
        self.pipeline.log_message.connect(self.log_message.emit)

    def cancel(self):
        """取消处理"""
        self.is_cancelled = True

    def run(self):
        """执行并行批量处理"""
        try:
            # 注意：StreamlinePipeline暂不支持进度回调
            # 进度更新需要通过其他方式获取

            # 执行流水线批量处理
            result = self.pipeline.process_batch_streamline(
                self.pairs, self.resume_from_cache
            )

            # 发送完成信号
            self.batch_finished.emit(result["success"], result["message"], result)

        except Exception as e:
            self.batch_finished.emit(False, f"流水线批量处理失败: {str(e)}", {})


class DubbingGUI(QMainWindow):
    """DubbingX 主窗口"""

    # 信号定义
    log_message = Signal(str)  # 日志消息

    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.parallel_batch_worker_thread = None
        self.gui_pipeline = GUIStreamlinePipeline()

        # 连接日志信号
        self.log_message.connect(self.append_log_message)

        # 连接直接状态信号（优先级高于日志解析）
        self.gui_pipeline.step_status_changed.connect(self.update_step_status_direct)

        # 添加状态更新的线程同步和时间戳跟踪
        self._status_update_lock = threading.Lock()
        self._step_timestamps = {}  # 格式: {(task_name, step_id): timestamp}
        self._step_sequence = {}    # 格式: {(task_name, step_id): sequence_number}
        self._global_sequence = 0   # 全局序列号

        # 设置窗口属性
        self.setWindowTitle("DubbingX - 智能视频配音系统")
        self.setGeometry(100, 100, 1600, 900)  # 增加窗口尺寸：宽度1600，高度900

        # 设置字体
        font = QFont("微软雅黑", 11)  # 增大字体到11pt
        self.setFont(font)

        # 设置应用样式 - 现代化亮色主题
        self.setup_theme()

        # 初始化UI
        self.init_ui()

        # 设置日志
        self.setup_logging()

        # 初始化logger
        self.logger = logging.getLogger(__name__)

        # 状态变量
        self.current_mode = "single"  # "single" 或 "batch"
        self.current_video_path = ""
        self.current_subtitle_path = ""
        self.current_folder_path = ""
        self.video_subtitle_pairs = []

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        try:
            print("开始关闭应用程序...")
            self._cleanup_resources()
            print("应用程序资源清理完成")
        except Exception as e:
            print(f"关闭时清理资源出错: {e}")
        finally:
            event.accept()
            self._force_exit()

    def _cleanup_resources(self):
        """清理所有资源"""
        self._cleanup_worker_threads()
        self._cleanup_pipelines()
        self._cleanup_log_handlers()

    def _cleanup_worker_threads(self):
        """清理工作线程"""
        self._cleanup_single_thread()
        self._cleanup_batch_thread()

    def _cleanup_single_thread(self):
        """清理单文件处理线程"""
        if hasattr(self, "worker_thread") and self.worker_thread is not None:
            print("停止单文件处理线程...")
            self._terminate_thread(self.worker_thread)
            self.worker_thread = None

    def _cleanup_batch_thread(self):
        """清理批量处理线程"""
        if (
            hasattr(self, "parallel_batch_worker_thread")
            and self.parallel_batch_worker_thread is not None
        ):
            print("停止批量处理线程...")
            if hasattr(self.parallel_batch_worker_thread, "cancel"):
                self.parallel_batch_worker_thread.cancel()
            self._terminate_thread(self.parallel_batch_worker_thread)
            self.parallel_batch_worker_thread = None

    def _terminate_thread(self, thread):
        """终止线程"""
        if thread.isRunning():
            thread.terminate()
            thread.wait(1000)
            if thread.isRunning():
                thread.kill()

    def _cleanup_pipelines(self):
        """清理流水线资源"""
        if hasattr(self, "gui_pipeline") and self.gui_pipeline is not None:
            print("停止GUI流水线...")
            self._stop_task_scheduler()
            self._cleanup_gui_pipeline()

    def _stop_task_scheduler(self):
        """停止任务调度器"""
        if (
            hasattr(self.gui_pipeline, "task_scheduler")
            and self.gui_pipeline.task_scheduler is not None
        ):
            try:
                self.gui_pipeline.task_scheduler.stop(timeout=2.0)
                self._force_close_thread_pools()
            except Exception as e:
                print(f"停止TaskScheduler时出错: {e}")

    def _force_close_thread_pools(self):
        """强制关闭所有线程池"""
        if hasattr(self.gui_pipeline.task_scheduler, "worker_pools"):
            for (
                step_id,
                executor,
            ) in self.gui_pipeline.task_scheduler.worker_pools.items():
                print(f"强制关闭步骤 {step_id} 的线程池...")
                self._force_terminate_threads(executor)
                executor.shutdown(wait=False)

    def _force_terminate_threads(self, executor):
        """强制终止线程池中的线程"""
        if hasattr(executor, "_threads"):
            for thread in list(executor._threads):
                if thread.is_alive():
                    try:
                        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_long(thread.ident), ctypes.py_object(SystemExit)
                        )
                        print(f"线程 {thread.ident} 终止状态: {ret}")
                    except Exception as e:
                        print(f"强制终止线程失败: {e}")

    def _cleanup_gui_pipeline(self):
        """清理GUI流水线"""
        if hasattr(self.gui_pipeline, "cleanup"):
            try:
                self.gui_pipeline.cleanup()
            except Exception as e:
                print(f"清理GUI流水线时出错: {e}")
        self.gui_pipeline = None

    def _cleanup_log_handlers(self):
        """清理日志处理器"""
        try:
            root_logger = logging.getLogger()
            handlers_to_remove = [
                h for h in root_logger.handlers[:] if hasattr(h, "signal_emitter")
            ]

            for handler in handlers_to_remove:
                root_logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass
        except Exception as e:
            print(f"清理日志处理器时出错: {e}")

    def _force_exit(self):
        """强制退出应用程序"""
        self._print_thread_info()
        self._attempt_normal_exit()
        self._emergency_exit()

    def _print_thread_info(self):
        """打印线程信息"""
        active_threads = threading.active_count()
        print(f"当前活跃线程数: {active_threads}")

        if active_threads > 1:
            print("活跃线程列表:")
            for thread in threading.enumerate():
                print(f"  - {thread.name} (daemon: {thread.daemon})")

    def _attempt_normal_exit(self):
        """尝试正常退出"""
        app = QApplication.instance()
        if app:
            app.quit()

    def _emergency_exit(self):
        """紧急强制退出"""
        print("强制退出进程...")
        try:
            os._exit(0)
        except SystemExit:
            signal.SIGTERM

            os.kill(os.getpid(), signal.SIGTERM)

    def setup_theme(self):
        """设置应用主题"""
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f8f9fa;
                color: #212529;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                margin: 8px 0px;
                padding-top: 15px;
                background-color: #ffffff;
                color: #495057;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #495057;
                font-weight: bold;
                background-color: #ffffff;
            }
            QPushButton {
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 10px 20px;
                background-color: #ffffff;
                font-size: 12px;
                font-weight: normal;
                min-height: 25px;
                color: #495057;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
            }
            QLineEdit {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 8px;
                background-color: #ffffff;
                font-size: 11px;
                color: #495057;
            }
            QLineEdit:focus {
                border-color: #0d6efd;
                box-shadow: 0 0 0 0.2rem rgba(13, 110, 253, 0.25);
            }
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                text-align: center;
                background-color: #f8f9fa;
                font-size: 11px;
                color: #495057;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d6efd, stop:1 #6610f2);
                border-radius: 3px;
            }
            QLabel {
                font-size: 12px;
                color: #495057;
                font-weight: normal;
                background-color: transparent;
            }
            QRadioButton {
                font-size: 12px;
                spacing: 8px;
                color: #495057;
                background-color: transparent;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #ced4da;
                border-radius: 8px;
                background-color: #ffffff;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #0d6efd;
                border-radius: 8px;
                background-color: #0d6efd;
            }
            QCheckBox {
                font-size: 12px;
                spacing: 8px;
                color: #495057;
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #ced4da;
                border-radius: 3px;
                background-color: #ffffff;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #198754;
                background-color: #198754;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAiIGhlaWdodD0iMTAiIHZpZXdCb3g9IjAgMCAxMCAxMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEuNSA1TDQgNy41TDguNSAzIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K);
            }
            QTextEdit {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: "Consolas", "Courier New", "微软雅黑", monospace;
                font-size: 10px;
                background-color: #ffffff;
                color: #495057;
                selection-background-color: #cfe2ff;
            }
            QTableWidget {
                gridline-color: #dee2e6;
                background-color: #ffffff;
                alternate-background-color: #f8f9fa;
                font-size: 11px;
                color: #495057;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #dee2e6;
            }
            QTableWidget::item:selected {
                background-color: #cfe2ff;
                color: #495057;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f1f3f4);
                padding: 10px;
                border: 1px solid #dee2e6;
                font-weight: bold;
                font-size: 11px;
                color: #495057;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                background-color: #ffffff;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 12px;
                color: #495057;
                border: 1px solid #dee2e6;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f8f9fa);
                font-weight: bold;
                color: #0d6efd;
            }
            QTabBar::tab:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e9ecef, stop:1 #dee2e6);
            }
            QScrollBar:vertical {
                background-color: #f8f9fa;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #6c757d;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #495057;
            }
            QScrollBar:horizontal {
                background-color: #f8f9fa;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #6c757d;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #495057;
            }
            /* 消息框样式 */
            QMessageBox {
                background-color: #ffffff;
                color: #495057;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
            QMessageBox QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0d6efd, stop:1 #0b5ed7);
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                min-height: 32px;
            }
            QMessageBox QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0b5ed7, stop:1 #0a58ca);
            }
            QMessageBox QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0a58ca, stop:1 #0950b5);
            }
            QMessageBox QLabel {
                color: #495057;
                font-size: 12px;
                background-color: transparent;
            }
            QMessageBox QMessageBoxCritical {
                background-color: #ffffff;
            }
            QMessageBox QMessageBoxInformation {
                background-color: #ffffff;
            }
            QMessageBox QMessageBoxWarning {
                background-color: #ffffff;
            }
            QMessageBox QMessageBoxQuestion {
                background-color: #ffffff;
            }
        """
        )

        # 状态变量
        self.current_mode = "single"  # "single" 或 "batch"
        self.current_video_path = ""
        self.current_subtitle_path = ""
        self.current_folder_path = ""
        self.video_subtitle_pairs = []

    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(6)  # 减少间距
        main_layout.setContentsMargins(10, 10, 10, 10)  # 减少边距

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧面板
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # 右侧面板（日志和结果）
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # 设置分割器比例
        splitter.setSizes([550, 850])

    def create_left_panel(self) -> QWidget:
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)  # 减少间距
        layout.setContentsMargins(8, 8, 8, 8)  # 减少边距

        # 模式选择
        mode_group = QGroupBox("处理模式")
        mode_layout = QHBoxLayout(mode_group)
        mode_layout.setContentsMargins(15, 12, 15, 10)  # 减少内边距

        # 创建按钮组确保单选
        self.mode_button_group = QButtonGroup()

        self.single_mode_radio = QRadioButton("单文件模式")
        self.single_mode_radio.setChecked(True)
        self.single_mode_radio.toggled.connect(self.on_mode_changed)
        self.mode_button_group.addButton(self.single_mode_radio)
        mode_layout.addWidget(self.single_mode_radio)

        self.batch_mode_radio = QRadioButton("批量处理模式")
        self.batch_mode_radio.toggled.connect(self.on_mode_changed)
        self.mode_button_group.addButton(self.batch_mode_radio)
        mode_layout.addWidget(self.batch_mode_radio)

        mode_layout.addStretch()
        layout.addWidget(mode_group)

        # 单文件模式面板
        self.single_file_panel = self.create_single_file_panel()
        layout.addWidget(self.single_file_panel)

        # 批量处理面板
        self.batch_panel = self.create_batch_panel()
        self.batch_panel.hide()
        layout.addWidget(self.batch_panel)

        # 处理控制组
        control_group = QGroupBox("处理控制")
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(15, 12, 15, 12)  # 减少内边距

        # 控制按钮
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("开始处理")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setStyleSheet(
            """
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #198754, stop:1 #157347);
                color: white; 
                font-weight: bold; 
                font-size: 11px;
                min-height: 32px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #157347, stop:1 #146c43);
            }
            QPushButton:pressed { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #146c43, stop:1 #0f5132);
            }
        """
        )
        button_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("取消处理")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(
            """
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #dc3545, stop:1 #bb2d3b);
                color: white; 
                font-weight: bold; 
                font-size: 11px;
                min-height: 32px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #bb2d3b, stop:1 #b02a37);
            }
            QPushButton:pressed { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #b02a37, stop:1 #a02834);
            }
        """
        )
        button_layout.addWidget(self.cancel_btn)

        control_layout.addLayout(button_layout)

        # 缓存控制按钮
        cache_layout = QHBoxLayout()

        self.cache_info_btn = QPushButton("缓存信息")
        self.cache_info_btn.clicked.connect(self.show_cache_info)
        cache_layout.addWidget(self.cache_info_btn)

        self.clear_cache_btn = QPushButton("清理缓存")
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        cache_layout.addWidget(self.clear_cache_btn)

        self.clear_outputs_btn = QPushButton("清理输出目录")
        self.clear_outputs_btn.clicked.connect(self.clear_output_directories)
        cache_layout.addWidget(self.clear_outputs_btn)

        self.repair_cache_btn = QPushButton("修复缓存")
        self.repair_cache_btn.clicked.connect(self.repair_cache)
        cache_layout.addWidget(self.repair_cache_btn)

        control_layout.addLayout(cache_layout)

        layout.addWidget(control_group)

        # 处理选项组
        options_group = QGroupBox("处理选项")
        options_layout = QVBoxLayout(options_group)
        options_layout.setContentsMargins(15, 12, 15, 12)  # 减少内边距

        # 默认从缓存恢复处理（移除用户选择）

        # 批量处理模式选项 - 默认并行处理
        batch_mode_layout = QHBoxLayout()

        batch_mode_layout.addWidget(QLabel("默认并行批量处理"))

        batch_mode_layout.addStretch()
        options_layout.addLayout(batch_mode_layout)

        # 并行处理选项
        parallel_options_layout = QHBoxLayout()

        parallel_options_layout.addWidget(QLabel("最大工作线程:"))

        self.max_workers_spinbox = QLineEdit()
        self.max_workers_spinbox.setPlaceholderText("自动")
        self.max_workers_spinbox.setMaximumWidth(60)
        parallel_options_layout.addWidget(self.max_workers_spinbox)

        parallel_options_layout.addStretch()
        options_layout.addLayout(parallel_options_layout)

        # Index-TTS API配置
        api_options_layout = QHBoxLayout()

        api_options_layout.addWidget(QLabel("Index-TTS API:"))

        self.api_url_edit = QLineEdit()
        self.api_url_edit.setPlaceholderText("http://127.0.0.1:7860")
        self.api_url_edit.setText("http://127.0.0.1:7860")  # 设置默认值
        self.api_url_edit.setMinimumWidth(200)
        api_options_layout.addWidget(self.api_url_edit)

        api_options_layout.addStretch()
        options_layout.addLayout(api_options_layout)

        layout.addWidget(options_group)

        # 添加弹性空间
        layout.addStretch()

        return panel

    def create_single_file_panel(self) -> QWidget:
        """创建单文件模式面板"""
        panel = QGroupBox("文件选择")
        layout = QGridLayout(panel)
        layout.setContentsMargins(15, 12, 15, 12)  # 减少内边距
        layout.setSpacing(10)  # 减少间距

        # 视频文件
        video_label = QLabel("视频文件:")
        video_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #333333;")
        layout.addWidget(video_label, 0, 0)

        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("选择视频文件...")
        layout.addWidget(self.video_path_edit, 0, 1)

        self.video_browse_btn = QPushButton("浏览")
        self.video_browse_btn.clicked.connect(self.browse_video_file)
        layout.addWidget(self.video_browse_btn, 0, 2)

        # 字幕文件
        subtitle_label = QLabel("字幕文件:")
        subtitle_label.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: #333333;"
        )
        layout.addWidget(subtitle_label, 1, 0)

        self.subtitle_path_edit = QLineEdit()
        self.subtitle_path_edit.setPlaceholderText(
            "选择字幕文件（可选，自动匹配同名文件）..."
        )
        layout.addWidget(self.subtitle_path_edit, 1, 1)

        self.subtitle_browse_btn = QPushButton("浏览")
        self.subtitle_browse_btn.clicked.connect(self.browse_subtitle_file)
        layout.addWidget(self.subtitle_browse_btn, 1, 2)

        return panel

    def create_batch_panel(self) -> QWidget:
        """创建批量处理面板"""
        panel = QGroupBox("批量处理")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 12, 15, 12)  # 减少内边距
        layout.setSpacing(8)  # 减少间距

        # 文件夹选择
        folder_layout = QHBoxLayout()

        folder_label = QLabel("文件夹:")
        folder_label.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: #333333;"
        )
        folder_layout.addWidget(folder_label)

        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("选择包含视频和字幕的文件夹...")
        folder_layout.addWidget(self.folder_path_edit)

        self.folder_browse_btn = QPushButton("浏览文件夹")
        self.folder_browse_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(self.folder_browse_btn)

        self.scan_btn = QPushButton("扫描匹配")
        self.scan_btn.clicked.connect(self.scan_folder)
        self.scan_btn.setStyleSheet(
            """
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0d6efd, stop:1 #0b5ed7);
                color: white; 
                font-weight: bold; 
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0b5ed7, stop:1 #0a58ca);
            }
            QPushButton:pressed { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0a58ca, stop:1 #0950b5);
            }
        """
        )
        folder_layout.addWidget(self.scan_btn)

        layout.addLayout(folder_layout)

        # 文件列表表格
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(4)
        self.file_table.setHorizontalHeaderLabels(
            ["视频文件", "字幕文件", "状态", "选择"]
        )

        # 设置表格样式
        self.file_table.setStyleSheet(
            """
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                gridline-color: #e9ecef;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #e9ecef;
            }
            QTableWidget::item:selected {
                background-color: #e7f3ff;
                color: #0d6efd;
            }
            QHeaderView::section:horizontal {
                background-color: #f8f9fa;
                color: #495057;
                padding: 8px;
                border: 1px solid #dee2e6;
                font-weight: bold;
                font-size: 12px;
            }
            QHeaderView::section:vertical {
                background-color: #f8f9fa;
                color: #495057;
                padding: 8px;
                border: 1px solid #dee2e6;
                font-weight: bold;
                font-size: 12px;
            }
            QTableCornerButton::section {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
            }
            QTableWidget#verticalHeader {
                background-color: #f8f9fa;
            }
        """
        )

        # 隐藏垂直表头（序号列）
        self.file_table.verticalHeader().setVisible(False)

        # 限制表格高度
        self.file_table.setMaximumHeight(180)
        self.file_table.setMinimumHeight(100)

        # 设置表格列宽
        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        layout.addWidget(self.file_table)

        # 批量操作按钮
        batch_control_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self.select_all_files)
        batch_control_layout.addWidget(self.select_all_btn)

        self.deselect_all_btn = QPushButton("全不选")
        self.deselect_all_btn.clicked.connect(self.deselect_all_files)
        batch_control_layout.addWidget(self.deselect_all_btn)

        batch_control_layout.addStretch()
        layout.addLayout(batch_control_layout)

        return panel

    def create_right_panel(self) -> QWidget:
        """创建右侧信息面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # 创建标签页
        tab_widget = QTabWidget()

        # 处理状态标签页 - 放在最前面
        status_tab = self.create_status_tab()
        tab_widget.addTab(status_tab, "处理状态")

        # 日志标签页
        log_tab = self.create_log_tab()
        tab_widget.addTab(log_tab, "日志输出")

        layout.addWidget(tab_widget)

        return panel

    def create_log_tab(self) -> QWidget:
        """创建日志标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # 日志控制
        log_control_layout = QHBoxLayout()

        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.clear_log)
        log_control_layout.addWidget(clear_log_btn)

        save_log_btn = QPushButton("保存日志")
        save_log_btn.clicked.connect(self.save_log)
        log_control_layout.addWidget(save_log_btn)

        log_control_layout.addStretch()
        layout.addLayout(log_control_layout)

        # 日志文本区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #ffffff;
                color: #495057;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 15px;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 9px;
                line-height: 1.4;
            }
        """
        )
        layout.addWidget(self.log_text)

        return tab

    def create_status_tab(self) -> QWidget:
        """创建处理状态标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # 状态表格标题
        title_label = QLabel("任务处理状态")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # 处理状态表格 - 显示8个步骤的详细状态
        self.status_table = QTableWidget()
        self.status_table.setColumnCount(11)  # 视频文件 + 8个步骤 + 状态 + 进度
        self.status_table.setHorizontalHeaderLabels(
            [
                "视频文件",
                "步骤1\n字幕预处理",
                "步骤2\n媒体分离",
                "步骤3\n参考音频",
                "步骤4\nTTS生成",
                "步骤5\n音频对齐",
                "步骤6\n对齐字幕",
                "步骤7\n视频调速",
                "步骤8\n合并输出",
                "整体状态",
                "进度",
            ]
        )

        # 设置表格样式
        self.status_table.setStyleSheet(
            """
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                gridline-color: #e9ecef;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #e9ecef;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            QTableWidget#verticalHeader {
                background-color: #f8f9fa;
            }
            """
        )

        # 隐藏垂直表头
        self.status_table.verticalHeader().setVisible(False)

        # 设置表格列宽
        header = self.status_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # 视频文件名

        # 8个步骤列设置为固定宽度
        for i in range(1, 9):  # 列索引 1-8
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        header.setSectionResizeMode(9, QHeaderView.ResizeToContents)  # 整体状态
        header.setSectionResizeMode(10, QHeaderView.ResizeToContents)  # 进度

        layout.addWidget(self.status_table)

        # 状态说明
        legend_layout = QHBoxLayout()
        legend_label = QLabel("状态图标说明：")
        legend_label.setFont(QFont("Microsoft YaHei", 9))
        legend_layout.addWidget(legend_label)

        icons_label = QLabel("⏸️ 未开始   🔄 处理中   ✅ 已完成   ❌ 失败")
        icons_label.setFont(QFont("Microsoft YaHei", 9))
        icons_label.setStyleSheet("color: #6c757d;")
        legend_layout.addWidget(icons_label)

        legend_layout.addStretch()
        layout.addLayout(legend_layout)

        return tab

    def append_log_message(self, message: str):
        """追加日志消息到文本框并解析任务状态"""
        self.log_text.append(message)

        # 解析日志消息来更新任务状态
        self.parse_log_for_task_status(message)

    def parse_log_for_task_status(self, message: str):
        """解析日志消息以更新任务状态

        注意：此方法作为备用机制保留，直接状态信号（update_step_status_direct）具有更高优先级
        """
        try:
            # 1. 解析工作线程开始处理任务的日志
            # 格式: "工作线程 Step-X-stepname-Y 开始处理任务 streamline_task_XXX_taskname"
            if (
                "工作线程" in message
                and "开始处理任务" in message
                and "streamline_task_" in message
            ):
                step_match = re.search(r"Step-(\d+)-", message)
                task_match = re.search(r"streamline_task_\d+_(.*?)(?:\s|$)", message)

                if step_match and task_match:
                    step_id = int(step_match.group(1))
                    task_name = task_match.group(1).strip()
                    self.update_task_step_status(task_name, step_id, "processing")
                    return

            # 2. 解析处理器开始处理任务的日志（基类统一格式）
            # 格式: "开始处理任务 streamline_task_XXX_taskname - 步骤: stepname"
            elif (
                "开始处理任务" in message
                and "streamline_task_" in message
                and "步骤:" in message
            ):
                task_match = re.search(
                    r"开始处理任务 streamline_task_\d+_(.*?) - 步骤: (\w+)", message
                )

                if task_match:
                    task_name = task_match.group(1).strip()
                    step_name = task_match.group(2)

                    # 将步骤名映射到步骤ID
                    step_name_to_id = {
                        "preprocess_subtitle": 0,
                        "separate_media": 1,
                        "generate_reference_audio": 2,
                        "generate_tts": 3,
                        "align_audio": 4,
                        "generate_aligned_srt": 5,
                        "process_video_speed": 6,
                        "merge_audio_video": 7,
                    }

                    step_id = step_name_to_id.get(step_name)
                    if step_id is not None:
                        self.update_task_step_status(task_name, step_id, "processing")
                    return

            # 3. 解析各步骤的自定义开始日志（处理器内部日志）
            # 这些日志可能不包含完整的task_id，需要推断当前正在处理的任务
            elif "开始" in message:
                step_keywords = {
                    "开始预处理字幕": 0,
                    "开始媒体分离": 1,
                    "开始生成参考音频": 2,
                    "开始TTS生成": 3,
                    "开始音频对齐": 4,
                    "开始生成对齐字幕": 5,
                    "开始视频速度调整": 6,
                    "开始音视频合并": 7,
                }

                # 检查是否匹配任何步骤关键词
                for keyword, step_id in step_keywords.items():
                    if keyword in message:
                        # 如果消息包含streamline_task_，提取任务名
                        if "streamline_task_" in message:
                            task_match = re.search(
                                r"streamline_task_\d+_(.*?)(?:\s|:|$)", message
                            )
                            if task_match:
                                task_name = task_match.group(1).strip()
                                self.update_task_step_status(
                                    task_name, step_id, "processing"
                                )
                        else:
                            # 如果没有task_id，更新所有可能正在此步骤的任务
                            self.update_current_step_tasks_status(step_id, "processing")
                        return

                # 4. 通用的开始处理检测（包含streamline_task_的其他开始日志）
                if "streamline_task_" in message:
                    task_match = re.search(
                        r"streamline_task_\d+_(.*?)(?:\s|:|$)", message
                    )
                    if task_match:
                        task_name = task_match.group(1).strip()

                        # 根据消息内容推断步骤
                        if any(word in message for word in ["TTS生成", "generate_tts"]):
                            self.update_task_step_status(task_name, 3, "processing")
                        elif any(
                            word in message
                            for word in ["参考音频", "generate_reference_audio"]
                        ):
                            self.update_task_step_status(task_name, 2, "processing")
                        elif any(
                            word in message for word in ["媒体分离", "separate_media"]
                        ):
                            self.update_task_step_status(task_name, 1, "processing")
                        elif any(
                            word in message
                            for word in ["字幕预处理", "preprocess_subtitle"]
                        ):
                            self.update_task_step_status(task_name, 0, "processing")
                        elif any(
                            word in message for word in ["音频对齐", "align_audio"]
                        ):
                            self.update_task_step_status(task_name, 4, "processing")
                        elif any(
                            word in message
                            for word in ["对齐字幕", "generate_aligned_srt"]
                        ):
                            self.update_task_step_status(task_name, 5, "processing")
                        elif any(
                            word in message
                            for word in ["视频调速", "视频速度", "process_video_speed"]
                        ):
                            self.update_task_step_status(task_name, 6, "processing")
                        elif any(
                            word in message for word in ["合并", "merge_audio_video"]
                        ):
                            self.update_task_step_status(task_name, 7, "processing")
                    return

            # 5. 简化的开始处理检测 - 检测TTS相关日志
            elif (
                "开始TTS生成" in message or "gradio_api" in message.lower()
            ) and hasattr(self, "status_table"):
                # 没有具体任务名的情况下，尝试更新所有正在第3步的任务
                self.update_processing_tasks_status()
                return

            # 6. 解析任务完成的日志
            # 格式: "任务 streamline_task_XXX_taskname 步骤 stepname 处理成功"
            elif "处理成功" in message and "streamline_task_" in message:
                import re

                task_match = re.search(
                    r"任务 streamline_task_\d+_(.*?) 步骤 (\w+) 处理成功", message
                )

                if task_match:
                    task_name = task_match.group(1).strip()
                    step_name = task_match.group(2)

                    # 将步骤名映射到步骤ID
                    step_name_to_id = {
                        "preprocess_subtitle": 0,
                        "separate_media": 1,
                        "generate_reference_audio": 2,
                        "generate_tts": 3,
                        "align_audio": 4,
                        "generate_aligned_srt": 5,
                        "process_video_speed": 6,
                        "merge_audio_video": 7,
                    }

                    step_id = step_name_to_id.get(step_name)
                    if step_id is not None:
                        self.update_task_step_status(task_name, step_id, "completed")
                return

            # 7. 解析步骤完成的日志（处理器内部完成日志）
            elif any(
                keyword in message
                for keyword in [
                    "字幕预处理完成",
                    "媒体分离完成",
                    "参考音频生成完成",
                    "TTS生成完成",
                    "音频对齐完成",
                    "对齐字幕生成完成",
                    "视频速度调整完成",
                    "音视频合并完成",
                ]
            ):
                # 这些完成日志通常紧跟在处理日志之后，可以用来确认完成状态
                # 但由于没有task_id，我们依赖前面的基类完成日志来更新状态
                return

            # 8. 解析任务失败的日志
            # 格式: "任务 streamline_task_XXX_taskname 在队列 step_X_stepname 处理失败"
            elif "处理失败" in message and "streamline_task_" in message:
                import re

                task_match = re.search(
                    r"任务 streamline_task_\d+_(.*?) 在队列 step_(\d+)_", message
                )

                if task_match:
                    task_name = task_match.group(1).strip()
                    step_id = int(task_match.group(2))
                    self.update_task_step_status(task_name, step_id, "failed")
                return

        except Exception as e:
            # 不要让日志解析错误影响GUI运行
            pass

    def update_current_step_tasks_status(self, step_id: int, status: str):
        """更新指定步骤的所有相关任务状态 - 用于处理没有task_id的日志"""
        try:
            if not hasattr(self, "status_table"):
                return

            # 查找所有可能正在处理此步骤的任务
            for row in range(self.status_table.rowCount()):
                # 检查前一个步骤是否已完成，当前步骤是否未开始
                if step_id > 0:
                    prev_step_item = self.status_table.item(
                        row, step_id
                    )  # 前一步的列索引
                    curr_step_item = self.status_table.item(
                        row, step_id + 1
                    )  # 当前步的列索引

                    # 如果前一步已完成，当前步未开始，则更新当前步为处理中
                    if (
                        prev_step_item
                        and prev_step_item.text() == "✅"
                        and curr_step_item
                        and curr_step_item.text() == "⏸️"
                    ):

                        video_item = self.status_table.item(row, 0)
                        if video_item:
                            video_name = Path(video_item.text()).stem
                            self.update_task_step_status(video_name, step_id, status)
                else:
                    # 步骤0的情况，直接更新所有未开始的第一步
                    first_step_item = self.status_table.item(row, 1)  # 步骤1的列索引
                    if first_step_item and first_step_item.text() == "⏸️":
                        video_item = self.status_table.item(row, 0)
                        if video_item:
                            video_name = Path(video_item.text()).stem
                            self.update_task_step_status(video_name, step_id, status)

        except Exception as e:
            self.logger.error(f"更新当前步骤任务状态失败: {e}")

    def update_processing_tasks_status(self):
        """更新正在处理的任务状态 - 当无法从日志中提取具体任务名时使用"""
        try:
            if not hasattr(self, "status_table"):
                return

            # 查找已完成前3个步骤但第4步还未开始的任务，将其第4步设为处理中
            for row in range(self.status_table.rowCount()):
                step2_item = self.status_table.item(row, 3)  # 步骤2 (列索引3)
                step3_item = self.status_table.item(row, 4)  # 步骤3 (列索引4)

                # 如果步骤2已完成，步骤3还未开始，则设置步骤3为处理中
                if (
                    step2_item
                    and step2_item.text() == "✅"
                    and step3_item
                    and step3_item.text() == "⏸️"
                ):

                    video_item = self.status_table.item(row, 0)
                    if video_item:
                        video_name = Path(video_item.text()).stem
                        self.update_task_step_status(video_name, 3, "processing")

        except Exception as e:
            pass

    def test_status_update(self):
        """测试状态更新功能"""
        # 手动测试状态更新
        if hasattr(self, "status_table") and self.status_table.rowCount() > 0:
            # 获取第一行的视频文件名作为测试
            video_item = self.status_table.item(0, 0)
            if video_item:
                video_name = Path(video_item.text()).stem
                # 设置步骤3为处理中状态
                self.update_task_step_status(video_name, 3, "processing")

    def setup_logging(self):
        """设置日志"""
        # 创建日志处理器
        log_handler = LogHandler(self)
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # 添加到根日志记录器
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.INFO)  # 修改为INFO级别显示主要日志

    def on_mode_changed(self):
        """模式切换处理"""
        if self.single_mode_radio.isChecked():
            self.current_mode = "single"
            self.single_file_panel.show()
            self.batch_panel.hide()
        else:
            self.current_mode = "batch"
            self.single_file_panel.hide()
            self.batch_panel.show()

    def browse_video_file(self):
        """浏览选择视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v *.webm);;所有文件 (*)",
        )

        if file_path:
            self.video_path_edit.setText(file_path)
            self.current_video_path = file_path

            # 尝试自动匹配字幕文件
            self.auto_match_subtitle()

    def browse_subtitle_file(self):
        """浏览选择字幕文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择字幕文件",
            "",
            "字幕文件 (*.srt *.ass *.ssa *.sub *.vtt);;所有文件 (*)",
        )

        if file_path:
            self.subtitle_path_edit.setText(file_path)
            self.current_subtitle_path = file_path

    def browse_folder(self):
        """浏览选择文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择包含视频和字幕的文件夹", ""
        )

        if folder_path:
            self.folder_path_edit.setText(folder_path)
            self.current_folder_path = folder_path

    def scan_folder(self):
        """扫描文件夹中的视频字幕对"""
        if not hasattr(self, "current_folder_path") or not self.current_folder_path:
            QMessageBox.warning(self, "警告", "请先选择文件夹！")
            return

        try:
            # 递归搜索视频和字幕文件
            pairs = VideoSubtitleMatcher.find_video_subtitle_pairs(
                self.current_folder_path
            )

            if not pairs:
                QMessageBox.information(self, "信息", "未找到任何视频文件！")
                return

            self.video_subtitle_pairs = pairs
            self.update_file_table()

            QMessageBox.information(
                self,
                "扫描完成",
                f"找到 {len(pairs)} 个有完全匹配字幕的视频文件。\n（已自动搜索所有子文件夹，只显示文件名完全匹配的视频-字幕对）",
            )

        except Exception as e:
            error_msg = f"扫描文件夹失败: {str(e)}\n详细错误:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "错误", error_msg)

    def update_file_table(self):
        """更新文件列表表格"""
        self.file_table.setRowCount(len(self.video_subtitle_pairs))

        for i, (video_path, subtitle_path) in enumerate(self.video_subtitle_pairs):
            # 视频文件名
            video_name = Path(video_path).name
            video_item = QTableWidgetItem(video_name)
            video_item.setToolTip(video_path)
            video_item.setFlags(
                video_item.flags() & ~Qt.ItemIsEditable
            )  # 设置为不可编辑
            self.file_table.setItem(i, 0, video_item)

            # 字幕文件名（现在一定有匹配的字幕）
            subtitle_name = Path(subtitle_path).name
            subtitle_item = QTableWidgetItem(subtitle_name)
            subtitle_item.setToolTip(subtitle_path)
            subtitle_item.setFlags(
                subtitle_item.flags() & ~Qt.ItemIsEditable
            )  # 设置为不可编辑
            self.file_table.setItem(i, 1, subtitle_item)

            # 状态（现在一定是就绪状态）
            status_item = QTableWidgetItem("就绪")
            status_item.setForeground(QColor("#198754"))  # 绿色
            status_item.setFlags(
                status_item.flags() & ~Qt.ItemIsEditable
            )  # 设置为不可编辑
            self.file_table.setItem(i, 2, status_item)

            # 选择框
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # 现在所有项都有字幕，默认全选
            self.file_table.setCellWidget(i, 3, checkbox)

    def update_task_step_status(
        self, task_name: str, step_id: int, status: str, message: str = "", force_update: bool = False
    ):
        """
        更新特定任务的步骤状态（线程安全版本）

        Args:
            task_name: 任务名称（通常是视频文件名，不含扩展名）
            step_id: 步骤ID (0-7)
            status: 状态 ("processing", "completed", "failed")
            message: 状态消息
            force_update: 是否强制更新，忽略状态降级保护
        """
        # 使用线程锁保护状态更新
        with self._status_update_lock:
            return self._update_task_step_status_internal(task_name, step_id, status, message, force_update)

    def _update_task_step_status_internal(
        self, task_name: str, step_id: int, status: str, message: str = "", force_update: bool = False
    ):
        """内部状态更新方法"""
        try:
            # 生成新的时间戳和序列号
            current_time = time.time()
            self._global_sequence += 1
            new_sequence = self._global_sequence
            
            step_key = (task_name, step_id)
            
            # 检查是否应该跳过此次更新（时间戳检查）
            if not force_update:
                last_timestamp = self._step_timestamps.get(step_key, 0)
                last_sequence = self._step_sequence.get(step_key, 0)
                
                # 如果收到的是旧状态（基于时间戳），跳过更新
                if current_time < last_timestamp + 0.1:  # 100ms内的重复更新视为可能的重复信号
                    if new_sequence <= last_sequence:
                        self.logger.debug(
                            f"跳过旧状态更新: task_name={task_name}, step={step_id}, "
                            f"status={status}, seq={new_sequence} <= {last_sequence}"
                        )
                        return
            
            # 更新时间戳和序列号
            self._step_timestamps[step_key] = current_time
            self._step_sequence[step_key] = new_sequence

            # 查找任务在状态表格中的行索引
            task_row = -1
            matched_method = "unknown"

            # 先尝试完全精确匹配（最优先）
            for i in range(self.status_table.rowCount()):
                video_item = self.status_table.item(i, 0)
                if video_item:
                    video_name = Path(video_item.text()).stem  # 去掉扩展名
                    if video_name == task_name:
                        task_row = i
                        matched_method = "exact"
                        break

            # 如果精确匹配失败，尝试去掉特殊字符后精确匹配（第二优先）
            if task_row == -1:
                clean_task_name = re.sub(
                    r"[^\w\u4e00-\u9fff]", "", task_name
                )  # 只保留字母、数字、中文
                for i in range(self.status_table.rowCount()):
                    video_item = self.status_table.item(i, 0)
                    if video_item:
                        video_name = Path(video_item.text()).stem
                        clean_video_name = re.sub(r"[^\w\u4e00-\u9fff]", "", video_name)
                        if clean_task_name == clean_video_name:
                            task_row = i
                            matched_method = "clean_exact"
                            break

            # 最后才尝试包含匹配（容易误匹配，降低优先级）
            if task_row == -1:
                for i in range(self.status_table.rowCount()):
                    video_item = self.status_table.item(i, 0)
                    if video_item:
                        video_name = Path(video_item.text()).stem
                        # 只有在task_name较长时才使用包含匹配，避免误匹配
                        if len(task_name) > 5 and (
                            task_name in video_name or video_name in task_name
                        ):
                            task_row = i
                            matched_method = "contains"
                            break

            if task_row == -1:
                self.logger.warning(f"未找到任务 {task_name} 在状态表格中的对应行")
                # 添加调试信息，显示所有可用的视频文件名
                available_names = []
                for i in range(self.status_table.rowCount()):
                    item = self.status_table.item(i, 0)
                    if item:
                        available_names.append(Path(item.text()).stem)
                self.logger.debug(f"可用的视频文件名: {available_names}")
                return

            self.logger.debug(
                f"找到匹配行: task_name={task_name}, task_row={task_row}, method={matched_method}, seq={new_sequence}"
            )

            # 更新步骤状态（列索引 1-8）
            step_col = 1 + step_id
            step_item = self.status_table.item(task_row, step_col)

            if not step_item:
                step_item = QTableWidgetItem()
                step_item.setFlags(
                    step_item.flags() & ~Qt.ItemIsEditable
                )  # 设置为不可编辑
                self.status_table.setItem(task_row, step_col, step_item)

            # 检查当前状态和新状态
            current_status_icon = step_item.text()
            new_status_icon = ""

            # 设置状态图标和颜色
            if status == "processing":
                new_status_icon = "🔄"  # 处理中
                color = QColor("#fd7e14")  # 橙色
                tooltip = f"步骤{step_id + 1}: 处理中"
            elif status == "completed":
                new_status_icon = "✅"  # 完成
                color = QColor("#198754")  # 绿色
                tooltip = f"步骤{step_id + 1}: 已完成"
            elif status == "failed":
                new_status_icon = "❌"  # 失败
                color = QColor("#dc3545")  # 红色
                tooltip = f"步骤{step_id + 1}: 失败"

                # 当步骤失败时，将后续步骤重置为未开始状态
                self._reset_subsequent_steps(task_row, step_id)
            else:
                new_status_icon = "⏸️"  # 未开始
                color = QColor("#6c757d")  # 灰色
                tooltip = f"步骤{step_id + 1}: 未开始"

            # 改进的状态更新判断逻辑
            should_update = False

            if current_status_icon != new_status_icon:
                should_update = self._should_update_status(
                    current_status_icon, new_status_icon, force_update, task_name, step_id, message
                )

            if should_update:
                step_item.setText(new_status_icon)
                step_item.setForeground(color)

                if message:
                    tooltip += f" - {message}"

                step_item.setToolTip(tooltip)
                step_item.setTextAlignment(Qt.AlignCenter)

                # 更新整体状态
                self.update_overall_task_status(task_row)
                
                # 强制刷新表格显示
                if hasattr(self, 'status_table'):
                    self.status_table.update()
                    self.status_table.viewport().update()
                    # 确保特定单元格的更新
                    model_index = self.status_table.model().index(task_row, step_col)
                    self.status_table.update(model_index)
                    
                self.logger.debug(
                    f"状态已更新: task_name={task_name}, step={step_id}, "
                    f"{current_status_icon} -> {new_status_icon}, seq={new_sequence}"
                )
            else:
                self.logger.debug(
                    f"状态无变化或被保护，跳过更新: task_name={task_name}, step={step_id}, "
                    f"current={current_status_icon}, new={new_status_icon}, seq={new_sequence}"
                )

        except Exception as e:
            self.logger.error(f"更新任务步骤状态失败: {e}")

    def _should_update_status(self, current_icon: str, new_icon: str, force_update: bool, 
                             task_name: str, step_id: int, message: str) -> bool:
        """
        判断是否应该更新状态（改进的逻辑）
        
        Args:
            current_icon: 当前状态图标
            new_icon: 新状态图标  
            force_update: 是否强制更新
            task_name: 任务名称
            step_id: 步骤ID
            message: 状态消息
        
        Returns:
            是否应该更新状态
        """
        if force_update:
            self.logger.debug(f"强制更新状态: {task_name} 步骤{step_id}, {current_icon} -> {new_icon}")
            return True
        
        # 状态优先级：✅完成(3) > ❌失败(2) > 🔄处理中(1) > ⏸️未开始(0)
        status_priority = {"✅": 3, "❌": 2, "🔄": 1, "⏸️": 0}
        current_priority = status_priority.get(current_icon, 0)
        new_priority = status_priority.get(new_icon, 0)
        
        # 允许状态更新的情况
        update_allowed = False
        reason = ""
        
        if new_priority >= current_priority:
            # 正常的状态升级
            update_allowed = True
            reason = "状态升级"
        elif new_icon == "❌":
            # 任何状态都可以变为失败
            update_allowed = True
            reason = "失败状态"
        elif current_icon == "✅" and new_icon == "🔄" and "重试" in message:
            # 允许重试时从完成回退到处理中
            update_allowed = True
            reason = "重试回退"
        elif current_icon == "❌" and new_icon == "🔄":
            # 允许从失败回退到处理中（重试）
            update_allowed = True
            reason = "失败重试"
        elif current_icon == "⏸️":
            # 从未开始可以变为任何状态
            update_allowed = True
            reason = "初始状态"
        elif abs(current_priority - new_priority) <= 1:
            # 相邻状态间允许变化（处理并发更新）
            update_allowed = True
            reason = "相邻状态"
        
        if update_allowed:
            self.logger.debug(
                f"状态更新: task_name={task_name}, step={step_id}, "
                f"{current_icon} -> {new_icon} ({reason})"
            )
        else:
            self.logger.warning(
                f"阻止状态降级: task_name={task_name}, step={step_id}, "
                f"{current_icon} -> {new_icon}, message='{message}'"
            )
        
        return update_allowed

    def update_step_status_direct(
        self, task_id: str, step_id: int, status: str, message: str = ""
    ):
        """直接更新状态，不依赖日志解析（最高优先级）"""
        try:
            # 从task_id提取任务名: streamline_task_XXX_videoname
            parts = task_id.split("_")
            if len(parts) >= 3:
                task_name = "_".join(parts[2:])  # 取videoname部分
            else:
                task_name = task_id

            # 添加调试信息
            self.logger.debug(
                f"直接状态更新: task_id={task_id}, task_name={task_name}, step_id={step_id}, status={status}"
            )

            # 直接状态更新通常有更高权威性，使用force_update=True
            force_update = True
            if "重试" in message:
                # 重试消息明确表示需要强制更新
                force_update = True

            # 立即更新状态（线程安全版本）
            self.update_task_step_status(task_name, step_id, status, message, force_update)

            # 强制刷新界面（确保状态立即显示）
            if hasattr(self, 'status_table'):
                self.status_table.viewport().update()
                
            # 处理应用事件，确保界面更新
            QApplication.processEvents()

        except Exception as e:
            self.logger.error(f"直接状态更新失败: {e}")
            self.logger.debug(
                f"失败的参数: task_id={task_id}, step_id={step_id}, status={status}"
            )

    def _reset_subsequent_steps(self, task_row: int, failed_step_id: int):
        """
        重置失败步骤之后的所有步骤为未开始状态

        Args:
            task_row: 任务在表格中的行索引
            failed_step_id: 失败的步骤ID
        """
        try:
            step_names = [
                "字幕预处理",
                "媒体分离",
                "参考音频",
                "TTS生成",
                "音频对齐",
                "对齐字幕",
                "视频调速",
                "合并输出",
            ]

            # 重置失败步骤之后的所有步骤
            for step_id in range(failed_step_id + 1, 8):
                step_col = 1 + step_id
                step_item = self.status_table.item(task_row, step_col)

                if not step_item:
                    step_item = QTableWidgetItem()
                    step_item.setFlags(
                        step_item.flags() & ~Qt.ItemIsEditable
                    )  # 设置为不可编辑
                    self.status_table.setItem(task_row, step_col, step_item)

                step_item.setText("⏸️")  # 未开始
                step_item.setForeground(QColor("#6c757d"))  # 灰色
                step_item.setToolTip(
                    f"步骤{step_id + 1}: {step_names[step_id]} - 未开始"
                )
                step_item.setTextAlignment(Qt.AlignCenter)

        except Exception as e:
            self.logger.error(f"重置后续步骤状态失败: {e}")

    def update_overall_task_status(self, row: int):
        """更新任务的整体状态"""
        try:
            # 检查所有步骤状态
            completed_steps = 0
            failed_steps = 0
            processing_steps = 0

            for step_idx in range(8):
                step_col = 1 + step_idx
                step_item = self.status_table.item(row, step_col)
                if step_item:
                    text = step_item.text()
                    if text == "✅":
                        completed_steps += 1
                    elif text == "❌":
                        failed_steps += 1
                    elif text == "🔄":
                        processing_steps += 1

            # 更新整体状态列（列索引 9）
            status_item = self.status_table.item(row, 9)
            if not status_item:
                status_item = QTableWidgetItem()
                status_item.setFlags(
                    status_item.flags() & ~Qt.ItemIsEditable
                )  # 设置为不可编辑
                self.status_table.setItem(row, 9, status_item)

            # 更新进度列（列索引 10）
            progress_item = self.status_table.item(row, 10)
            if not progress_item:
                progress_item = QTableWidgetItem()
                progress_item.setFlags(
                    progress_item.flags() & ~Qt.ItemIsEditable
                )  # 设置为不可编辑
                self.status_table.setItem(row, 10, progress_item)

            # 设置状态和进度
            if failed_steps > 0:
                status_item.setText("失败")
                status_item.setForeground(QColor("#dc3545"))  # 红色
                progress_item.setText(f"{completed_steps}/8")
                progress_item.setForeground(QColor("#dc3545"))
            elif completed_steps == 8:
                status_item.setText("完成")
                status_item.setForeground(QColor("#198754"))  # 绿色
                progress_item.setText("8/8")
                progress_item.setForeground(QColor("#198754"))
            elif processing_steps > 0:
                status_item.setText("处理中")
                status_item.setForeground(QColor("#fd7e14"))  # 橙色
                progress_item.setText(f"{completed_steps}/8")
                progress_item.setForeground(QColor("#fd7e14"))
            elif completed_steps > 0:
                status_item.setText("进行中")
                status_item.setForeground(QColor("#0d6efd"))  # 蓝色
                progress_item.setText(f"{completed_steps}/8")
                progress_item.setForeground(QColor("#0d6efd"))
            else:
                status_item.setText("就绪")
                status_item.setForeground(QColor("#198754"))  # 绿色
                progress_item.setText("0/8")
                progress_item.setForeground(QColor("#6c757d"))

            progress_item.setTextAlignment(Qt.AlignCenter)

        except Exception as e:
            self.logger.error(f"更新整体任务状态失败: {e}")

    def initialize_status_table(self, video_subtitle_pairs: List[Tuple[str, str]]):
        """初始化处理状态表格，加载已有缓存状态"""
        try:
            self.status_table.setRowCount(len(video_subtitle_pairs))

            for i, (video_path, subtitle_path) in enumerate(video_subtitle_pairs):
                # 视频文件名
                video_name = Path(video_path).name
                video_item = QTableWidgetItem(video_name)
                video_item.setToolTip(video_path)
                video_item.setFlags(
                    video_item.flags() & ~Qt.ItemIsEditable
                )  # 设置为不可编辑
                self.status_table.setItem(i, 0, video_item)

                # 步骤名称
                step_names = [
                    "字幕预处理",
                    "媒体分离",
                    "参考音频",
                    "TTS生成",
                    "音频对齐",
                    "对齐字幕",
                    "视频调速",
                    "合并输出",
                ]

                # 尝试加载缓存状态
                cached_status = self._load_cached_task_status(video_path)

                # 初始化8个步骤状态列（列索引 1-8）
                for step_idx in range(8):
                    if cached_status and step_idx in cached_status:
                        # 从缓存恢复状态
                        step_status = cached_status[step_idx]
                        if step_status == "completed":
                            step_item = QTableWidgetItem("✅")
                            step_item.setForeground(QColor("#198754"))  # 绿色
                            tooltip = (
                                f"步骤{step_idx + 1}: {step_names[step_idx]} - 已完成"
                            )
                        elif step_status == "failed":
                            step_item = QTableWidgetItem("❌")
                            step_item.setForeground(QColor("#dc3545"))  # 红色
                            tooltip = (
                                f"步骤{step_idx + 1}: {step_names[step_idx]} - 失败"
                            )
                        elif step_status == "processing":
                            step_item = QTableWidgetItem("🔄")
                            step_item.setForeground(QColor("#fd7e14"))  # 橙色
                            tooltip = (
                                f"步骤{step_idx + 1}: {step_names[step_idx]} - 处理中"
                            )
                        else:
                            step_item = QTableWidgetItem("⏸️")
                            step_item.setForeground(QColor("#6c757d"))  # 灰色
                            tooltip = (
                                f"步骤{step_idx + 1}: {step_names[step_idx]} - 未开始"
                            )
                    else:
                        # 默认未开始状态
                        step_item = QTableWidgetItem("⏸️")
                        step_item.setForeground(QColor("#6c757d"))  # 灰色
                        tooltip = f"步骤{step_idx + 1}: {step_names[step_idx]} - 未开始"

                    step_item.setToolTip(tooltip)
                    step_item.setTextAlignment(Qt.AlignCenter)
                    step_item.setFlags(
                        step_item.flags() & ~Qt.ItemIsEditable
                    )  # 设置为不可编辑
                    self.status_table.setItem(i, 1 + step_idx, step_item)

                # 更新整体状态
                self.update_overall_task_status(i)

        except Exception as e:
            self.logger.error(f"初始化状态表格失败: {e}")

    def _load_cached_task_status(self, video_path: str) -> Optional[Dict[int, str]]:
        """
        加载缓存中的任务状态（改进版本）

        Args:
            video_path: 视频文件路径

        Returns:
            步骤状态字典 {step_id: status} 或 None
        """
        try:
            # 构建缓存文件路径 - 需要与pipeline中的路径一致
            video_path_obj = Path(video_path)
            clean_name = sanitize_filename(video_path_obj.stem)
            output_dir = video_path_obj.parent / "outputs" / clean_name

            # 使用pipeline_cache命名规则
            cache_file = output_dir / f"{video_path_obj.stem}_pipeline_cache.json"

            self.logger.debug(f"尝试加载缓存文件: {cache_file}")

            if not cache_file.exists():
                self.logger.debug(f"缓存文件不存在: {cache_file}")
                return None

            # 加载缓存
            cache_manager = TaskCacheManager()
            cache_data = cache_manager.load_task_cache(cache_file, video_path)

            if not cache_data:
                self.logger.debug("缓存数据为空")
                return None

            task_data = cache_data.get("task", {})
            step_details = task_data.get("step_details", {})
            step_results = task_data.get("step_results", {})

            self.logger.debug(f"加载的步骤详情: {list(step_details.keys())}")
            self.logger.debug(f"加载的步骤结果: {list(step_results.keys())}")

            # 转换为GUI需要的格式（改进版本）
            status_map = {}

            # 状态优先级映射和权重
            # 权重: step_results > step_details（结果比详情更权威）
            result_weight = 100
            detail_weight = 50

            # 临时状态收集 {step_id: [(status, weight, source)]}
            temp_status = {}

            # 从step_results获取状态（更权威）
            for step_id_str, result in step_results.items():
                try:
                    step_id = int(step_id_str)
                    if result.get("success", False) and not result.get("partial_success", False):
                        # 完全成功
                        status = "completed"
                        weight = result_weight + 20  # 完全成功权重最高
                    elif result.get("partial_success", False):
                        # 部分成功，视情况而定
                        status = "failed"
                        weight = result_weight + 10
                    elif "error" in result or result.get("success") is False:
                        # 明确失败
                        status = "failed"
                        weight = result_weight + 15
                    else:
                        # 未知状态，跳过
                        continue
                    
                    if step_id not in temp_status:
                        temp_status[step_id] = []
                    temp_status[step_id].append((status, weight, "step_results"))
                except (ValueError, TypeError):
                    continue

            # 从step_details获取状态
            for step_id_str, detail in step_details.items():
                try:
                    step_id = int(step_id_str)
                    detail_status = detail.get("status", "").lower()
                    
                    # 状态格式转换和权重分配
                    if detail_status == "completed":
                        status = "completed"
                        weight = detail_weight + 10
                    elif detail_status == "failed":
                        status = "failed"
                        weight = detail_weight + 8
                    elif detail_status in ["processing", "running"]:
                        # 处理中状态，可能是中断的任务，权重较低
                        status = "processing"
                        weight = detail_weight + 5
                    elif detail_status in ["pending", "waiting"]:
                        # 待处理状态
                        status = "pending"
                        weight = detail_weight + 2
                    else:
                        # 未知状态，跳过
                        continue
                    
                    if step_id not in temp_status:
                        temp_status[step_id] = []
                    temp_status[step_id].append((status, weight, "step_details"))
                except (ValueError, TypeError):
                    continue

            # 选择最高权重的状态
            for step_id, status_list in temp_status.items():
                # 按权重排序，选择权重最高的
                status_list.sort(key=lambda x: x[1], reverse=True)
                best_status, best_weight, source = status_list[0]
                
                # 特殊处理：如果有processing状态但没有对应的结果，可能是中断的任务
                has_processing = any(s[0] == "processing" for s in status_list)
                has_result = any(s[2] == "step_results" for s in status_list)
                
                if has_processing and not has_result:
                    # 有处理中状态但没有结果，可能是中断的任务，重置为pending
                    self.logger.debug(f"步骤 {step_id} 检测到中断状态，重置为pending")
                    best_status = "pending"
                
                status_map[step_id] = best_status
                self.logger.debug(
                    f"步骤 {step_id}: 选择状态 '{best_status}' (权重: {best_weight}, 来源: {source})"
                )

            # 验证状态序列的合理性
            status_map = self._validate_and_fix_status_sequence(status_map)

            self.logger.debug(f"最终状态映射: {status_map}")
            return status_map if status_map else None

        except Exception as e:
            self.logger.debug(f"加载缓存状态失败: {e}")
            return None

    def _validate_and_fix_status_sequence(self, status_map: Dict[int, str]) -> Dict[int, str]:
        """
        验证和修复状态序列的合理性
        
        Args:
            status_map: 原始状态映射
            
        Returns:
            修复后的状态映射
        """
        if not status_map:
            return status_map
        
        # 获取所有步骤，按顺序排序
        all_steps = sorted(status_map.keys())
        fixed_map = {}
        
        # 状态序列验证规则：
        # 1. 完成的步骤之后不应该有未开始的步骤（除非是失败后的重置）
        # 2. 处理中的步骤之前的步骤应该都是完成的
        # 3. 失败的步骤之后的步骤应该是未开始的
        
        last_completed = -1
        failed_step = -1
        
        for step_id in all_steps:
            status = status_map[step_id]
            
            if status == "completed":
                if failed_step != -1 and step_id > failed_step:
                    # 失败步骤之后的步骤不应该是完成状态
                    self.logger.debug(f"修复状态序列: 步骤 {step_id} 在失败步骤 {failed_step} 之后，重置为pending")
                    fixed_map[step_id] = "pending"
                else:
                    fixed_map[step_id] = "completed"
                    last_completed = step_id
            elif status == "failed":
                fixed_map[step_id] = "failed"
                if failed_step == -1:
                    failed_step = step_id
            elif status == "processing":
                if step_id > 0 and last_completed < step_id - 1:
                    # 处理中的步骤之前有未完成的步骤，可能不一致
                    self.logger.debug(
                        f"修复状态序列: 步骤 {step_id} 处理中，但前面有未完成步骤，重置为pending"
                    )
                    fixed_map[step_id] = "pending"
                else:
                    fixed_map[step_id] = "processing"
            else:  # pending 或其他
                fixed_map[step_id] = "pending"
        
        return fixed_map

    def select_all_files(self):
        """全选文件"""
        for i in range(self.file_table.rowCount()):
            checkbox = self.file_table.cellWidget(i, 3)
            if checkbox:
                checkbox.setChecked(True)

    def deselect_all_files(self):
        """全不选文件"""
        for i in range(self.file_table.rowCount()):
            checkbox = self.file_table.cellWidget(i, 3)
            if checkbox:
                checkbox.setChecked(False)

    def get_selected_pairs(self) -> List[Tuple[str, Optional[str]]]:
        """获取选中的视频字幕对"""
        selected = []
        for i in range(self.file_table.rowCount()):
            checkbox = self.file_table.cellWidget(i, 3)
            if checkbox and checkbox.isChecked():
                selected.append(self.video_subtitle_pairs[i])
        return selected

    def auto_match_subtitle(self):
        """自动匹配字幕文件"""
        if not self.current_video_path:
            return

        video_path = Path(self.current_video_path)
        video_dir = video_path.parent
        video_name = video_path.stem

        # 常见的字幕文件扩展名
        subtitle_extensions = [".srt", ".ass", ".ssa", ".sub", ".vtt"]

        for ext in subtitle_extensions:
            subtitle_file = video_dir / f"{video_name}{ext}"
            if subtitle_file.exists():
                self.subtitle_path_edit.setText(str(subtitle_file))
                self.current_subtitle_path = str(subtitle_file)
                return

    def start_processing(self):
        """开始处理 - 统一的入口方法"""
        if self.current_mode == "single":
            # 单文件模式：将单个文件转换为批量处理格式
            if not self.current_video_path:
                QMessageBox.warning(self, "警告", "请选择视频文件！")
                return

            if not Path(self.current_video_path).exists():
                QMessageBox.warning(self, "警告", "视频文件不存在！")
                return

            # 转换为批量处理格式（包含一个文件的列表）
            video_subtitle_pairs = [
                (
                    self.current_video_path,
                    self.current_subtitle_path if self.current_subtitle_path else None,
                )
            ]

            # 调用统一的批量处理方法
            self._start_unified_processing(video_subtitle_pairs, is_single_mode=True)
        else:
            # 批量模式：获取选中的文件列表
            selected_pairs = self.get_selected_pairs()
            if not selected_pairs:
                QMessageBox.warning(self, "警告", "请至少选择一个要处理的视频！")
                return

            # 调用统一的批量处理方法
            self._start_unified_processing(selected_pairs, is_single_mode=False)

    def _start_unified_processing(
        self, video_subtitle_pairs: List[Tuple[str, str]], is_single_mode: bool = False
    ):
        """统一的处理方法，支持单文件和批量模式"""
        # 初始化TTS处理器
        api_url = self.api_url_edit.text().strip() or "http://127.0.0.1:7860"
        self._initialize_tts_processor(api_url)

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        try:
            # 创建统一的批量处理工作线程
            self.worker_thread = StreamlineBatchDubbingWorkerThread(
                video_subtitle_pairs, True  # 默认从缓存恢复
            )

            # 连接信号 - 统一使用批量处理的信号处理
            self.worker_thread.batch_finished.connect(self._unified_processing_finished)

            # 初始化状态表格并连接日志信号（单文件和批量模式都显示）
            self.initialize_status_table(video_subtitle_pairs)
            self.worker_thread.log_message.connect(self.append_log_message)

            # 启动线程
            self.worker_thread.start()

        except Exception as e:
            error_msg = f"启动处理失败: {str(e)}\n详细错误:\n{traceback.format_exc()}"
            self.log_text.append(error_msg)
            QMessageBox.critical(self, "错误", error_msg)

            # 重置UI状态
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

    def _unified_processing_finished(
        self, success: bool, message: str, result: Dict[str, Any]
    ):
        """统一的处理完成回调"""
        try:
            self.log_text.append(f"\n处理完成: {message}")

            # 恢复UI状态
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

            # 根据当前模式显示相应的结果
            if self.current_mode == "single":
                self._handle_single_mode_result(success, message, result)
            else:
                self._handle_batch_mode_result(success, message, result)

        except Exception as e:
            self.logger.error(f"处理完成回调失败: {e}")

    def _handle_single_mode_result(
        self, success: bool, message: str, result: Dict[str, Any]
    ):
        """处理单文件模式的结果"""
        if success:
            QMessageBox.information(self, "完成", message)

            # 尝试打开输出目录
            if result and result.get("results"):
                first_result = result["results"][0]
                if first_result.get("output_dir"):
                    output_dir = Path(first_result["output_dir"])
                    if output_dir.exists():
                        try:
                            import subprocess

                            subprocess.run(["explorer", str(output_dir)], check=False)
                        except Exception as e:
                            self.logger.debug(f"打开输出目录失败: {e}")
        else:
            QMessageBox.critical(self, "处理失败", message)

    def _handle_batch_mode_result(
        self, success: bool, message: str, result: Dict[str, Any]
    ):
        """处理批量模式的结果"""
        if success:
            QMessageBox.information(self, "批量处理完成", message)
        else:
            QMessageBox.warning(self, "批量处理完成", message)

    def cancel_processing(self):
        """取消处理"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
            self.worker_thread.wait()

        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def show_cache_info(self):
        """显示缓存信息"""
        QMessageBox.information(self, "信息", "请通过日志输出查看缓存相关信息")

    def clear_cache(self):
        """清理缓存"""
        QMessageBox.information(
            self, "信息", "请手动删除相应的缓存文件，或重新处理文件"
        )

    def repair_cache(self):
        """修复缓存"""
        QMessageBox.information(self, "信息", "请通过重新处理文件来修复缓存")

    def clear_output_directories(self):
        """清理重复的输出目录"""
        reply = QMessageBox.question(
            self,
            "确认清理",
            "确定要清理重复的输出目录吗？\n\n此操作将：\n• 分析现有输出目录\n• 识别并删除重复目录\n• 保留最新的处理结果\n\n建议先备份重要数据！",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            try:
                import re
                import shutil
                from datetime import datetime
                from pathlib import Path

                # 查找可能的outputs目录
                current_dir = Path.cwd()
                outputs_dirs = []

                # 在当前目录和父目录中查找outputs目录
                for search_dir in [current_dir, current_dir.parent]:
                    potential_outputs = search_dir / "outputs"
                    if potential_outputs.exists() and potential_outputs.is_dir():
                        outputs_dirs.append(potential_outputs)

                if not outputs_dirs:
                    QMessageBox.information(self, "信息", "未找到输出目录")
                    return

                # 分析和清理每个outputs目录
                total_cleaned = 0
                cleanup_log = []

                for outputs_dir in outputs_dirs:
                    # 获取所有子目录
                    subdirs = [d for d in outputs_dir.iterdir() if d.is_dir()]

                    if not subdirs:
                        continue

                    # 按照统一的文件名清理逻辑分组
                    def _sanitize_filename(filename: str) -> str:
                        sanitized = re.sub(r'[<>:"/\\\\|?*]', "_", filename)
                        sanitized = re.sub(r"[@#&%=+]", "_", sanitized)
                        sanitized = re.sub(r"[.\\-\\s]", "_", sanitized)
                        sanitized = re.sub(r"_+", "_", sanitized)
                        sanitized = sanitized.strip("_")
                        if not sanitized:
                            sanitized = "unnamed"
                        return sanitized

                    # 将目录按照清理后的名称分组
                    grouped_dirs = {}
                    for subdir in subdirs:
                        clean_name = _sanitize_filename(subdir.name)
                        if clean_name not in grouped_dirs:
                            grouped_dirs[clean_name] = []
                        grouped_dirs[clean_name].append(subdir)

                    # 清理重复目录
                    for clean_name, dirs in grouped_dirs.items():
                        if len(dirs) > 1:
                            # 按修改时间排序，保留最新的
                            dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
                            keep_dir = dirs[0]

                            for remove_dir in dirs[1:]:
                                try:
                                    shutil.rmtree(remove_dir)
                                    cleanup_log.append(f"删除: {remove_dir.name}")
                                    total_cleaned += 1
                                except Exception as e:
                                    cleanup_log.append(
                                        f"删除失败 {remove_dir.name}: {e}"
                                    )

                            cleanup_log.append(f"保留: {keep_dir.name} (最新)")

                # 显示清理结果
                if total_cleaned > 0:
                    result_msg = f"清理完成！删除了 {total_cleaned} 个重复目录。\\n\\n详细信息:\\n" + "\\n".join(
                        cleanup_log[-10:]
                    )  # 只显示最后10条
                    QMessageBox.information(self, "清理完成", result_msg)
                else:
                    QMessageBox.information(self, "清理完成", "未发现重复的输出目录")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"清理输出目录失败: {str(e)}")

    def _initialize_tts_processor(self, api_url: str):
        """初始化TTS处理器"""
        try:
            # 处理用户输入的API URL格式
            processed_url = self._process_api_url(api_url)

            self.log_message.emit(f"正在初始化TTS处理器，API地址: {processed_url}")
            initialize_tts_processor(processed_url)
            self.log_message.emit("TTS处理器初始化成功")
        except Exception as e:
            error_msg = f"TTS处理器初始化失败: {str(e)}"
            self.log_message.emit(error_msg)
            QMessageBox.warning(self, "警告", error_msg)

    def _process_api_url(self, api_url: str) -> str:
        """处理API URL格式，确保格式正确"""
        if not api_url or api_url.strip() == "":
            return "http://127.0.0.1:7860"

        api_url = api_url.strip()

        # 如果用户只输入了IP:端口格式，自动添加http://前缀
        if not api_url.startswith(("http://", "https://")):
            # 检查是否是IP:端口格式
            if ":" in api_url and not api_url.startswith("//"):
                api_url = f"http://{api_url}"
            else:
                # 如果格式不明确，使用默认值
                api_url = "http://127.0.0.1:7860"

        return api_url

    def clear_log(self):
        """清空日志"""
        self.log_text.clear()

    def save_log(self):
        """保存日志"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存日志",
            f"dubbingx_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "文本文件 (*.txt);;所有文件 (*)",
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "成功", f"日志已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存日志失败: {str(e)}")


def main():
    """主函数"""
    try:
        app = QApplication(sys.argv)

        # 设置应用程序属性
        app.setApplicationName("DubbingX")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("DubbingX Team")

        # 设置应用程序退出时的行为
        app.setQuitOnLastWindowClosed(True)

        # 创建主窗口
        window = DubbingGUI()
        window.show()

        # 设置简单的信号处理
        import signal as sig

        def signal_handler(signum, frame):
            print(f"\n收到系统信号 {signum}，立即退出...")
            os._exit(0)

        sig.signal(sig.SIGINT, signal_handler)
        if hasattr(sig, "SIGTERM"):
            sig.signal(sig.SIGTERM, signal_handler)

        # 运行应用程序
        app.exec()

    except Exception as e:
        print(f"应用程序启动失败: {e}")

    # 最终确保退出
    print("主函数结束，强制退出进程")
    os._exit(0)


if __name__ == "__main__":
    main()
