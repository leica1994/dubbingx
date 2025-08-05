"""
DubbingX GUI 最终版 - 智能视频配音系统图形界面
简洁清晰的界面设计，优化可读性
"""

import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox,
                               QFileDialog, QGridLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QRadioButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget,
                               QTextEdit, QVBoxLayout, QWidget)

from core.audio_align_processor import (align_audio_with_subtitles,
                                        generate_aligned_srt,
                                        process_video_speed_adjustment)
from core.dubbing_pipeline import DubbingPipeline, StreamlinePipeline
from core.media_processor import (generate_reference_audio, merge_audio_video,
                                  separate_media)
from core.subtitle.subtitle_processor import (convert_subtitle,
                                              sync_srt_timestamps_to_ass)
from core.subtitle_preprocessor import preprocess_subtitle
from core.tts_processor import generate_tts_from_reference


class GUIStreamlinePipeline(QObject, StreamlinePipeline):
    """GUI专用的流水线处理器，支持日志输出"""

    # 信号定义
    log_message = Signal(str)  # 日志消息

    def __init__(self, output_dir: Optional[str] = None):
        QObject.__init__(self)
        StreamlinePipeline.__init__(self, output_dir)

        # 设置日志处理器
        self.setup_logging()

    def setup_logging(self):
        """设置日志处理器，将日志发送到信号"""

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

        # 添加到当前logger和所有子模块的logger
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # 延迟设置流水线组件日志处理器（在初始化完成后）
        self._delayed_setup_logging(handler)
    
    def _delayed_setup_logging(self, handler):
        """延迟设置流水线组件的日志处理器"""
        # 检查task_scheduler是否已初始化
        if hasattr(self, 'task_scheduler') and self.task_scheduler:
            self.task_scheduler.logger.addHandler(handler)
            self.task_scheduler.logger.setLevel(logging.INFO)
            
            # 为所有处理器添加日志处理器
            for processor in self.task_scheduler.processors.values():
                processor.logger.addHandler(handler)
                processor.logger.setLevel(logging.INFO)
                
            # 为队列管理器添加日志处理器
            if hasattr(self.task_scheduler, 'queue_manager'):
                self.task_scheduler.queue_manager.logger.addHandler(handler)
                self.task_scheduler.queue_manager.logger.setLevel(logging.INFO)
                
            self.logger.info("流水线日志处理器设置完成")

    def process_batch_streamline(
        self, 
        video_subtitle_pairs: List[Tuple[str, Optional[str]]],
        resume_from_cache: bool = True
    ) -> Dict[str, Any]:
        """
        使用流水线模式批量处理视频（带GUI日志支持）
        """
        # 在处理开始前，确保日志处理器已设置
        if hasattr(self, 'task_scheduler') and self.task_scheduler:
            # 重新设置日志处理器以确保所有组件都有日志输出
            handler = None
            for h in self.logger.handlers:
                if hasattr(h, 'signal_emitter'):
                    handler = h
                    break
            
            if handler:
                self._delayed_setup_logging(handler)
        
        # 调用父类方法
        return super().process_batch_streamline(video_subtitle_pairs, resume_from_cache)


class GUIDubbingPipeline(QObject, DubbingPipeline):
    """GUI专用的配音处理流水线，支持日志输出"""

    # 信号定义
    log_message = Signal(str)  # 日志消息

    def __init__(self, output_dir: Optional[str] = None):
        QObject.__init__(self)
        DubbingPipeline.__init__(self, output_dir)

        # 设置日志处理器
        self.setup_logging()

    def setup_logging(self):
        """设置日志处理器，将日志发送到信号"""

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

        # 添加到当前logger
        self.logger.addHandler(handler)

    def process_video_with_progress(
        self,
        video_path: str,
        subtitle_path: Optional[str] = None,
        resume_from_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        处理视频配音的完整流程
        """
        try:
            self.logger.info(f"开始处理视频: {video_path}")

            # 获取文件路径
            paths = self._get_file_paths(video_path, subtitle_path)

            # 初始化缓存
            cache_data = {
                "video_path": video_path,
                "subtitle_path": str(paths.subtitle_path),
                "output_dir": str(paths.output_dir),
                "created_at": datetime.now().isoformat(),
                "completed_steps": {},
                "file_paths": {
                    "video_path": str(paths.video_path),
                    "subtitle_path": str(paths.subtitle_path),
                    "processed_subtitle": str(paths.processed_subtitle),
                    "vocal_audio": str(paths.vocal_audio),
                    "background_audio": str(paths.background_audio),
                    "silent_video": str(paths.silent_video),
                    "media_separation_dir": str(paths.media_separation_dir),
                    "reference_audio_dir": str(paths.reference_audio_dir),
                    "tts_output_dir": str(paths.tts_output_dir),
                    "aligned_audio_dir": str(paths.aligned_audio_dir),
                    "adjusted_video_dir": str(paths.adjusted_video_dir),
                    "reference_results": str(paths.reference_results),
                    "tts_results": str(paths.tts_results),
                    "aligned_results": str(paths.aligned_results),
                    "aligned_audio": str(paths.aligned_audio),
                    "aligned_srt": str(paths.aligned_srt),
                    "final_video": str(paths.final_video),
                    "speed_adjusted_video": str(paths.speed_adjusted_video),
                    "pipeline_cache": str(paths.pipeline_cache),
                },
            }

            # 检查和修复缓存
            existing_cache = None
            if resume_from_cache:
                repair_result = self.check_and_repair_cache(video_path)
                self.logger.info(f"缓存检查结果: {repair_result['message']}")

                if repair_result.get("cache_exists"):
                    existing_cache = self._load_pipeline_cache(paths.pipeline_cache)
                    if existing_cache:
                        cache_data = existing_cache
                        self.logger.info("使用缓存继续处理")
                    else:
                        self.logger.info("缓存加载失败，开始全新处理")
                else:
                    self.logger.info("未找到缓存，开始全新处理")
            else:
                self.logger.info("禁用缓存，开始全新处理")

            # 处理各个步骤
            steps_info = [
                ("preprocess_subtitle", self._process_preprocess_subtitle, paths),
                ("separate_media", self._process_separate_media, paths),
                (
                    "generate_reference_audio",
                    self._process_generate_reference_audio,
                    paths,
                ),
                ("generate_tts", self._process_generate_tts, paths),
                ("align_audio", self._process_align_audio, paths),
                (
                    "generate_aligned_srt",
                    self._process_generate_aligned_srt,
                    paths,
                    video_path,
                ),
                ("process_video_speed", self._process_video_speed, paths),
                ("merge_audio_video", self._process_merge_audio_video, paths),
            ]

            for step_id, process_func, *args in steps_info:
                if not self._check_step_completed(cache_data, step_id):
                    self.logger.info(f"开始处理步骤: {step_id}")
                    try:
                        if len(args) == 1:
                            result = process_func(args[0], cache_data)
                        else:
                            result = process_func(args[0], args[1], cache_data)

                        if not result.get("success", True):
                            return result
                    except Exception as e:
                        return {
                            "success": False,
                            "message": f"步骤 {step_id} 失败: {str(e)}",
                            "error": str(e),
                        }
                else:
                    self.logger.info(f"步骤 {step_id} 已完成，跳过")

            self.logger.info(f"处理完成！输出文件: {paths.final_video}")

            # 计算完成的步骤数
            completed_steps = sum(
                1
                for step in cache_data["completed_steps"].values()
                if step.get("completed", False)
            )

            return {
                "success": True,
                "message": "视频配音处理完成",
                "output_file": str(paths.final_video),
                "output_dir": str(paths.output_dir),
                "steps_completed": completed_steps,
                "cache_file": str(paths.pipeline_cache),
                "resumed_from_cache": resume_from_cache and existing_cache is not None,
            }

        except Exception as e:
            self.logger.error(f"处理过程中发生错误: {str(e)}")
            return {"success": False, "message": f"处理失败: {str(e)}", "error": str(e)}

    def _process_preprocess_subtitle(self, paths, cache_data):
        """处理字幕预处理步骤"""
        result = preprocess_subtitle(str(paths.subtitle_path), str(paths.output_dir))
        self._mark_step_completed(cache_data, "preprocess_subtitle", {"result": result})
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        return result

    def _process_separate_media(self, paths, cache_data):
        """处理媒体分离步骤"""
        self.logger.info("开始分离音视频...")
        result = separate_media(str(paths.video_path), str(paths.media_separation_dir))
        self._mark_step_completed(cache_data, "separate_media", {"result": result})
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        return result

    def _process_generate_reference_audio(self, paths, cache_data):
        """处理生成参考音频步骤"""
        self.logger.info("开始生成参考音频...")
        result = generate_reference_audio(
            str(paths.vocal_audio),
            str(paths.processed_subtitle),
            str(paths.reference_audio_dir),
        )
        self._mark_step_completed(
            cache_data, "generate_reference_audio", {"result": result}
        )
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        return result

    def _process_generate_tts(self, paths, cache_data):
        """处理TTS生成步骤"""
        if not paths.reference_results.exists():
            return {
                "success": False,
                "message": "参考结果文件不存在",
                "error": f"文件不存在: {paths.reference_results}",
            }

        self.logger.info("开始生成TTS音频...")
        result = generate_tts_from_reference(
            str(paths.reference_results), str(paths.tts_output_dir)
        )

        if not result.get("success", False):
            return {
                "success": False,
                "message": "TTS生成失败",
                "error": result.get("error", "未知错误"),
            }

        self._mark_step_completed(cache_data, "generate_tts", {"result": result})
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        return result

    def _process_align_audio(self, paths, cache_data):
        """处理音频对齐步骤"""
        self.logger.info("开始音频对齐...")
        result = align_audio_with_subtitles(
            tts_results_path=str(paths.tts_results),
            srt_path=str(paths.processed_subtitle),
            output_path=str(paths.aligned_audio),
        )

        # 保存对齐结果到JSON文件
        result_copy = result.copy()
        result_copy["saved_at"] = datetime.now().isoformat()
        with open(paths.aligned_results, "w", encoding="utf-8") as f:
            json.dump(result_copy, f, ensure_ascii=False, indent=2)

        self._mark_step_completed(cache_data, "align_audio", {"result": result})
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        return result

    def _process_generate_aligned_srt(self, paths, video_path, cache_data):
        """处理生成对齐字幕步骤"""
        # 生成对齐后的SRT字幕
        generate_aligned_srt(
            str(paths.aligned_results),
            str(paths.processed_subtitle),
            str(paths.aligned_srt),
        )

        # 检查原始字幕格式
        original_subtitle_path = str(paths.subtitle_path)
        original_subtitle_ext = Path(original_subtitle_path).suffix.lower()

        if original_subtitle_ext != ".srt":
            self.logger.info(
                f"检测到原始字幕格式为 {original_subtitle_ext}，正在转换..."
            )

            if original_subtitle_ext == ".ass":
                aligned_ass_path = (
                    paths.output_dir / f"{Path(video_path).stem}_aligned.ass"
                )
                sync_success = sync_srt_timestamps_to_ass(
                    original_subtitle_path,
                    str(paths.aligned_srt),
                    str(aligned_ass_path),
                )
                if sync_success:
                    self.logger.info(f"ASS字幕时间戳同步完成: {aligned_ass_path}")
                else:
                    self.logger.error("ASS字幕时间戳同步失败")
            else:
                aligned_subtitle_path = (
                    paths.output_dir
                    / f"{Path(video_path).stem}_aligned{original_subtitle_ext}"
                )
                convert_success = convert_subtitle(
                    str(paths.aligned_srt), str(aligned_subtitle_path)
                )
                if convert_success:
                    self.logger.info(f"字幕格式转换完成: {aligned_subtitle_path}")
                else:
                    self.logger.error("字幕格式转换失败")

        self._mark_step_completed(cache_data, "generate_aligned_srt")
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        return {"success": True}

    def _process_video_speed(self, paths, cache_data):
        """处理视频速度调整步骤"""
        self.logger.info("开始处理视频速度调整...")
        process_video_speed_adjustment(
            str(paths.silent_video),
            str(paths.processed_subtitle),
            str(paths.aligned_srt),
        )
        self._mark_step_completed(cache_data, "process_video_speed")
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        return {"success": True}

    def _process_merge_audio_video(self, paths, cache_data):
        """处理音视频合并步骤"""
        self.logger.info("开始合并音视频...")
        merge_audio_video(
            str(paths.speed_adjusted_video),
            str(paths.aligned_audio),
            str(paths.final_video),
        )
        self._mark_step_completed(cache_data, "merge_audio_video")
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        return {"success": True}


class VideoSubtitleMatcher:
    """视频字幕匹配器"""

    @staticmethod
    def find_video_subtitle_pairs(folder_path: str) -> List[Tuple[str, str]]:
        """
        在文件夹中查找视频和对应的字幕文件，只返回完全匹配的对

        Args:
            folder_path: 文件夹路径

        Returns:
            List of (video_path, subtitle_path) tuples，只返回有完全匹配字幕的视频
        """
        folder = Path(folder_path)
        if not folder.exists():
            return []

        # 支持的视频格式
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
        # 支持的字幕格式
        subtitle_extensions = {".srt", ".ass", ".ssa", ".sub", ".vtt"}

        # 递归查找所有视频文件，使用集合避免重复
        video_files_set = set()
        for ext in video_extensions:
            video_files_set.update(folder.rglob(f"*{ext}"))
            video_files_set.update(folder.rglob(f"*{ext.upper()}"))
        
        # 转换为列表并排序
        video_files = sorted(list(video_files_set))

        pairs = []
        # 用于跟踪已匹配的字幕文件，避免重复匹配
        matched_subtitles = set()

        for video_file in video_files:
            video_name = video_file.stem
            video_folder = video_file.parent

            # 只查找完全匹配的字幕文件
            subtitle_file = None
            for ext in subtitle_extensions:
                potential_subtitle = video_folder / f"{video_name}{ext}"
                if (
                    potential_subtitle.exists()
                    and str(potential_subtitle) not in matched_subtitles
                ):
                    subtitle_file = potential_subtitle
                    matched_subtitles.add(str(potential_subtitle))
                    break

                # 尝试大写扩展名
                potential_subtitle = video_folder / f"{video_name}{ext.upper()}"
                if (
                    potential_subtitle.exists()
                    and str(potential_subtitle) not in matched_subtitles
                ):
                    subtitle_file = potential_subtitle
                    matched_subtitles.add(str(potential_subtitle))
                    break

            # 只有找到完全匹配的字幕文件才添加到结果中
            if subtitle_file:
                pairs.append((str(video_file), str(subtitle_file)))

        return pairs


class LogHandler(logging.Handler):
    """自定义日志处理器，用于将日志输出到GUI"""

    def __init__(self, signal_emitter):
        super().__init__()
        self.signal_emitter = signal_emitter

    def emit(self, record):
        msg = self.format(record)
        self.signal_emitter.log_message.emit(msg)


class DubbingWorkerThread(QThread):
    """配音处理工作线程"""

    # 信号定义
    processing_finished = Signal(bool, str, dict)  # 是否成功, 消息, 结果详情

    def __init__(
        self,
        video_path: str,
        subtitle_path: Optional[str] = None,
        resume_from_cache: bool = True,
    ):
        super().__init__()
        self.video_path = video_path
        self.subtitle_path = subtitle_path
        self.resume_from_cache = resume_from_cache
        self.is_cancelled = False

        # 创建GUI专用管道
        self.pipeline = GUIDubbingPipeline()

    def cancel(self):
        """取消处理"""
        self.is_cancelled = True

    def run(self):
        """执行配音处理"""
        try:
            if self.is_cancelled:
                return

            # 执行处理
            result = self.pipeline.process_video_with_progress(
                self.video_path, self.subtitle_path, self.resume_from_cache
            )

            if self.is_cancelled:
                return

            # 发送完成信号
            self.processing_finished.emit(result["success"], result["message"], result)

        except Exception as e:
            self.processing_finished.emit(False, f"处理失败: {str(e)}", {})


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
        self.gui_pipeline = GUIDubbingPipeline()
        self.pipeline = DubbingPipeline()

        # 连接日志信号
        self.log_message.connect(self.append_log_message)

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

        # 状态变量
        self.current_mode = "single"  # "single" 或 "batch"
        self.current_video_path = ""
        self.current_subtitle_path = ""
        self.current_folder_path = ""
        self.video_subtitle_pairs = []

    def closeEvent(self, event):
        """处理窗口关闭事件，确保所有资源被正确清理"""
        try:
            print("开始关闭应用程序...")
            
            # 停止并清理所有工作线程
            self._cleanup_worker_threads()
            
            # 清理流水线资源
            self._cleanup_pipelines()
            
            # 清理日志处理器
            self._cleanup_log_handlers()
            
            print("应用程序资源清理完成")
            
        except Exception as e:
            print(f"关闭时清理资源出错: {e}")
        
        finally:
            # 接受关闭事件
            event.accept()
            
            # 立即强制退出
            self._force_exit()

    def _cleanup_worker_threads(self):
        """清理工作线程"""
        # 停止并清理工作线程
        if hasattr(self, 'worker_thread') and self.worker_thread is not None:
            print("停止单文件处理线程...")
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()
                self.worker_thread.wait(1000)  # 减少等待时间到1秒
                if self.worker_thread.isRunning():
                    self.worker_thread.kill()
            self.worker_thread = None

        # 停止并清理批量处理线程
        if hasattr(self, 'parallel_batch_worker_thread') and self.parallel_batch_worker_thread is not None:
            print("停止批量处理线程...")
            if self.parallel_batch_worker_thread.isRunning():
                # 先尝试取消任务
                if hasattr(self.parallel_batch_worker_thread, 'cancel'):
                    self.parallel_batch_worker_thread.cancel()
                
                # 终止线程
                self.parallel_batch_worker_thread.terminate()
                self.parallel_batch_worker_thread.wait(1000)  # 减少等待时间到1秒
                if self.parallel_batch_worker_thread.isRunning():
                    self.parallel_batch_worker_thread.kill()
            self.parallel_batch_worker_thread = None

    def _cleanup_pipelines(self):
        """清理流水线资源"""
        # 清理GUI流水线
        if hasattr(self, 'gui_pipeline') and self.gui_pipeline is not None:
            print("停止GUI流水线...")
            # 停止TaskScheduler
            if hasattr(self.gui_pipeline, 'task_scheduler') and self.gui_pipeline.task_scheduler is not None:
                try:
                    # 设置更短的超时时间
                    self.gui_pipeline.task_scheduler.stop(timeout=2.0)
                    
                    # 强制关闭所有线程池
                    if hasattr(self.gui_pipeline.task_scheduler, 'worker_pools'):
                        for step_id, executor in self.gui_pipeline.task_scheduler.worker_pools.items():
                            print(f"强制关闭步骤 {step_id} 的线程池...")
                            if hasattr(executor, '_threads'):
                                # 强制终止所有线程
                                for thread in list(executor._threads):
                                    if thread.is_alive():
                                        # 使用私有方法强制终止线程
                                        try:
                                            import ctypes
                                            ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                                ctypes.c_long(thread.ident), 
                                                ctypes.py_object(SystemExit)
                                            )
                                            if ret == 0:
                                                print(f"线程 {thread.ident} 终止失败")
                                            else:
                                                print(f"线程 {thread.ident} 已强制终止")
                                        except Exception as e:
                                            print(f"强制终止线程失败: {e}")
                            
                            # 立即关闭线程池，不等待
                            executor.shutdown(wait=False)
                    
                except Exception as e:
                    print(f"停止TaskScheduler时出错: {e}")
            
            # 清理其他流水线资源
            if hasattr(self.gui_pipeline, 'cleanup'):
                try:
                    self.gui_pipeline.cleanup()
                except Exception as e:
                    print(f"清理GUI流水线时出错: {e}")
            self.gui_pipeline = None

        # 清理普通流水线
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            print("清理普通流水线...")
            if hasattr(self.pipeline, 'cleanup'):
                try:
                    self.pipeline.cleanup()
                except Exception as e:
                    print(f"清理普通流水线时出错: {e}")
            self.pipeline = None

    def _cleanup_log_handlers(self):
        """清理日志处理器"""
        try:
            root_logger = logging.getLogger()
            handlers_to_remove = []
            for handler in root_logger.handlers[:]:
                if hasattr(handler, 'signal_emitter'):
                    handlers_to_remove.append(handler)
            
            for handler in handlers_to_remove:
                root_logger.removeHandler(handler)
                try:
                    handler.close()
                except:
                    pass
        except Exception as e:
            print(f"清理日志处理器时出错: {e}")

    def _force_exit(self):
        """强制退出应用程序"""
        import threading
        import os
        import sys
        
        # 打印活跃线程信息用于调试
        active_threads = threading.active_count()
        print(f"当前活跃线程数: {active_threads}")
        
        # 如果有多余的线程，列出它们
        if active_threads > 1:
            print("活跃线程列表:")
            for thread in threading.enumerate():
                print(f"  - {thread.name} (daemon: {thread.daemon})")
        
        # 尝试正常退出QApplication
        app = QApplication.instance()
        if app:
            app.quit()
        
        # 立即强制退出进程，不再等待
        print("强制退出进程...")
        try:
            # 使用最激进的退出方式
            os._exit(0)  # 立即终止进程，不执行清理
        except SystemExit:
            # 如果os._exit失败，尝试其他方式
            import signal
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

        # 结果标签页
        result_tab = self.create_result_tab()
        tab_widget.addTab(result_tab, "处理结果")

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
        self.status_table.setHorizontalHeaderLabels([
            "视频文件", 
            "步骤1\n字幕预处理", "步骤2\n媒体分离", "步骤3\n参考音频", "步骤4\nTTS生成",
            "步骤5\n音频对齐", "步骤6\n对齐字幕", "步骤7\n视频调速", "步骤8\n合并输出",
            "整体状态", "进度"
        ])

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
        
        header.setSectionResizeMode(9, QHeaderView.ResizeToContents)   # 整体状态
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

    def create_result_tab(self) -> QWidget:
        """创建结果标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)

        # 批量处理进度条
        self.batch_progress_group = QGroupBox("批量处理进度")
        batch_progress_layout = QVBoxLayout(self.batch_progress_group)
        batch_progress_layout.setContentsMargins(15, 12, 15, 12)

        # 进度条
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                text-align: center;
                background-color: #f8f9fa;
                font-size: 11px;
                color: #495057;
                font-weight: bold;
                height: 25px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d6efd, stop:1 #6610f2);
                border-radius: 3px;
            }
        """
        )
        batch_progress_layout.addWidget(self.batch_progress_bar)

        # 当前处理文件标签
        self.current_file_label = QLabel("当前处理文件: 无")
        self.current_file_label.setStyleSheet("font-size: 11px; color: #6c757d;")
        batch_progress_layout.addWidget(self.current_file_label)

        # 默认隐藏进度条组
        self.batch_progress_group.hide()

        layout.addWidget(self.batch_progress_group)

        # 结果信息
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("处理完成后将在此显示结果信息...")
        self.result_text.setStyleSheet(
            """
            QTextEdit {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 20px;
                font-size: 12px;
                line-height: 1.6;
                color: #333;
            }
        """
        )
        layout.addWidget(self.result_text)

        return tab

    def append_log_message(self, message: str):
        """追加日志消息到文本框并解析任务状态"""
        self.log_text.append(message)
        
        # 解析日志消息来更新任务状态
        self.parse_log_for_task_status(message)
    
    def parse_log_for_task_status(self, message: str):
        """解析日志消息以更新任务状态"""
        try:
            # 1. 解析工作线程开始处理任务的日志
            # 格式: "工作线程 Step-X-stepname-Y 开始处理任务 streamline_task_XXX_taskname"
            if "工作线程" in message and "开始处理任务" in message and "streamline_task_" in message:
                import re
                step_match = re.search(r'Step-(\d+)-', message)
                task_match = re.search(r'streamline_task_\d+_(.*?)(?:\s|$)', message)
                
                if step_match and task_match:
                    step_id = int(step_match.group(1))
                    task_name = task_match.group(1).strip()
                    self.update_task_step_status(task_name, step_id, "processing")
                    return
            
            # 2. 解析处理器开始处理任务的日志（基类统一格式）
            # 格式: "开始处理任务 streamline_task_XXX_taskname - 步骤: stepname"
            elif "开始处理任务" in message and "streamline_task_" in message and "步骤:" in message:
                import re
                task_match = re.search(r'开始处理任务 streamline_task_\d+_(.*?) - 步骤: (\w+)', message)
                
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
                        "merge_audio_video": 7
                    }
                    
                    step_id = step_name_to_id.get(step_name)
                    if step_id is not None:
                        self.update_task_step_status(task_name, step_id, "processing")
                    return
            
            # 3. 解析各步骤的自定义开始日志（处理器内部日志）
            # 这些日志可能不包含完整的task_id，需要推断当前正在处理的任务
            elif "开始" in message:
                import re
                step_keywords = {
                    "开始预处理字幕": 0,
                    "开始媒体分离": 1, 
                    "开始生成参考音频": 2,
                    "开始TTS生成": 3,
                    "开始音频对齐": 4,
                    "开始生成对齐字幕": 5,
                    "开始视频速度调整": 6,
                    "开始音视频合并": 7
                }
                
                # 检查是否匹配任何步骤关键词
                for keyword, step_id in step_keywords.items():
                    if keyword in message:
                        # 如果消息包含streamline_task_，提取任务名
                        if "streamline_task_" in message:
                            task_match = re.search(r'streamline_task_\d+_(.*?)(?:\s|:|$)', message)
                            if task_match:
                                task_name = task_match.group(1).strip()
                                self.update_task_step_status(task_name, step_id, "processing")
                        else:
                            # 如果没有task_id，更新所有可能正在此步骤的任务
                            self.update_current_step_tasks_status(step_id, "processing")
                        return
                
                # 4. 通用的开始处理检测（包含streamline_task_的其他开始日志）
                if "streamline_task_" in message:
                    task_match = re.search(r'streamline_task_\d+_(.*?)(?:\s|:|$)', message)
                    if task_match:
                        task_name = task_match.group(1).strip()
                        
                        # 根据消息内容推断步骤
                        if any(word in message for word in ["TTS生成", "generate_tts"]):
                            self.update_task_step_status(task_name, 3, "processing")
                        elif any(word in message for word in ["参考音频", "generate_reference_audio"]):
                            self.update_task_step_status(task_name, 2, "processing")
                        elif any(word in message for word in ["媒体分离", "separate_media"]):
                            self.update_task_step_status(task_name, 1, "processing")
                        elif any(word in message for word in ["字幕预处理", "preprocess_subtitle"]):
                            self.update_task_step_status(task_name, 0, "processing")
                        elif any(word in message for word in ["音频对齐", "align_audio"]):
                            self.update_task_step_status(task_name, 4, "processing")
                        elif any(word in message for word in ["对齐字幕", "generate_aligned_srt"]):
                            self.update_task_step_status(task_name, 5, "processing")
                        elif any(word in message for word in ["视频调速", "视频速度", "process_video_speed"]):
                            self.update_task_step_status(task_name, 6, "processing")
                        elif any(word in message for word in ["合并", "merge_audio_video"]):
                            self.update_task_step_status(task_name, 7, "processing")
                    return
            
            # 5. 简化的开始处理检测 - 检测TTS相关日志
            elif ("开始TTS生成" in message or "gradio_api" in message.lower()) and hasattr(self, 'status_table'):
                # 没有具体任务名的情况下，尝试更新所有正在第3步的任务
                self.update_processing_tasks_status()
                return
            
            # 6. 解析任务完成的日志
            # 格式: "任务 streamline_task_XXX_taskname 步骤 stepname 处理成功"
            elif "处理成功" in message and "streamline_task_" in message:
                import re
                task_match = re.search(r'任务 streamline_task_\d+_(.*?) 步骤 (\w+) 处理成功', message)
                
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
                        "merge_audio_video": 7
                    }
                    
                    step_id = step_name_to_id.get(step_name)
                    if step_id is not None:
                        self.update_task_step_status(task_name, step_id, "completed")
                return
            
            # 7. 解析步骤完成的日志（处理器内部完成日志）
            elif any(keyword in message for keyword in [
                "字幕预处理完成", "媒体分离完成", "参考音频生成完成", "TTS生成完成",
                "音频对齐完成", "对齐字幕生成完成", "视频速度调整完成", "音视频合并完成"
            ]):
                # 这些完成日志通常紧跟在处理日志之后，可以用来确认完成状态
                # 但由于没有task_id，我们依赖前面的基类完成日志来更新状态
                return
                
            # 8. 解析任务失败的日志
            # 格式: "任务 streamline_task_XXX_taskname 在队列 step_X_stepname 处理失败"
            elif "处理失败" in message and "streamline_task_" in message:
                import re
                task_match = re.search(r'任务 streamline_task_\d+_(.*?) 在队列 step_(\d+)_', message)
                
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
            if not hasattr(self, 'status_table'):
                return
                
            # 查找所有可能正在处理此步骤的任务
            for row in range(self.status_table.rowCount()):
                # 检查前一个步骤是否已完成，当前步骤是否未开始
                if step_id > 0:
                    prev_step_item = self.status_table.item(row, step_id)  # 前一步的列索引
                    curr_step_item = self.status_table.item(row, step_id + 1)  # 当前步的列索引
                    
                    # 如果前一步已完成，当前步未开始，则更新当前步为处理中
                    if (prev_step_item and prev_step_item.text() == "✅" and
                        curr_step_item and curr_step_item.text() == "⏸️"):
                        
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
            if not hasattr(self, 'status_table'):
                return
                
            # 查找已完成前3个步骤但第4步还未开始的任务，将其第4步设为处理中
            for row in range(self.status_table.rowCount()):
                step2_item = self.status_table.item(row, 3)  # 步骤2 (列索引3)
                step3_item = self.status_table.item(row, 4)  # 步骤3 (列索引4)
                
                # 如果步骤2已完成，步骤3还未开始，则设置步骤3为处理中
                if (step2_item and step2_item.text() == "✅" and 
                    step3_item and step3_item.text() == "⏸️"):
                    
                    video_item = self.status_table.item(row, 0)
                    if video_item:
                        video_name = Path(video_item.text()).stem
                        self.update_task_step_status(video_name, 3, "processing")
                        
        except Exception as e:
            pass
    
    def test_status_update(self):
        """测试状态更新功能"""
        # 手动测试状态更新
        if hasattr(self, 'status_table') and self.status_table.rowCount() > 0:
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
            import traceback

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
            self.file_table.setItem(i, 0, video_item)

            # 字幕文件名（现在一定有匹配的字幕）
            subtitle_name = Path(subtitle_path).name
            subtitle_item = QTableWidgetItem(subtitle_name)
            subtitle_item.setToolTip(subtitle_path)
            self.file_table.setItem(i, 1, subtitle_item)

            # 状态（现在一定是就绪状态）
            status_item = QTableWidgetItem("就绪")
            status_item.setForeground(QColor("#198754"))  # 绿色
            self.file_table.setItem(i, 2, status_item)

            # 选择框
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # 现在所有项都有字幕，默认全选
            self.file_table.setCellWidget(i, 3, checkbox)

    def update_task_step_status(self, task_name: str, step_id: int, status: str, message: str = ""):
        """
        更新特定任务的步骤状态
        
        Args:
            task_name: 任务名称（通常是视频文件名，不含扩展名）
            step_id: 步骤ID (0-7)
            status: 状态 ("processing", "completed", "failed")
            message: 状态消息
        """
        try:
            # 查找任务在状态表格中的行索引
            task_row = -1
            for i in range(self.status_table.rowCount()):
                video_item = self.status_table.item(i, 0)
                if video_item:
                    video_name = Path(video_item.text()).stem  # 去掉扩展名
                    if video_name in task_name or task_name in video_name:
                        task_row = i
                        break
            
            if task_row == -1:
                self.logger.warning(f"未找到任务 {task_name} 在状态表格中的对应行")
                return
            
            # 更新步骤状态（列索引 1-8）
            step_col = 1 + step_id
            step_item = self.status_table.item(task_row, step_col)
            
            if not step_item:
                step_item = QTableWidgetItem()
                self.status_table.setItem(task_row, step_col, step_item)
            
            # 设置状态图标和颜色
            if status == "processing":
                step_item.setText("🔄")  # 处理中
                step_item.setForeground(QColor("#fd7e14"))  # 橙色
                tooltip = f"步骤{step_id + 1}: 处理中"
            elif status == "completed":
                step_item.setText("✅")  # 完成
                step_item.setForeground(QColor("#198754"))  # 绿色
                tooltip = f"步骤{step_id + 1}: 已完成"
            elif status == "failed":
                step_item.setText("❌")  # 失败
                step_item.setForeground(QColor("#dc3545"))  # 红色
                tooltip = f"步骤{step_id + 1}: 失败"
            else:
                step_item.setText("⏸️")  # 未开始
                step_item.setForeground(QColor("#6c757d"))  # 灰色
                tooltip = f"步骤{step_id + 1}: 未开始"
            
            if message:
                tooltip += f" - {message}"
            
            step_item.setToolTip(tooltip)
            step_item.setTextAlignment(Qt.AlignCenter)
            
            # 更新整体状态
            self.update_overall_task_status(task_row)
            
        except Exception as e:
            self.logger.error(f"更新任务步骤状态失败: {e}")
    
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
                self.status_table.setItem(row, 9, status_item)
            
            # 更新进度列（列索引 10）
            progress_item = self.status_table.item(row, 10)
            if not progress_item:
                progress_item = QTableWidgetItem()
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
        """初始化处理状态表格"""
        try:
            self.status_table.setRowCount(len(video_subtitle_pairs))

            for i, (video_path, subtitle_path) in enumerate(video_subtitle_pairs):
                # 视频文件名
                video_name = Path(video_path).name
                video_item = QTableWidgetItem(video_name)
                video_item.setToolTip(video_path)
                self.status_table.setItem(i, 0, video_item)

                # 初始化8个步骤状态列（列索引 1-8）
                step_names = [
                    "字幕预处理", "媒体分离", "参考音频", "TTS生成",
                    "音频对齐", "对齐字幕", "视频调速", "合并输出"
                ]
                
                for step_idx in range(8):
                    step_item = QTableWidgetItem("⏸️")  # 暂停符号表示未开始
                    step_item.setToolTip(f"步骤{step_idx + 1}: {step_names[step_idx]} - 未开始")
                    step_item.setTextAlignment(Qt.AlignCenter)
                    step_item.setForeground(QColor("#6c757d"))  # 灰色
                    self.status_table.setItem(i, 1 + step_idx, step_item)

                # 整体状态（列索引 9）
                status_item = QTableWidgetItem("就绪")
                status_item.setForeground(QColor("#198754"))  # 绿色
                self.status_table.setItem(i, 9, status_item)

                # 进度（列索引 10）
                progress_item = QTableWidgetItem("0/8")
                progress_item.setTextAlignment(Qt.AlignCenter)
                progress_item.setForeground(QColor("#6c757d"))  # 灰色
                self.status_table.setItem(i, 10, progress_item)
                
        except Exception as e:
            self.logger.error(f"初始化状态表格失败: {e}")

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
        """开始处理"""
        if self.current_mode == "single":
            self.start_single_processing()
        else:
            self.start_batch_processing()

    def start_single_processing(self):
        """开始单文件处理"""
        # 验证输入
        if not self.current_video_path:
            QMessageBox.warning(self, "警告", "请选择视频文件！")
            return

        if not Path(self.current_video_path).exists():
            QMessageBox.warning(self, "警告", "视频文件不存在！")
            return

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        try:
            # 创建工作线程
            self.worker_thread = DubbingWorkerThread(
                self.current_video_path,
                self.current_subtitle_path if self.current_subtitle_path else None,
                True,  # 默认从缓存恢复
            )

            # 连接信号
            self.worker_thread.processing_finished.connect(self.processing_finished)

            # 连接GUI管道的信号
            self.gui_pipeline = self.worker_thread.pipeline
            self.gui_pipeline.log_message.connect(self.append_log_message)

            # 启动线程
            self.worker_thread.start()

        except Exception as e:
            import traceback

            error_msg = f"启动处理失败: {str(e)}\n详细错误:\n{traceback.format_exc()}"
            self.log_text.append(error_msg)
            QMessageBox.critical(self, "错误", error_msg)

            # 恢复UI状态
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

    def start_batch_processing(self):
        """开始批量处理"""
        selected_pairs = self.get_selected_pairs()

        if not selected_pairs:
            QMessageBox.warning(self, "警告", "请至少选择一个要处理的视频！")
            return

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        try:
            # 并行批量处理（默认）
            self.batch_progress_group.show()
            self.batch_progress_bar.setMaximum(len(selected_pairs))
            self.batch_progress_bar.setValue(0)
            self.current_file_label.setText("当前处理文件: 准备中...")

            # 获取最大工作线程数
            max_workers = None
            if self.max_workers_spinbox.text().strip():
                try:
                    max_workers = int(self.max_workers_spinbox.text().strip())
                    if max_workers < 1:
                        max_workers = None
                except ValueError:
                    max_workers = None

            # 创建流水线批量工作线程
            self.parallel_batch_worker_thread = StreamlineBatchDubbingWorkerThread(
                selected_pairs, True  # 默认从缓存恢复
            )

            # 连接信号
            self.parallel_batch_worker_thread.progress_update.connect(
                self.parallel_batch_progress_update
            )
            self.parallel_batch_worker_thread.batch_finished.connect(
                self.parallel_batch_finished
            )
            self.parallel_batch_worker_thread.log_message.connect(
                self.append_log_message
            )

            # 初始化状态表格
            self.initialize_status_table(selected_pairs)

            # 启动线程
            self.parallel_batch_worker_thread.start()

        except Exception as e:
            import traceback

            error_msg = (
                f"启动批量处理失败: {str(e)}\n详细错误:\n{traceback.format_exc()}"
            )
            self.log_text.append(error_msg)
            QMessageBox.critical(self, "错误", error_msg)

            # 恢复UI状态
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

    def cancel_processing(self):
        """取消处理"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
            self.worker_thread.wait()

        if (
            self.parallel_batch_worker_thread
            and self.parallel_batch_worker_thread.isRunning()
        ):
            self.parallel_batch_worker_thread.cancel()
            self.parallel_batch_worker_thread.wait()

        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.batch_progress_group.hide()

    def parallel_batch_progress_update(self, current: int, total: int, filename: str):
        """并行批量处理进度更新"""
        self.batch_progress_bar.setValue(current)
        self.current_file_label.setText(f"当前处理文件: {Path(filename).name}")

    def parallel_batch_finished(
        self, success: bool, message: str, result: Dict[str, Any]
    ):
        """并行批量处理完成"""
        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.batch_progress_group.hide()

        # 显示结果信息
        result_info = f"""并行批量处理完成！

处理统计:
• 总文件数: {result.get('total_count', 0)}
• 成功处理: {result.get('success_count', 0)}
• 处理失败: {result.get('failed_count', 0)}
• 总耗时: {result.get('total_time', 0):.2f} 秒
• 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

提示: 并行处理已完成，请查看各文件的输出目录
"""
        self.result_text.setText(result_info)

        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.warning(self, "完成", message)

    def processing_finished(self, success: bool, message: str, result: Dict[str, Any]):
        """单文件处理完成"""
        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if success:
            # 显示结果信息
            result_info = f"""处理完成！

输出信息:
• 输出文件: {result.get('output_file', '未知')}
• 输出目录: {result.get('output_dir', '未知')}
• 完成步骤: {result.get('steps_completed', 0)}/8
• 缓存恢复: {'是' if result.get('resumed_from_cache', False) else '否'}
• 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

您的配音视频已准备就绪！
"""
            self.result_text.setText(result_info)

            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.critical(self, "失败", message)

    def show_cache_info(self):
        """显示缓存信息"""
        if self.current_mode == "single":
            if not self.current_video_path:
                QMessageBox.warning(self, "警告", "请先选择视频文件！")
                return
            video_path = self.current_video_path
        else:
            QMessageBox.information(
                self, "信息", "批量模式下请单独查看各文件的缓存信息"
            )
            return

        try:
            cache_info = self.pipeline.get_pipeline_cache_info(video_path)

            if cache_info["success"]:
                if cache_info["cache_exists"]:
                    info_text = f"""缓存文件信息:

文件路径: {cache_info['cache_file']}
创建时间: {cache_info.get('created_at', '未知')}
更新时间: {cache_info.get('updated_at', '未知')}
总步骤数: {cache_info.get('total_steps', 0)}
已完成步骤: {cache_info.get('completed_steps', 0)}
剩余步骤: {cache_info.get('remaining_steps', 0)}

已完成的步骤:
{chr(10).join('• ' + step for step in cache_info.get('completed_step_names', []))}

剩余的步骤:
{chr(10).join('• ' + step for step in cache_info.get('remaining_step_names', []))}
"""
                else:
                    info_text = "缓存文件不存在"

                QMessageBox.information(self, "缓存信息", info_text)
            else:
                QMessageBox.warning(self, "错误", cache_info["message"])

        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取缓存信息失败: {str(e)}")

    def clear_cache(self):
        """清理缓存"""
        if self.current_mode == "single":
            if not self.current_video_path:
                QMessageBox.warning(self, "警告", "请先选择视频文件！")
                return
            video_path = self.current_video_path
        else:
            QMessageBox.information(self, "信息", "批量模式下请单独清理各文件的缓存")
            return

        reply = QMessageBox.question(
            self,
            "确认清理",
            "确定要清理缓存吗？这将删除所有已保存的处理进度。",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            try:
                result = self.pipeline.clear_pipeline_cache(video_path)

                if result["success"]:
                    QMessageBox.information(self, "成功", result["message"])
                else:
                    QMessageBox.warning(self, "错误", result["message"])

            except Exception as e:
                QMessageBox.critical(self, "错误", f"清理缓存失败: {str(e)}")

    def repair_cache(self):
        """修复缓存"""
        if self.current_mode == "single":
            if not self.current_video_path:
                QMessageBox.warning(self, "警告", "请先选择视频文件！")
                return
            video_path = self.current_video_path
        else:
            QMessageBox.information(self, "信息", "批量模式下请单独修复各文件的缓存")
            return

        try:
            result = self.pipeline.check_and_repair_cache(video_path)

            if result["success"]:
                QMessageBox.information(self, "成功", result["message"])
            else:
                QMessageBox.warning(self, "错误", result["message"])

        except Exception as e:
            QMessageBox.critical(self, "错误", f"修复缓存失败: {str(e)}")

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
        import signal
        def signal_handler(signum, frame):
            print(f"\n收到系统信号 {signum}，立即退出...")
            import os
            os._exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        
        # 运行应用程序
        app.exec()
        
    except Exception as e:
        print(f"应用程序启动失败: {e}")
    
    # 最终确保退出
    import os
    print("主函数结束，强制退出进程")
    os._exit(0)


if __name__ == "__main__":
    main()
