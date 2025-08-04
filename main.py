"""
DubbingX GUI 最终版 - 智能视频配音系统图形界面
简洁清晰的界面设计，优化可读性
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QLineEdit, QPushButton, QTextEdit, 
    QProgressBar, QFileDialog, QMessageBox, QGroupBox, QSplitter,
    QFrame, QTabWidget, QCheckBox, QRadioButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QButtonGroup
)
from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtGui import QFont, QColor

from core.dubbing_pipeline import DubbingPipeline
from core.media_processor import merge_audio_video, generate_reference_audio, separate_media
from core.subtitle_preprocessor import preprocess_subtitle
from core.tts_processor import generate_tts_from_reference
from core.audio_align_processor import (
    align_audio_with_subtitles,
    generate_aligned_srt,
    process_video_speed_adjustment
)
from core.subtitle.subtitle_processor import convert_subtitle, sync_srt_timestamps_to_ass


class GUIDubbingPipeline(QObject, DubbingPipeline):
    """GUI专用的配音处理流水线，支持实时进度更新"""
    
    # 信号定义
    step_started = Signal(str, str)  # 步骤ID, 步骤名称
    step_progress = Signal(str, int)  # 步骤ID, 进度百分比
    step_completed = Signal(str, bool, str)  # 步骤ID, 是否成功, 消息
    log_message = Signal(str)  # 日志消息
    
    def __init__(self, output_dir: Optional[str] = None):
        QObject.__init__(self)
        DubbingPipeline.__init__(self, output_dir)
        
        # 设置日志处理器
        self.setup_logging()
        
        # 步骤映射
        self.step_names = {
            'preprocess_subtitle': '1. 预处理字幕',
            'separate_media': '2. 分离音视频',
            'generate_reference_audio': '3. 生成参考音频',
            'generate_tts': '4. 生成TTS音频',
            'align_audio': '5. 音频对齐',
            'generate_aligned_srt': '6. 生成对齐字幕',
            'process_video_speed': '7. 处理视频速度',
            'merge_audio_video': '8. 合并音视频'
        }
        
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
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # 添加到当前logger
        self.logger.addHandler(handler)
        
    def emit_step_started(self, step_id: str):
        """发送步骤开始信号"""
        step_name = self.step_names.get(step_id, step_id)
        self.step_started.emit(step_id, step_name)
        self.step_progress.emit(step_id, 0)
        
    def emit_step_progress(self, step_id: str, progress: int):
        """发送步骤进度信号"""
        self.step_progress.emit(step_id, progress)
        
    def emit_step_completed(self, step_id: str, success: bool, message: str = ""):
        """发送步骤完成信号"""
        if success:
            self.step_progress.emit(step_id, 100)
        self.step_completed.emit(step_id, success, message)
        
    def process_video_with_progress(self, video_path: str, subtitle_path: Optional[str] = None, 
                                   resume_from_cache: bool = True) -> Dict[str, Any]:
        """
        处理视频配音的完整流程，带实时进度更新
        """
        try:
            self.logger.info(f"开始处理视频: {video_path}")
            
            # 获取文件路径
            paths = self._get_file_paths(video_path, subtitle_path)
            
            # 初始化缓存
            cache_data = {
                'video_path': video_path,
                'subtitle_path': str(paths.subtitle_path),
                'output_dir': str(paths.output_dir),
                'created_at': datetime.now().isoformat(),
                'completed_steps': {},
                'file_paths': {
                    'video_path': str(paths.video_path),
                    'subtitle_path': str(paths.subtitle_path),
                    'processed_subtitle': str(paths.processed_subtitle),
                    'vocal_audio': str(paths.vocal_audio),
                    'background_audio': str(paths.background_audio),
                    'silent_video': str(paths.silent_video),
                    'media_separation_dir': str(paths.media_separation_dir),
                    'reference_audio_dir': str(paths.reference_audio_dir),
                    'tts_output_dir': str(paths.tts_output_dir),
                    'aligned_audio_dir': str(paths.aligned_audio_dir),
                    'adjusted_video_dir': str(paths.adjusted_video_dir),
                    'reference_results': str(paths.reference_results),
                    'tts_results': str(paths.tts_results),
                    'aligned_results': str(paths.aligned_results),
                    'aligned_audio': str(paths.aligned_audio),
                    'aligned_srt': str(paths.aligned_srt),
                    'final_video': str(paths.final_video),
                    'speed_adjusted_video': str(paths.speed_adjusted_video),
                    'pipeline_cache': str(paths.pipeline_cache)
                }
            }
            
            # 检查和修复缓存
            existing_cache = None
            if resume_from_cache:
                repair_result = self.check_and_repair_cache(video_path)
                self.logger.info(f"缓存检查结果: {repair_result['message']}")
                
                if repair_result.get('cache_exists'):
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
                ('preprocess_subtitle', self._process_preprocess_subtitle, paths),
                ('separate_media', self._process_separate_media, paths),
                ('generate_reference_audio', self._process_generate_reference_audio, paths),
                ('generate_tts', self._process_generate_tts, paths),
                ('align_audio', self._process_align_audio, paths),
                ('generate_aligned_srt', self._process_generate_aligned_srt, paths, video_path),
                ('process_video_speed', self._process_video_speed, paths),
                ('merge_audio_video', self._process_merge_audio_video, paths)
            ]
            
            for step_id, process_func, *args in steps_info:
                if not self._check_step_completed(cache_data, step_id):
                    self.emit_step_started(step_id)
                    try:
                        if len(args) == 1:
                            result = process_func(args[0], cache_data)
                        else:
                            result = process_func(args[0], args[1], cache_data)
                        
                        if result.get('success', True):
                            self.emit_step_completed(step_id, True, "完成")
                        else:
                            self.emit_step_completed(step_id, False, result.get('message', '失败'))
                            return result
                    except Exception as e:
                        self.emit_step_completed(step_id, False, str(e))
                        return {
                            'success': False,
                            'message': f'步骤 {step_id} 失败: {str(e)}',
                            'error': str(e)
                        }
                else:
                    self.emit_step_completed(step_id, True, "从缓存恢复")
            
            self.logger.info(f"处理完成！输出文件: {paths.final_video}")
            
            # 计算完成的步骤数
            completed_steps = sum(1 for step in cache_data['completed_steps'].values() if step.get('completed', False))
            
            return {
                'success': True,
                'message': '视频配音处理完成',
                'output_file': str(paths.final_video),
                'output_dir': str(paths.output_dir),
                'steps_completed': completed_steps,
                'cache_file': str(paths.pipeline_cache),
                'resumed_from_cache': resume_from_cache and existing_cache is not None
            }
            
        except Exception as e:
            self.logger.error(f"处理过程中发生错误: {str(e)}")
            return {
                'success': False,
                'message': f'处理失败: {str(e)}',
                'error': str(e)
            }
    
    def _process_preprocess_subtitle(self, paths, cache_data):
        """处理字幕预处理步骤"""
        self.emit_step_progress('preprocess_subtitle', 10)
        result = preprocess_subtitle(str(paths.subtitle_path), str(paths.output_dir))
        self.emit_step_progress('preprocess_subtitle', 90)
        self._mark_step_completed(cache_data, 'preprocess_subtitle', {'result': result})
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        self.emit_step_progress('preprocess_subtitle', 100)
        return result
        
    def _process_separate_media(self, paths, cache_data):
        """处理媒体分离步骤"""
        self.emit_step_progress('separate_media', 5)
        self.logger.info("开始分离音视频...")
        result = separate_media(str(paths.video_path), str(paths.media_separation_dir))
        self.emit_step_progress('separate_media', 95)
        self._mark_step_completed(cache_data, 'separate_media', {'result': result})
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        self.emit_step_progress('separate_media', 100)
        return result
        
    def _process_generate_reference_audio(self, paths, cache_data):
        """处理生成参考音频步骤"""
        self.emit_step_progress('generate_reference_audio', 10)
        self.logger.info("开始生成参考音频...")
        result = generate_reference_audio(
            str(paths.vocal_audio),
            str(paths.processed_subtitle),
            str(paths.reference_audio_dir)
        )
        self.emit_step_progress('generate_reference_audio', 90)
        self._mark_step_completed(cache_data, 'generate_reference_audio', {'result': result})
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        self.emit_step_progress('generate_reference_audio', 100)
        return result
        
    def _process_generate_tts(self, paths, cache_data):
        """处理TTS生成步骤"""
        if not paths.reference_results.exists():
            return {
                'success': False,
                'message': '参考结果文件不存在',
                'error': f'文件不存在: {paths.reference_results}'
            }
        
        self.emit_step_progress('generate_tts', 5)
        self.logger.info("开始生成TTS音频...")
        result = generate_tts_from_reference(str(paths.reference_results), str(paths.tts_output_dir))
        self.emit_step_progress('generate_tts', 95)
        
        if not result.get('success', False):
            return {
                'success': False,
                'message': 'TTS生成失败',
                'error': result.get('error', '未知错误')
            }
        
        self._mark_step_completed(cache_data, 'generate_tts', {'result': result})
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        self.emit_step_progress('generate_tts', 100)
        return result
        
    def _process_align_audio(self, paths, cache_data):
        """处理音频对齐步骤"""
        self.emit_step_progress('align_audio', 10)
        self.logger.info("开始音频对齐...")
        result = align_audio_with_subtitles(
            tts_results_path=str(paths.tts_results),
            srt_path=str(paths.processed_subtitle),
            output_path=str(paths.aligned_audio)
        )
        
        self.emit_step_progress('align_audio', 70)
        
        # 保存对齐结果到JSON文件
        result_copy = result.copy()
        result_copy['saved_at'] = datetime.now().isoformat()
        with open(paths.aligned_results, 'w', encoding='utf-8') as f:
            json.dump(result_copy, f, ensure_ascii=False, indent=2)
        
        self.emit_step_progress('align_audio', 90)
        self._mark_step_completed(cache_data, 'align_audio', {'result': result})
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        self.emit_step_progress('align_audio', 100)
        return result
        
    def _process_generate_aligned_srt(self, paths, video_path, cache_data):
        """处理生成对齐字幕步骤"""
        self.emit_step_progress('generate_aligned_srt', 10)
        
        # 生成对齐后的SRT字幕
        generate_aligned_srt(
            str(paths.aligned_results),
            str(paths.processed_subtitle),
            str(paths.aligned_srt)
        )
        
        self.emit_step_progress('generate_aligned_srt', 60)
        
        # 检查原始字幕格式
        original_subtitle_path = str(paths.subtitle_path)
        original_subtitle_ext = Path(original_subtitle_path).suffix.lower()
        
        if original_subtitle_ext != '.srt':
            self.logger.info(f"检测到原始字幕格式为 {original_subtitle_ext}，正在转换...")
            
            if original_subtitle_ext == '.ass':
                aligned_ass_path = paths.output_dir / f"{Path(video_path).stem}_aligned.ass"
                sync_success = sync_srt_timestamps_to_ass(
                    original_subtitle_path,
                    str(paths.aligned_srt),
                    str(aligned_ass_path)
                )
                if sync_success:
                    self.logger.info(f"ASS字幕时间戳同步完成: {aligned_ass_path}")
                else:
                    self.logger.error("ASS字幕时间戳同步失败")
            else:
                aligned_subtitle_path = paths.output_dir / f"{Path(video_path).stem}_aligned{original_subtitle_ext}"
                convert_success = convert_subtitle(
                    str(paths.aligned_srt),
                    str(aligned_subtitle_path)
                )
                if convert_success:
                    self.logger.info(f"字幕格式转换完成: {aligned_subtitle_path}")
                else:
                    self.logger.error("字幕格式转换失败")
        
        self.emit_step_progress('generate_aligned_srt', 90)
        self._mark_step_completed(cache_data, 'generate_aligned_srt')
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        self.emit_step_progress('generate_aligned_srt', 100)
        return {'success': True}
        
    def _process_video_speed(self, paths, cache_data):
        """处理视频速度调整步骤"""
        self.emit_step_progress('process_video_speed', 10)
        self.logger.info("开始处理视频速度调整...")
        process_video_speed_adjustment(
            str(paths.silent_video),
            str(paths.processed_subtitle),
            str(paths.aligned_srt)
        )
        self.emit_step_progress('process_video_speed', 90)
        self._mark_step_completed(cache_data, 'process_video_speed')
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        self.emit_step_progress('process_video_speed', 100)
        return {'success': True}
        
    def _process_merge_audio_video(self, paths, cache_data):
        """处理音视频合并步骤"""
        self.emit_step_progress('merge_audio_video', 10)
        self.logger.info("开始合并音视频...")
        merge_audio_video(
            str(paths.speed_adjusted_video),
            str(paths.aligned_audio),
            str(paths.final_video)
        )
        self.emit_step_progress('merge_audio_video', 95)
        self._mark_step_completed(cache_data, 'merge_audio_video')
        self._save_pipeline_cache(paths.pipeline_cache, cache_data)
        self.emit_step_progress('merge_audio_video', 100)
        return {'success': True}


class VideoSubtitleMatcher:
    """视频字幕匹配器"""
    
    @staticmethod
    def find_video_subtitle_pairs(folder_path: str) -> List[Tuple[str, Optional[str]]]:
        """
        在文件夹中查找视频和对应的字幕文件（默认递归搜索子文件夹）
        
        Args:
            folder_path: 文件夹路径
        
        Returns:
            List of (video_path, subtitle_path) tuples
        """
        folder = Path(folder_path)
        if not folder.exists():
            return []
        
        # 支持的视频格式
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v', '.webm', '.ts'}
        # 支持的字幕格式
        subtitle_extensions = {'.srt', '.ass', '.ssa', '.sub', '.vtt'}
        
        # 递归查找所有视频文件
        video_files = []
        for ext in video_extensions:
            video_files.extend(folder.rglob(f"*{ext}"))
            video_files.extend(folder.rglob(f"*{ext.upper()}"))
        
        pairs = []
        for video_file in video_files:
            video_name = video_file.stem
            video_folder = video_file.parent
            
            # 查找对应的字幕文件
            subtitle_file = None
            
            # 完全匹配
            for ext in subtitle_extensions:
                potential_subtitle = video_folder / f"{video_name}{ext}"
                if potential_subtitle.exists():
                    subtitle_file = potential_subtitle
                    break
                    
                # 尝试大写扩展名
                potential_subtitle = video_folder / f"{video_name}{ext.upper()}"
                if potential_subtitle.exists():
                    subtitle_file = potential_subtitle
                    break
            
            # 如果没有完全匹配，尝试模糊匹配
            if not subtitle_file:
                # 在整个文件夹树中查找
                for file in video_folder.rglob("*"):
                    if file.suffix.lower() in subtitle_extensions:
                        file_stem = file.stem
                        # 检查是否包含视频文件名的主要部分
                        if video_name.lower() in file_stem.lower() or file_stem.lower() in video_name.lower():
                            subtitle_file = file
                            break
            
            pairs.append((str(video_file), str(subtitle_file) if subtitle_file else None))
        
        return pairs


class LogHandler(logging.Handler):
    """自定义日志处理器，用于将日志输出到GUI"""
    
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def emit(self, record):
        msg = self.format(record)
        self.text_widget.append(msg)


class DubbingWorkerThread(QThread):
    """配音处理工作线程"""
    
    # 信号定义
    processing_finished = Signal(bool, str, dict)  # 是否成功, 消息, 结果详情
    
    def __init__(self, video_path: str, subtitle_path: Optional[str] = None, 
                 resume_from_cache: bool = True):
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
                self.video_path, 
                self.subtitle_path, 
                self.resume_from_cache
            )
            
            if self.is_cancelled:
                return
                
            # 发送完成信号
            self.processing_finished.emit(
                result['success'], 
                result['message'], 
                result
            )
            
        except Exception as e:
            self.processing_finished.emit(False, f"处理失败: {str(e)}", {})


class BatchDubbingWorkerThread(QThread):
    """批量配音处理工作线程"""
    
    # 信号定义
    batch_progress = Signal(int, int)  # 当前项目, 总项目数
    item_finished = Signal(int, bool, str)  # 项目索引, 是否成功, 消息
    batch_finished = Signal(bool, str, dict)  # 是否成功, 消息, 结果详情
    
    def __init__(self, video_subtitle_pairs: List[Tuple[str, Optional[str]]], 
                 resume_from_cache: bool = True):
        super().__init__()
        self.pairs = video_subtitle_pairs
        self.resume_from_cache = resume_from_cache
        self.is_cancelled = False
        
        # 创建管道
        self.pipeline = DubbingPipeline()
        
    def cancel(self):
        """取消处理"""
        self.is_cancelled = True
        
    def run(self):
        """执行批量处理"""
        try:
            success_count = 0
            failed_count = 0
            results = []
            
            for i, (video_path, subtitle_path) in enumerate(self.pairs):
                if self.is_cancelled:
                    break
                    
                self.batch_progress.emit(i + 1, len(self.pairs))
                
                try:
                    result = self.pipeline.process_video(
                        video_path, 
                        subtitle_path, 
                        self.resume_from_cache
                    )
                    
                    results.append(result)
                    
                    if result['success']:
                        success_count += 1
                        self.item_finished.emit(i, True, "完成")
                    else:
                        failed_count += 1
                        self.item_finished.emit(i, False, result.get('message', '失败'))
                        
                except Exception as e:
                    failed_count += 1
                    self.item_finished.emit(i, False, str(e))
                    results.append({
                        'success': False,
                        'message': str(e),
                        'error': str(e)
                    })
            
            # 发送批量完成信号
            batch_result = {
                'success': failed_count == 0,
                'total_count': len(self.pairs),
                'success_count': success_count,
                'failed_count': failed_count,
                'results': results
            }
            
            message = f'批量处理完成: {success_count} 成功, {failed_count} 失败'
            self.batch_finished.emit(batch_result['success'], message, batch_result)
            
        except Exception as e:
            self.batch_finished.emit(False, f"批量处理失败: {str(e)}", {})


class DubbingGUI(QMainWindow):
    """DubbingX 主窗口"""
    
    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.batch_worker_thread = None
        self.gui_pipeline = GUIDubbingPipeline()
        self.pipeline = DubbingPipeline()
        
        # 设置窗口属性
        self.setWindowTitle("DubbingX - 智能视频配音系统")
        self.setGeometry(100, 100, 1400, 800)  # 减少高度
        
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
        
    def setup_theme(self):
        """设置应用主题"""
        self.setStyleSheet("""
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
        """)
        
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
        
        # 右侧面板（日志和进度）
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
        self.start_btn.setStyleSheet("""
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
        """)
        button_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("取消处理")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("""
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
        """)
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
        
        # 从缓存恢复选项
        self.resume_cache_checkbox = QCheckBox("从缓存恢复处理")
        self.resume_cache_checkbox.setChecked(True)
        options_layout.addWidget(self.resume_cache_checkbox)
        
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
        subtitle_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #333333;")
        layout.addWidget(subtitle_label, 1, 0)
        
        self.subtitle_path_edit = QLineEdit()
        self.subtitle_path_edit.setPlaceholderText("选择字幕文件（可选，自动匹配同名文件）...")
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
        folder_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #333333;")
        folder_layout.addWidget(folder_label)
        
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("选择包含视频和字幕的文件夹...")
        folder_layout.addWidget(self.folder_path_edit)
        
        self.folder_browse_btn = QPushButton("浏览文件夹")
        self.folder_browse_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(self.folder_browse_btn)
        
        self.scan_btn = QPushButton("扫描匹配")
        self.scan_btn.clicked.connect(self.scan_folder)
        self.scan_btn.setStyleSheet("""
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
        """)
        folder_layout.addWidget(self.scan_btn)
        
        layout.addLayout(folder_layout)
        
        # 文件列表表格
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(4)
        self.file_table.setHorizontalHeaderLabels(["视频文件", "字幕文件", "状态", "选择"])
        
        # 设置表格样式
        self.file_table.setStyleSheet("""
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
            QHeaderView::section {
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
        """)
        
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
        
        # 进度标签页
        progress_tab = self.create_progress_tab()
        tab_widget.addTab(progress_tab, "处理进度")
        
        # 日志标签页
        log_tab = self.create_log_tab()
        tab_widget.addTab(log_tab, "日志输出")
        
        # 结果标签页
        result_tab = self.create_result_tab()
        tab_widget.addTab(result_tab, "处理结果")
        
        layout.addWidget(tab_widget)
        
        return panel
        
    def create_progress_tab(self) -> QWidget:
        """创建进度标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)  # 减少间距
        layout.setContentsMargins(15, 15, 15, 15)  # 减少边距
        
        # 总体进度
        overall_group = QGroupBox("总体进度")
        overall_layout = QVBoxLayout(overall_group)
        overall_layout.setContentsMargins(15, 12, 15, 12)  # 减少内边距
        
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setMinimumHeight(24)  # 减少高度
        overall_layout.addWidget(self.overall_progress)
        
        self.overall_status_label = QLabel("准备就绪")
        self.overall_status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333; background: transparent;")
        overall_layout.addWidget(self.overall_status_label)
        
        layout.addWidget(overall_group)
        
        # 批量进度（批量模式时显示）
        self.batch_progress_group = QGroupBox("批量进度")
        batch_progress_layout = QVBoxLayout(self.batch_progress_group)
        batch_progress_layout.setContentsMargins(15, 12, 15, 12)  # 减少内边距
        
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setMinimumHeight(24)  # 减少高度
        batch_progress_layout.addWidget(self.batch_progress)
        
        self.batch_status_label = QLabel("等待开始")
        self.batch_status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333; background: transparent;")
        batch_progress_layout.addWidget(self.batch_status_label)
        
        layout.addWidget(self.batch_progress_group)
        self.batch_progress_group.hide()
        
        # 详细步骤进度
        steps_group = QGroupBox("处理步骤")
        steps_layout = QVBoxLayout(steps_group)
        steps_layout.setContentsMargins(15, 12, 15, 12)  # 减少内边距
        steps_layout.setSpacing(5)  # 减少间距
        
        # 创建步骤列表
        self.step_labels = {}
        self.step_progress_bars = {}
        
        steps = [
            ("preprocess_subtitle", "1. 预处理字幕"),
            ("separate_media", "2. 分离音视频"),
            ("generate_reference_audio", "3. 生成参考音频"),
            ("generate_tts", "4. 生成TTS音频"),
            ("align_audio", "5. 音频对齐"),
            ("generate_aligned_srt", "6. 生成对齐字幕"),
            ("process_video_speed", "7. 处理视频速度"),
            ("merge_audio_video", "8. 合并音视频")
        ]
        
        for step_id, step_name in steps:
            step_frame = QFrame()
            step_frame.setStyleSheet("""
                QFrame { 
                    background-color: #f9f9f9; 
                    border: 1px solid #e0e0e0;
                    border-radius: 4px; 
                    padding: 4px; 
                }
            """)
            step_layout = QHBoxLayout(step_frame)
            step_layout.setContentsMargins(10, 6, 10, 6)  # 减少边距
            
            label = QLabel(step_name)
            label.setMinimumWidth(180)
            label.setStyleSheet("font-size: 12px; font-weight: bold; background: transparent; border: none; color: #333333;")
            step_layout.addWidget(label)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setMinimumHeight(20)  # 减少高度
            step_layout.addWidget(progress_bar)
            
            status_label = QLabel("等待中")
            status_label.setMinimumWidth(90)
            status_label.setStyleSheet("font-size: 11px; color: #666; background: transparent; border: none; font-weight: normal;")
            step_layout.addWidget(status_label)
            
            steps_layout.addWidget(step_frame)
            
            self.step_labels[step_id] = status_label
            self.step_progress_bars[step_id] = progress_bar
            
        layout.addWidget(steps_group)
        
        return tab
        
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
        self.log_text.setStyleSheet("""
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
        """)
        layout.addWidget(self.log_text)
        
        return tab
        
    def create_result_tab(self) -> QWidget:
        """创建结果标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 结果信息
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("处理完成后将在此显示结果信息...")
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 20px;
                font-size: 12px;
                line-height: 1.6;
                color: #333;
            }
        """)
        layout.addWidget(self.result_text)
        
        return tab
        
    def setup_logging(self):
        """设置日志"""
        # 创建日志处理器
        log_handler = LogHandler(self.log_text)
        log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # 添加到根日志记录器
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.INFO)
        
    def on_mode_changed(self):
        """模式切换处理"""
        if self.single_mode_radio.isChecked():
            self.current_mode = "single"
            self.single_file_panel.show()
            self.batch_panel.hide()
            self.batch_progress_group.hide()
        else:
            self.current_mode = "batch"
            self.single_file_panel.hide()
            self.batch_panel.show()
            self.batch_progress_group.show()
        
    def browse_video_file(self):
        """浏览选择视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v *.webm);;所有文件 (*)"
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
            "字幕文件 (*.srt *.ass *.ssa *.sub *.vtt);;所有文件 (*)"
        )
        
        if file_path:
            self.subtitle_path_edit.setText(file_path)
            self.current_subtitle_path = file_path
            
    def browse_folder(self):
        """浏览选择文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择包含视频和字幕的文件夹",
            ""
        )
        
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            self.current_folder_path = folder_path
            
    def scan_folder(self):
        """扫描文件夹中的视频字幕对"""
        if not hasattr(self, 'current_folder_path') or not self.current_folder_path:
            QMessageBox.warning(self, "警告", "请先选择文件夹！")
            return
            
        try:
            # 递归搜索视频和字幕文件
            pairs = VideoSubtitleMatcher.find_video_subtitle_pairs(self.current_folder_path)
            
            if not pairs:
                QMessageBox.information(self, "信息", "未找到任何视频文件！")
                return
                
            self.video_subtitle_pairs = pairs
            self.update_file_table()
            
            matched_count = sum(1 for _, subtitle in pairs if subtitle)
            QMessageBox.information(
                self, 
                "扫描完成", 
                f"找到 {len(pairs)} 个视频文件，其中 {matched_count} 个有匹配的字幕文件。\n（已自动搜索所有子文件夹）"
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
            
            # 字幕文件名
            if subtitle_path:
                subtitle_name = Path(subtitle_path).name
                subtitle_item = QTableWidgetItem(subtitle_name)
                subtitle_item.setToolTip(subtitle_path)
                self.file_table.setItem(i, 1, subtitle_item)
            else:
                item = QTableWidgetItem("未找到匹配字幕")
                # 设置红色文字
                item.setForeground(QColor("#dc3545"))
                self.file_table.setItem(i, 1, item)
            
            # 状态
            if subtitle_path:
                status_item = QTableWidgetItem("就绪")
                # 设置绿色文字
                status_item.setForeground(QColor("#198754"))
            else:
                status_item = QTableWidgetItem("缺少字幕")
                # 设置红色文字
                status_item.setForeground(QColor("#dc3545"))
            self.file_table.setItem(i, 2, status_item)
            
            # 选择框
            checkbox = QCheckBox()
            checkbox.setChecked(subtitle_path is not None)  # 只选择有字幕的项
            self.file_table.setCellWidget(i, 3, checkbox)
            
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
        subtitle_extensions = ['.srt', '.ass', '.ssa', '.sub', '.vtt']
        
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
        self.overall_status_label.setText("处理中...")
        
        # 重置进度
        self.reset_progress()
        
        try:
            # 创建工作线程
            self.worker_thread = DubbingWorkerThread(
                self.current_video_path,
                self.current_subtitle_path if self.current_subtitle_path else None,
                self.resume_cache_checkbox.isChecked()
            )
            
            # 连接信号
            self.worker_thread.processing_finished.connect(self.processing_finished)
            
            # 连接GUI管道的信号
            self.gui_pipeline = self.worker_thread.pipeline
            self.gui_pipeline.step_started.connect(self.step_started)
            self.gui_pipeline.step_progress.connect(self.step_progress)
            self.gui_pipeline.step_completed.connect(self.step_completed)
            self.gui_pipeline.log_message.connect(self.log_text.append)
            
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
            self.overall_status_label.setText("启动失败")
        
    def start_batch_processing(self):
        """开始批量处理"""
        selected_pairs = self.get_selected_pairs()
        
        if not selected_pairs:
            QMessageBox.warning(self, "警告", "请至少选择一个要处理的视频！")
            return
            
        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.overall_status_label.setText("批量处理中...")
        self.batch_status_label.setText(f"准备处理 {len(selected_pairs)} 个文件...")
        
        # 重置进度
        self.reset_progress()
        self.batch_progress.setValue(0)
        
        # 在批量模式下，显示批量进度信息
        self.batch_status_label.setText(f"即将处理 {len(selected_pairs)} 个文件...")
        
        try:
            # 创建批量工作线程
            self.batch_worker_thread = BatchDubbingWorkerThread(
                selected_pairs,
                self.resume_cache_checkbox.isChecked()
            )
            
            # 连接信号
            self.batch_worker_thread.batch_progress.connect(self.batch_progress_updated)
            self.batch_worker_thread.item_finished.connect(self.batch_item_finished)
            self.batch_worker_thread.batch_finished.connect(self.batch_finished)
            
            # 启动线程
            self.batch_worker_thread.start()
            
        except Exception as e:
            import traceback
            error_msg = f"启动批量处理失败: {str(e)}\n详细错误:\n{traceback.format_exc()}"
            self.log_text.append(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            
            # 恢复UI状态
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.overall_status_label.setText("批量处理启动失败")
            self.batch_status_label.setText("启动失败")
        
    def cancel_processing(self):
        """取消处理"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
            self.worker_thread.wait()
            
        if self.batch_worker_thread and self.batch_worker_thread.isRunning():
            self.batch_worker_thread.cancel()
            self.batch_worker_thread.wait()
            
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.overall_status_label.setText("已取消")
        self.batch_status_label.setText("已取消")
        
    def reset_progress(self):
        """重置进度显示"""
        self.overall_progress.setValue(0)
        
        for step_id, progress_bar in self.step_progress_bars.items():
            progress_bar.setValue(0)
            
        for step_id, label in self.step_labels.items():
            label.setText("等待中")
            label.setStyleSheet("font-size: 11px; color: #666; background: transparent; border: none; font-weight: normal;")
            
    def step_started(self, step_id: str, step_name: str):
        """步骤开始"""
        if step_id in self.step_labels:
            self.step_labels[step_id].setText("进行中")
            self.step_labels[step_id].setStyleSheet("font-size: 11px; color: #2196F3; font-weight: bold; background: transparent; border: none;")
            
    def step_progress(self, step_id: str, progress: int):
        """步骤进度更新"""
        if step_id in self.step_progress_bars:
            self.step_progress_bars[step_id].setValue(progress)
            
        if step_id in self.step_labels:
            if progress < 100:
                self.step_labels[step_id].setText(f"进行中 {progress}%")
                self.step_labels[step_id].setStyleSheet("font-size: 11px; color: #2196F3; font-weight: bold; background: transparent; border: none;")
            else:
                # 等待 step_completed 来设置最终状态
                pass
            
    def step_completed(self, step_id: str, success: bool, message: str):
        """步骤完成"""
        if step_id in self.step_progress_bars:
            self.step_progress_bars[step_id].setValue(100)
            
        if step_id in self.step_labels:
            if success:
                if "缓存恢复" in message:
                    self.step_labels[step_id].setText("缓存恢复")
                    self.step_labels[step_id].setStyleSheet("font-size: 11px; color: #FF9800; font-weight: bold; background: transparent; border: none;")
                else:
                    self.step_labels[step_id].setText("完成")
                    self.step_labels[step_id].setStyleSheet("font-size: 11px; color: #4CAF50; font-weight: bold; background: transparent; border: none;")
            else:
                self.step_labels[step_id].setText("失败")
                self.step_labels[step_id].setStyleSheet("font-size: 11px; color: #f44336; font-weight: bold; background: transparent; border: none;")
            
        # 更新总体进度 - 基于当前所有步骤的实际进度
        total_progress = sum(self.step_progress_bars[step_id].value() for step_id in self.step_progress_bars.keys())
        total_steps = len(self.step_progress_bars) * 100  # 每个步骤最大100分
        overall_progress = int((total_progress / total_steps) * 100)
        self.overall_progress.setValue(overall_progress)
        
    def batch_progress_updated(self, current: int, total: int):
        """批量进度更新"""
        progress = int((current / total) * 100)
        self.batch_progress.setValue(progress)
        self.batch_status_label.setText(f"正在处理第 {current}/{total} 个文件...")
        
        # 更新总体进度条显示批量进度
        self.overall_progress.setValue(progress)
        
    def batch_item_finished(self, index: int, success: bool, message: str):
        """批量处理中单个项目完成"""
        # 更新表格状态
        if index < self.file_table.rowCount():
            if success:
                status_item = QTableWidgetItem("完成")
                status_item.setForeground(QColor("#198754"))
            else:
                status_item = QTableWidgetItem(f"失败: {message}")
                status_item.setForeground(QColor("#dc3545"))
                status_item.setToolTip(message)
            self.file_table.setItem(index, 2, status_item)
            
    def batch_finished(self, success: bool, message: str, result: Dict[str, Any]):
        """批量处理完成"""
        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.overall_status_label.setText("批量处理完成")
        self.batch_status_label.setText(message)
        self.batch_progress.setValue(100)
        self.overall_progress.setValue(100)
        
        # 重置所有步骤进度条
        for step_id in self.step_progress_bars:
            self.step_progress_bars[step_id].setValue(100)
            if success:
                self.step_labels[step_id].setText("批量完成")
                self.step_labels[step_id].setStyleSheet("font-size: 10px; color: #4CAF50; font-weight: bold; background: transparent; border: none;")
            else:
                self.step_labels[step_id].setText("批量完成")
                self.step_labels[step_id].setStyleSheet("font-size: 10px; color: #FF9800; font-weight: bold; background: transparent; border: none;")
        
        # 显示结果信息
        result_info = f"""批量处理完成！

处理统计:
• 总文件数: {result.get('total_count', 0)}
• 成功处理: {result.get('success_count', 0)}
• 处理失败: {result.get('failed_count', 0)}
• 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

提示: 查看表格了解每个文件的详细状态
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
            self.overall_status_label.setText("处理完成")
            self.overall_progress.setValue(100)
            
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
            self.overall_status_label.setText("处理失败")
            QMessageBox.critical(self, "失败", message)
            
    def show_cache_info(self):
        """显示缓存信息"""
        if self.current_mode == "single":
            if not self.current_video_path:
                QMessageBox.warning(self, "警告", "请先选择视频文件！")
                return
            video_path = self.current_video_path
        else:
            QMessageBox.information(self, "信息", "批量模式下请单独查看各文件的缓存信息")
            return
            
        try:
            cache_info = self.pipeline.get_pipeline_cache_info(video_path)
            
            if cache_info['success']:
                if cache_info['cache_exists']:
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
                QMessageBox.warning(self, "错误", cache_info['message'])
                
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
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                result = self.pipeline.clear_pipeline_cache(video_path)
                
                if result['success']:
                    QMessageBox.information(self, "成功", result['message'])
                else:
                    QMessageBox.warning(self, "错误", result['message'])
                    
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
            
            if result['success']:
                QMessageBox.information(self, "成功", result['message'])
            else:
                QMessageBox.warning(self, "错误", result['message'])
                
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
            "文本文件 (*.txt);;所有文件 (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "成功", f"日志已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存日志失败: {str(e)}")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("DubbingX")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("DubbingX Team")
    
    # 创建主窗口
    window = DubbingGUI()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == '__main__':
    main()