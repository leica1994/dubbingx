"""流水线处理器 - 基于队列+监听模式实现真正的并行处理"""

import logging
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Signal

from core.util import sanitize_filename
from core.cache import UnifiedCacheManager, StepCacheManager
from core.gui.status import StatusEventManager
from .pipeline import Task, TaskScheduler
from .pipeline.processors import (
    AlignAudioProcessor,
    GenerateAlignedSrtProcessor,
    GenerateReferenceAudioProcessor,
    GenerateTTSProcessor,
    MergeAudioVideoProcessor,
    PreprocessSubtitleProcessor,
    ProcessVideoSpeedProcessor,
    SeparateMediaProcessor,
)



# 注意：原来的StatusNotificationManager已被替换为更高效的异步StatusEventManager

class DubbingPaths:
    """配音文件路径管理"""

    def __init__(
        self,
        video_path: str,
        subtitle_path: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.video_ext = self.video_path.suffix

        self.subtitle_path = self._resolve_subtitle_path(subtitle_path)
        self.output_dir = self._setup_output_directory(output_dir)
        self._initialize_paths()

    def _resolve_subtitle_path(self, subtitle_path: Optional[str]) -> Path:
        """解析字幕文件路径"""
        if subtitle_path:
            return Path(subtitle_path)
        return self._find_matching_subtitle()

    def _setup_output_directory(self, output_dir: Optional[Path]) -> Path:
        """设置输出目录"""
        if output_dir:
            return output_dir

        video_parent = self.video_path.parent
        outputs_dir = video_parent / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        clean_video_name = self._sanitize_filename(self.video_name)
        output_dir = outputs_dir / clean_video_name
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def _find_matching_subtitle(self) -> Path:
        """自动匹配同名字幕文件"""
        video_dir = self.video_path.parent
        video_name = self.video_name
        subtitle_extensions = [".srt", ".ass", ".ssa", ".sub", ".vtt"]

        for ext in subtitle_extensions:
            subtitle_file = video_dir / f"{video_name}{ext}"
            if subtitle_file.exists():
                return subtitle_file

        # 模糊匹配
        for file in video_dir.glob(f"{video_name}*"):
            if file.suffix.lower() in subtitle_extensions:
                return file

        raise FileNotFoundError(f"无法找到与视频 {self.video_name} 匹配的字幕文件")

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
        return sanitize_filename(filename)

    def _initialize_paths(self):
        """初始化所有文件路径"""
        # 为分离后的媒体文件创建子目录
        self._media_separation_dir = self.output_dir / "media_separation"

        # 初始化所有文件路径
        self._processed_subtitle = self.output_dir / f"{self.video_name}_processed.srt"
        self._vocal_audio = self._media_separation_dir / f"{self.video_name}_vocal.wav"
        self._background_audio = (
            self._media_separation_dir / f"{self.video_name}_background.wav"
        )
        self._silent_video = (
            self._media_separation_dir / f"{self.video_name}_silent{self.video_ext}"
        )
        self._reference_audio_dir = self.output_dir / "reference_audio"
        self._tts_output_dir = self.output_dir / "tts_output"
        self._aligned_audio_dir = self.output_dir / "aligned_audio"
        self._adjusted_video_dir = self.output_dir / "adjusted_video"
        self._reference_results = (
            self._reference_audio_dir
            / f"{self.video_name}_vocal_reference_audio_results.json"
        )
        self._tts_results = self._tts_output_dir / "tts_generation_results.json"
        self._aligned_results = (
            self._aligned_audio_dir / "aligned_tts_generation_results.json"
        )
        self._aligned_audio = (
            self._aligned_audio_dir / "aligned_tts_generation_results.wav"
        )
        self._aligned_srt = (
            self._aligned_audio_dir / "aligned_tts_generation_aligned.srt"
        )
        self._final_video = (
            self.output_dir / f"{self.video_name}_dubbed{self.video_ext}"
        )
        self._speed_adjusted_video = (
            self._adjusted_video_dir
            / f"final_speed_adjusted_{self.video_name}_silent{self.video_ext}"
        )
        self._pipeline_cache = (
            self.output_dir / f"{self.video_name}_pipeline_cache.json"
        )

        # 创建必要的子目录
        for dir_path in [
            self._media_separation_dir,
            self._reference_audio_dir,
            self._tts_output_dir,
            self._aligned_audio_dir,
            self._adjusted_video_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

    # 属性访问器
    @property
    def processed_subtitle(self) -> Path:
        return self._processed_subtitle

    @property
    def vocal_audio(self) -> Path:
        return self._vocal_audio

    @property
    def background_audio(self) -> Path:
        return self._background_audio

    @property
    def silent_video(self) -> Path:
        return self._silent_video

    @property
    def media_separation_dir(self) -> Path:
        return self._media_separation_dir

    @property
    def reference_audio_dir(self) -> Path:
        return self._reference_audio_dir

    @property
    def tts_output_dir(self) -> Path:
        return self._tts_output_dir

    @property
    def aligned_audio_dir(self) -> Path:
        return self._aligned_audio_dir

    @property
    def adjusted_video_dir(self) -> Path:
        return self._adjusted_video_dir

    @property
    def reference_results(self) -> Path:
        return self._reference_results

    @property
    def tts_results(self) -> Path:
        return self._tts_results

    @property
    def aligned_results(self) -> Path:
        return self._aligned_results

    @property
    def aligned_audio(self) -> Path:
        return self._aligned_audio

    @property
    def aligned_srt(self) -> Path:
        return self._aligned_srt

    @property
    def final_video(self) -> Path:
        return self._final_video

    @property
    def speed_adjusted_video(self) -> Path:
        return self._speed_adjusted_video

    @property
    def pipeline_cache(self) -> Path:
        return self._pipeline_cache

    # Get方法
    def get_processed_subtitle(self) -> Path:
        return self._processed_subtitle

    def get_vocal_audio(self) -> Path:
        return self._vocal_audio

    def get_background_audio(self) -> Path:
        return self._background_audio

    def get_silent_video(self) -> Path:
        return self._silent_video

    def get_media_separation_dir(self) -> Path:
        return self._media_separation_dir

    def get_reference_audio_dir(self) -> Path:
        return self._reference_audio_dir

    def get_tts_output_dir(self) -> Path:
        return self._tts_output_dir

    def get_aligned_audio_dir(self) -> Path:
        return self._aligned_audio_dir

    def get_adjusted_video_dir(self) -> Path:
        return self._adjusted_video_dir

    def get_reference_results(self) -> Path:
        return self._reference_results

    def get_tts_results(self) -> Path:
        return self._tts_results

    def get_aligned_results(self) -> Path:
        return self._aligned_results

    def get_aligned_audio(self) -> Path:
        return self._aligned_audio

    def get_aligned_srt(self) -> Path:
        return self._aligned_srt

    def get_final_video(self) -> Path:
        return self._final_video

    def get_speed_adjusted_video(self) -> Path:
        return self._speed_adjusted_video

    def get_pipeline_cache(self) -> Path:
        return self._pipeline_cache


class StreamlinePipeline(QObject):
    """基于队列+监听模式的流水线处理器"""

    step_status_changed = Signal(
        str, int, str, str
    )  # task_id, step_id, status, message

    def __init__(self, output_dir: Optional[str] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir) if output_dir else None
        
        # 初始化异步状态事件管理器
        self.status_event_manager = StatusEventManager(self)
        
        # 统一缓存管理器字典 {task_id: UnifiedCacheManager}
        self.cache_managers: Dict[str, UnifiedCacheManager] = {}
        
        # 连接异步信号到现有GUI信号
        signal_emitter = self.status_event_manager.get_signal_emitter()
        signal_emitter.step_status_changed.connect(self.step_status_changed)
        signal_emitter.step_progress_changed.connect(self._on_step_progress_changed)
        
        self.task_scheduler = self._initialize_task_scheduler()
        self.logger.info("StreamlinePipeline 初始化完成")

    def _on_step_progress_changed(self, task_id: str, step_id: int, progress: float, current: int, total: int, message: str):
        """处理步骤进度变化信号（可以在这里添加额外的进度处理逻辑）"""
        # 目前只是转发，未来可以添加进度汇总、状态检查等逻辑
        pass

    def _initialize_task_scheduler(self):
        """初始化任务调度器和处理器"""
        task_scheduler = TaskScheduler()
        processors = [
            PreprocessSubtitleProcessor(),
            SeparateMediaProcessor(),
            GenerateReferenceAudioProcessor(),
            GenerateTTSProcessor(),
            AlignAudioProcessor(),
            GenerateAlignedSrtProcessor(),
            ProcessVideoSpeedProcessor(),
            MergeAudioVideoProcessor(),
        ]

        for processor in processors:
            task_scheduler.register_processor(processor)

        return task_scheduler

    def notify_step_status(
        self, task_id: str, step_id: int, status: str, message: str = ""
    ):
        """通过异步状态管理器统一处理状态通知"""
        try:
            if status == "processing":
                self.status_event_manager.notify_step_started(task_id, step_id, message=message)
            elif status == "completed":
                self.status_event_manager.notify_step_completed(task_id, step_id, message=message)
            elif status == "failed":
                self.status_event_manager.notify_step_failed(task_id, step_id, error_message=message, message=message)
            else:
                # 其他状态，记录日志但不处理
                self.logger.debug(f"未知状态类型: {status}, 任务: {task_id}, 步骤: {step_id}")
            return True
        except Exception as e:
            self.logger.error(f"状态通知失败: {e}")
            return False

    def process_batch_streamline(
        self,
        video_subtitle_pairs: List[Tuple[str, Optional[str]]],
        resume_from_cache: bool = True,
        enable_vocal_separation: bool = False,
    ) -> Dict[str, Any]:
        """使用流水线模式批量处理视频"""
        start_time = time.time()
        self.logger.info(f"开始流水线批量处理 {len(video_subtitle_pairs)} 个视频")
        self.logger.info(f"音频分离模式: {'完整分离' if enable_vocal_separation else '快速模式'}")

        tasks = self._create_tasks(video_subtitle_pairs, resume_from_cache, enable_vocal_separation)

        try:
            results = self._execute_pipeline(tasks)
        except Exception as e:
            return self._create_error_result(e, video_subtitle_pairs, start_time)

        return self._format_final_results(results, video_subtitle_pairs, start_time)

    def _create_tasks(
        self,
        video_subtitle_pairs: List[Tuple[str, Optional[str]]],
        resume_from_cache: bool,
        enable_vocal_separation: bool,
    ):
        """创建任务列表"""
        tasks = []
        for i, (video_path, subtitle_path) in enumerate(video_subtitle_pairs):
            task_id = f"streamline_task_{i:03d}_{Path(video_path).stem}"
            paths = DubbingPaths(video_path, subtitle_path, self.output_dir)

            task = self._create_or_restore_task(
                task_id, video_path, subtitle_path, paths, resume_from_cache
            )
            self._setup_task_paths(task, paths)
            
            # 设置音频分离模式
            if not hasattr(task, 'options'):
                task.options = {}
            task.options['enable_vocal_separation'] = enable_vocal_separation
            
            tasks.append(task)

        return tasks

    def _create_or_restore_task(
        self,
        task_id: str,
        video_path: str,
        subtitle_path: str,
        paths: DubbingPaths,
        resume_from_cache: bool,
    ):
        """创建或从缓存恢复任务"""
        task = None
        
        # 初始化统一缓存管理器
        cache_manager = UnifiedCacheManager(paths.output_dir)
        self.cache_managers[task_id] = cache_manager
        
        # 注册缓存管理器到异步状态管理器
        self.status_event_manager.register_cache_manager(task_id, cache_manager)
        
        if resume_from_cache:
            # 使用统一缓存系统恢复任务
            try:
                # 初始化缓存结构（如果不存在）
                cache_manager.initialize_cache(task_id, video_path, subtitle_path)
                
                # 获取任务恢复点
                resume_point = cache_manager.get_task_resume_point()
                
                if resume_point > 0:
                    # 有已完成的步骤，从缓存恢复
                    completed_steps = cache_manager.get_completed_steps()
                    self.logger.info(f"从缓存恢复任务: {task_id}, 已完成步骤: {list(completed_steps)}, 恢复点: {resume_point}")
                    
                    # 创建任务并设置恢复状态
                    task = Task(
                        task_id=task_id, 
                        video_path=video_path, 
                        subtitle_path=subtitle_path
                    )
                    task.current_step = resume_point
                    
                    # 设置已完成的步骤结果
                    for step_id in completed_steps:
                        step_result = cache_manager.load_step_result(step_id)
                        if step_result:
                            task.step_results[str(step_id)] = {
                                "success": True,
                                "message": f"从缓存恢复: 步骤 {step_id}",
                                "data": step_result,
                                "error": None,
                                "processing_time": 0.0,
                                "partial_success": False,
                                "step_detail": None,
                            }
                else:
                    self.logger.info(f"缓存中无已完成步骤，创建新任务: {task_id}")
                    
            except Exception as e:
                self.logger.warning(f"从缓存恢复任务失败: {e}，创建新任务")
        else:
            # 初始化新的缓存结构
            cache_manager.initialize_cache(task_id, video_path, subtitle_path)

        if task is None:
            task = Task(
                task_id=task_id, video_path=video_path, subtitle_path=subtitle_path
            )

        # 重要：确保pipeline_ref始终被正确设置
        task.pipeline_ref = self
        self.logger.debug(f"任务 {task_id} 的pipeline_ref已设置")
        return task

    def get_cache_manager(self, task_id: str) -> Optional[UnifiedCacheManager]:
        """获取任务的统一缓存管理器"""
        return self.cache_managers.get(task_id)

    def get_step_cache_manager(self, task_id: str, step_id: int, step_name: str) -> Optional[StepCacheManager]:
        """获取步骤专用缓存管理器"""
        unified_cache = self.get_cache_manager(task_id)
        if unified_cache:
            return StepCacheManager(unified_cache, step_id, step_name)
        return None

    def _setup_task_paths(self, task, paths: DubbingPaths):
        """设置任务路径信息"""
        task.paths = {
            "video_path": str(paths.video_path),
            "subtitle_path": str(paths.subtitle_path),
            "processed_subtitle": str(paths.processed_subtitle),
            "vocal_audio": str(paths.vocal_audio),
            "background_audio": str(paths.background_audio),
            "silent_video": str(paths.silent_video),
            "output_dir": str(paths.output_dir),
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
        }

    def _execute_pipeline(self, tasks: list) -> Dict[str, Dict[str, Any]]:
        """执行流水线处理"""
        with self.task_scheduler:
            submit_results = self.task_scheduler.submit_batch_tasks(tasks)
            successful_submits = sum(submit_results)

            self.logger.info(f"成功提交 {successful_submits}/{len(tasks)} 个任务")

            if successful_submits == 0:
                raise RuntimeError("没有任务成功提交")

            self.logger.info("等待所有任务完成...")
            completed = self.task_scheduler.wait_for_completion(timeout=None)

            if not completed:
                self.logger.warning("任务处理被中断或出现异常")

            return self.task_scheduler.get_all_tasks_status()

    def _create_error_result(
        self, exception: Exception, video_subtitle_pairs: list, start_time: float
    ) -> Dict[str, Any]:
        """创建错误结果"""
        self.logger.error(f"流水线处理异常: {exception}", exc_info=True)
        return {
            "success": False,
            "message": f"流水线处理失败: {str(exception)}",
            "error": str(exception),
            "total_count": len(video_subtitle_pairs),
            "success_count": 0,
            "failed_count": len(video_subtitle_pairs),
            "total_time": time.time() - start_time,
            "results": [],
        }

    def _format_final_results(
        self,
        all_tasks_status: Dict[str, Dict[str, Any]],
        video_subtitle_pairs: list,
        start_time: float,
    ) -> Dict[str, Any]:
        """格式化最终结果"""
        results = self._format_streamline_results(
            all_tasks_status, video_subtitle_pairs
        )
        total_time = time.time() - start_time
        success_count = sum(1 for result in results if result["success"])
        failed_count = len(results) - success_count

        self.logger.info(f"流水线批量处理完成，耗时 {total_time:.2f}s")
        self.logger.info(f"成功: {success_count}, 失败: {failed_count}")

        return {
            "success": failed_count == 0,
            "message": f"流水线批量处理完成: {success_count} 成功, {failed_count} 失败",
            "total_count": len(video_subtitle_pairs),
            "success_count": success_count,
            "failed_count": failed_count,
            "total_time": total_time,
            "results": results,
        }

    def _format_streamline_results(
        self,
        all_tasks_status: Dict[str, Dict[str, Any]],
        video_subtitle_pairs: List[Tuple[str, Optional[str]]],
    ) -> List[Dict[str, Any]]:
        """格式化流水线处理结果"""
        results = []

        # 合并所有任务状态
        all_tasks = {}
        all_tasks.update(all_tasks_status.get("active", {}))
        all_tasks.update(all_tasks_status.get("completed", {}))
        all_tasks.update(all_tasks_status.get("failed", {}))

        for i, (video_path, subtitle_path) in enumerate(video_subtitle_pairs):
            task_id = f"streamline_task_{i:03d}_{Path(video_path).stem}"
            task_status = all_tasks.get(task_id)

            if not task_status:
                # 任务未找到
                results.append(
                    {
                        "success": False,
                        "task_id": task_id,
                        "video_path": video_path,
                        "subtitle_path": subtitle_path,
                        "message": "任务未找到",
                        "error": "任务可能提交失败",
                    }
                )
                continue

            # 判断任务是否成功
            status = task_status.get("status", "unknown")
            success = status == "completed"

            result = {
                "success": success,
                "task_id": task_id,
                "video_path": video_path,
                "subtitle_path": subtitle_path,
                "status": status,
                "current_step": task_status.get("current_step", 0),
                "progress": task_status.get("progress", 0.0),
                "processing_time": task_status.get("processing_time", 0.0),
                "message": "流水线处理完成" if success else "流水线处理失败",
            }

            if success:
                # 任务成功，添加输出信息
                output_dir = Path(video_path).parent / "outputs" / Path(video_path).stem
                final_video = (
                    output_dir
                    / f"{Path(video_path).stem}_dubbed{Path(video_path).suffix}"
                )

                result.update(
                    {
                        "output_file": str(final_video) if final_video.exists() else "",
                        "output_dir": str(output_dir),
                    }
                )
            else:
                # 任务失败，添加错误信息
                result.update(
                    {
                        "error": task_status.get("error_message", "未知错误"),
                        "retry_count": task_status.get("retry_count", 0),
                    }
                )

            results.append(result)

        return results

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """获取流水线统计信息"""
        if not hasattr(self, "task_scheduler"):
            return {"error": "流水线未初始化"}

        return self.task_scheduler.get_scheduler_stats()

    def cancel_all_tasks(self) -> int:
        """取消所有任务"""
        if not hasattr(self, "task_scheduler"):
            return 0

        return self.task_scheduler.cancel_all_tasks()

    def is_running(self) -> bool:
        """检查流水线是否在运行"""
        if not hasattr(self, "task_scheduler"):
            return False

        return self.task_scheduler.is_running()
