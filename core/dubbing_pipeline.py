import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.util import sanitize_filename


class DubbingPaths:
    """配音文件路径管理类"""

    def __init__(
            self,
            video_path: str,
            subtitle_path: Optional[str] = None,
            output_dir: Optional[Path] = None,
    ):
        """
        初始化配音路径管理器

        Args:
            video_path: 视频文件路径
            subtitle_path: 字幕文件路径，如果为None则自动匹配同名字幕文件
            output_dir: 输出目录路径，如果为None则使用视频父目录下的outputs
        """
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.video_ext = self.video_path.suffix  # 保存原始视频格式

        # 处理字幕文件路径
        if subtitle_path:
            self.subtitle_path = Path(subtitle_path)
        else:
            # 自动匹配同名字幕文件
            self.subtitle_path = self._find_matching_subtitle()

        if output_dir:
            self.output_dir = output_dir
        else:
            # 使用视频父目录下的outputs目录，并在其下创建文件名目录
            video_parent = self.video_path.parent
            outputs_dir = video_parent / "outputs"
            outputs_dir.mkdir(exist_ok=True)

            # 清理文件名并创建子目录
            clean_video_name = self._sanitize_filename(self.video_name)
            self.output_dir = outputs_dir / clean_video_name
            self.output_dir.mkdir(exist_ok=True)

        # 初始化所有路径
        self._initialize_paths()

    def _find_matching_subtitle(self) -> Path:
        """自动匹配同名字幕文件"""
        video_dir = self.video_path.parent
        video_name = self.video_name

        # 常见的字幕文件扩展名
        subtitle_extensions = [".srt", ".ass", ".ssa", ".sub", ".vtt"]

        for ext in subtitle_extensions:
            subtitle_file = video_dir / f"{video_name}{ext}"
            if subtitle_file.exists():
                return subtitle_file

        # 如果没有找到，尝试模糊匹配
        for file in video_dir.glob(f"{video_name}*"):
            if file.suffix.lower() in subtitle_extensions:
                return file

        # 如果仍然没有找到，抛出异常
        raise FileNotFoundError(f"无法找到与视频 {self.video_name} 匹配的字幕文件")

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，将不支持的字符转为下划线，统一格式避免重复目录"""
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


class StreamlinePipeline:
    """基于队列+监听模式的流水线处理器
    
    实现真正的流水线并行处理，支持不同任务在不同步骤同时执行
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化流水线处理器
        
        Args:
            output_dir: 输出目录路径
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir) if output_dir else None
        
        # 导入流水线组件
        from .pipeline import (
            TaskScheduler,
            Task,
            TaskStatus,
        )
        from .pipeline.processors import (
            PreprocessSubtitleProcessor,
            SeparateMediaProcessor,
            GenerateReferenceAudioProcessor,
            GenerateTTSProcessor,
            AlignAudioProcessor,
            GenerateAlignedSrtProcessor,
            ProcessVideoSpeedProcessor,
            MergeAudioVideoProcessor,
        )
        
        # 初始化任务调度器
        self.task_scheduler = TaskScheduler()
        
        # 注册所有步骤处理器
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
            self.task_scheduler.register_processor(processor)
        
        self.logger.info("StreamlinePipeline 初始化完成")
    
    def process_batch_streamline(
        self, 
        video_subtitle_pairs: List[Tuple[str, Optional[str]]],
        resume_from_cache: bool = True
    ) -> Dict[str, Any]:
        """
        使用流水线模式批量处理视频
        
        Args:
            video_subtitle_pairs: 包含(video_path, subtitle_path)元组的列表
            resume_from_cache: 是否从缓存恢复（暂不支持）
            
        Returns:
            批量处理结果
        """
        from .pipeline import Task, TaskStatus
        
        start_time = time.time()
        self.logger.info(f"开始流水线批量处理 {len(video_subtitle_pairs)} 个视频")
        
        # 创建任务列表
        tasks = []
        for i, (video_path, subtitle_path) in enumerate(video_subtitle_pairs):
            task_id = f"streamline_task_{i:03d}_{Path(video_path).stem}"
            
            # 设置路径信息
            paths = DubbingPaths(video_path, subtitle_path, self.output_dir)
            cache_file_path = paths.pipeline_cache
            
            # 尝试从缓存恢复Task
            task = None
            if resume_from_cache:
                task = Task.load_from_cache(cache_file_path, video_path, subtitle_path)
                if task:
                    self.logger.info(f"从缓存恢复任务: {task_id}")
                    # 更新任务ID以保持一致性
                    task.task_id = task_id
                else:
                    self.logger.info(f"未找到有效缓存，创建新任务: {task_id}")
            
            # 如果没有从缓存恢复，创建新任务
            if task is None:
                task = Task(
                    task_id=task_id,
                    video_path=video_path,
                    subtitle_path=subtitle_path
                )
            
            # 设置或更新路径信息
            task.paths = {
                "video_path": str(paths.video_path),
                "subtitle_path": str(paths.subtitle_path),
                "processed_subtitle": str(paths.processed_subtitle),
                "vocal_audio": str(paths.vocal_audio),
                "background_audio": str(paths.background_audio),
                "silent_video": str(paths.silent_video),
                "output_dir": str(paths.output_dir),  # 添加缺少的output_dir
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
            
            tasks.append(task)
        
        try:
            # 启动调度器
            with self.task_scheduler:
                # 提交所有任务
                submit_results = self.task_scheduler.submit_batch_tasks(tasks)
                successful_submits = sum(submit_results)
                
                self.logger.info(f"成功提交 {successful_submits}/{len(tasks)} 个任务")
                
                if successful_submits == 0:
                    return {
                        "success": False,
                        "message": "没有任务成功提交",
                        "total_count": len(video_subtitle_pairs),
                        "success_count": 0,
                        "failed_count": len(video_subtitle_pairs),
                        "total_time": time.time() - start_time,
                        "results": [],
                    }
                
                # 等待所有任务完成
                self.logger.info("等待所有任务完成...")
                completed = self.task_scheduler.wait_for_completion(timeout=3600)  # 1小时超时
                
                if not completed:
                    self.logger.warning("部分任务未在超时时间内完成")
                
                # 收集结果
                all_tasks_status = self.task_scheduler.get_all_tasks_status()
                results = self._format_streamline_results(all_tasks_status, video_subtitle_pairs)
                
        except Exception as e:
            self.logger.error(f"流水线处理异常: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"流水线处理失败: {str(e)}",
                "error": str(e),
                "total_count": len(video_subtitle_pairs),
                "success_count": 0,
                "failed_count": len(video_subtitle_pairs),
                "total_time": time.time() - start_time,
                "results": [],
            }
        
        # 统计结果
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
        video_subtitle_pairs: List[Tuple[str, Optional[str]]]
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
                results.append({
                    "success": False,
                    "task_id": task_id,
                    "video_path": video_path,
                    "subtitle_path": subtitle_path,
                    "message": "任务未找到",
                    "error": "任务可能提交失败",
                })
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
                final_video = output_dir / f"{Path(video_path).stem}_dubbed{Path(video_path).suffix}"
                
                result.update({
                    "output_file": str(final_video) if final_video.exists() else "",
                    "output_dir": str(output_dir),
                })
            else:
                # 任务失败，添加错误信息
                result.update({
                    "error": task_status.get("error_message", "未知错误"),
                    "retry_count": task_status.get("retry_count", 0),
                })
            
            results.append(result)
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """获取流水线统计信息"""
        if not hasattr(self, 'task_scheduler'):
            return {"error": "流水线未初始化"}
        
        return self.task_scheduler.get_scheduler_stats()
    
    def cancel_all_tasks(self) -> int:
        """取消所有任务"""
        if not hasattr(self, 'task_scheduler'):
            return 0
        
        return self.task_scheduler.cancel_all_tasks()
    
    def is_running(self) -> bool:
        """检查流水线是否在运行"""
        if not hasattr(self, 'task_scheduler'):
            return False
        
        return self.task_scheduler.is_running()