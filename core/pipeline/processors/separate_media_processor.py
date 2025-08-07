"""
媒体分离处理器 - 步骤 1

负责分离视频中的人声、背景音乐和创建无声视频
"""

from pathlib import Path

from core.util import sanitize_filename

from ..step_processor import StepProcessor
from ..task import ProcessResult, StepProgressDetail, Task
from .media_processor import separate_media_core


class SeparateMediaProcessor(StepProcessor):
    """媒体分离步骤处理器"""

    def __init__(self):
        super().__init__(
            step_id=1,
            step_name="separate_media",
            timeout=None,  # 移除超时限制，支持长视频处理
            max_retries=2,
        )

    def _execute_process(
        self, task: Task, step_detail: StepProgressDetail
    ) -> ProcessResult:
        """执行媒体分离"""
        try:
            # 验证视频文件路径
            video_path = Path(task.video_path)
            if not video_path.exists():
                return ProcessResult(
                    success=False,
                    message="视频文件不存在",
                    error=f"文件不存在: {task.video_path}",
                )

            # 获取输出目录
            output_dir = Path(task.paths.get("output_dir", "")) if task.paths else None
            if not output_dir or not output_dir.exists():
                # 应用文件名清理逻辑，与DubbingPaths保持一致
                clean_video_name = sanitize_filename(video_path.stem)
                output_dir = video_path.parent / "outputs" / clean_video_name
                output_dir.mkdir(parents=True, exist_ok=True)

            # 创建媒体分离子目录
            media_separation_dir = output_dir / "media_separation"
            media_separation_dir.mkdir(exist_ok=True)

            # 获取音频分离模式选项
            enable_vocal_separation = getattr(task, "options", {}).get(
                "enable_vocal_separation", False
            )

            # 获取音频质量级别选项
            audio_quality_level = getattr(task, "options", {}).get("audio_quality_level", 0)
            
            # 获取视频质量级别选项
            video_quality_level = getattr(task, "options", {}).get("video_quality_level", 0)

            audio_quality_desc = ["最高质量", "高质量音频", "高效压缩"][audio_quality_level]
            video_quality_desc = ["无损画质", "超高画质", "平衡质量", "高效压缩"][video_quality_level]
            self.logger.info(
                f"开始媒体分离: {task.video_path} (模式: {'完整分离' if enable_vocal_separation else '快速模式'}, 音频质量: {audio_quality_desc}, 视频质量: {video_quality_desc})"
            )

            # 调用媒体分离功能
            result = separate_media_core(
                str(task.video_path),
                str(media_separation_dir),
                enable_vocal_separation,
                audio_quality_level,
                video_quality_level,
            )

            if not result.get("success", False):
                return ProcessResult(
                    success=False,
                    message="媒体分离失败",
                    error=result.get("error", "未知错误"),
                )

            # 更新任务路径信息
            if not task.paths:
                task.paths = {}

            task.paths.update(
                {
                    "media_separation_dir": str(media_separation_dir),
                    "silent_video_path": result.get("silent_video_path", ""),
                    "vocal_audio_path": result.get("vocal_audio_path", ""),
                    "background_audio_path": result.get("background_audio_path", ""),
                }
            )

            self.logger.info("媒体分离完成")

            return ProcessResult(
                success=True,
                message="媒体分离完成",
                data={
                    "silent_video_path": result.get("silent_video_path"),
                    "vocal_audio_path": result.get("vocal_audio_path"),
                    "background_audio_path": result.get("background_audio_path"),
                    "separation_info": result.get("separation_info", {}),
                },
            )

        except Exception as e:
            self.logger.error(f"媒体分离异常: {e}", exc_info=True)
            return ProcessResult(
                success=False, message="媒体分离过程中发生异常", error=str(e)
            )
