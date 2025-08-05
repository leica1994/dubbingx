"""
音视频合并处理器 - 步骤 7

负责将调整后的视频与对齐音频合并，生成最终配音视频
"""

from pathlib import Path

from .media_processor import merge_audio_video_core
from ..step_processor import StepProcessor
from ..task import ProcessResult, Task, StepProgressDetail


class MergeAudioVideoProcessor(StepProcessor):
    """音视频合并步骤处理器"""

    def __init__(self):
        super().__init__(
            step_id=7,
            step_name="merge_audio_video",
            timeout=None,  # 移除超时限制，支持长视频处理
            max_retries=2,
        )

    def _execute_process(self, task: Task, step_detail: StepProgressDetail) -> ProcessResult:
        """执行音视频合并"""
        try:
            # 验证依赖文件
            if not task.paths:
                return ProcessResult(
                    success=False, message="缺少路径信息", error="task.paths 为空"
                )

            speed_adjusted_video_path = task.paths.get("speed_adjusted_video")
            aligned_audio_path = task.paths.get("aligned_audio")

            if (
                not speed_adjusted_video_path
                or not Path(speed_adjusted_video_path).exists()
            ):
                return ProcessResult(
                    success=False,
                    message="速度调整后的视频文件不存在",
                    error=f"文件不存在: {speed_adjusted_video_path}",
                )

            if not aligned_audio_path or not Path(aligned_audio_path).exists():
                return ProcessResult(
                    success=False,
                    message="对齐音频文件不存在",
                    error=f"文件不存在: {aligned_audio_path}",
                )

            # 获取输出目录和最终视频路径
            output_dir = Path(task.paths.get("output_dir", ""))
            video_name = Path(task.video_path).stem
            video_ext = Path(task.video_path).suffix
            final_video_path = output_dir / f"{video_name}_dubbed{video_ext}"

            self.logger.info(
                f"开始音视频合并: {speed_adjusted_video_path} + {aligned_audio_path}"
            )

            # 调用音视频合并功能
            result = merge_audio_video_core(
                speed_adjusted_video_path, aligned_audio_path, str(final_video_path)
            )

            if result and not result.get("success", True):
                return ProcessResult(
                    success=False,
                    message="音视频合并失败",
                    error=result.get("error", "未知错误"),
                )

            # 验证输出文件是否生成
            if not final_video_path.exists():
                return ProcessResult(
                    success=False,
                    message="最终视频文件未生成",
                    error=f"输出文件不存在: {final_video_path}",
                )

            # 更新任务路径信息
            task.paths.update(
                {
                    "final_video": str(final_video_path),
                }
            )

            # 获取文件大小信息
            file_size = final_video_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            self.logger.info(
                f"音视频合并完成: {final_video_path} ({file_size_mb:.1f}MB)"
            )

            return ProcessResult(
                success=True,
                message=f"音视频合并完成，输出文件: {final_video_path.name}",
                data={
                    "final_video_path": str(final_video_path),
                    "file_size": file_size,
                    "file_size_mb": round(file_size_mb, 1),
                    "merge_info": result if result else {},
                },
            )

        except Exception as e:
            self.logger.error(f"音视频合并异常: {e}", exc_info=True)
            return ProcessResult(
                success=False, message="音视频合并过程中发生异常", error=str(e)
            )
