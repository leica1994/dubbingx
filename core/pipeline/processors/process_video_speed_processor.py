"""
视频速度调整处理器 - 步骤 6

负责根据音频对齐结果调整视频播放速度
"""

from pathlib import Path

from .audio_align_processor import process_video_speed_adjustment_core
from ..step_processor import StepProcessor
from ..task import ProcessResult, Task, StepProgressDetail


class ProcessVideoSpeedProcessor(StepProcessor):
    """视频速度调整步骤处理器"""

    def __init__(self):
        super().__init__(
            step_id=6,
            step_name="process_video_speed",
            timeout=600.0,  # 10分钟超时
            max_retries=2,
        )

    def _execute_process(self, task: Task, step_detail: StepProgressDetail) -> ProcessResult:
        """执行视频速度调整"""
        try:
            # 验证依赖文件
            if not task.paths:
                return ProcessResult(
                    success=False, message="缺少路径信息", error="task.paths 为空"
                )

            silent_video_path = task.paths.get("silent_video_path")
            processed_subtitle_path = task.paths.get("processed_subtitle")
            aligned_srt_path = task.paths.get("aligned_srt")

            if not silent_video_path or not Path(silent_video_path).exists():
                return ProcessResult(
                    success=False,
                    message="无声视频文件不存在",
                    error=f"文件不存在: {silent_video_path}",
                )

            if (
                not processed_subtitle_path
                or not Path(processed_subtitle_path).exists()
            ):
                return ProcessResult(
                    success=False,
                    message="处理后的字幕文件不存在",
                    error=f"文件不存在: {processed_subtitle_path}",
                )

            if not aligned_srt_path or not Path(aligned_srt_path).exists():
                return ProcessResult(
                    success=False,
                    message="对齐字幕文件不存在",
                    error=f"文件不存在: {aligned_srt_path}",
                )

            # 获取输出目录
            output_dir = Path(task.paths.get("output_dir", ""))
            adjusted_video_dir = output_dir / "adjusted_video"
            adjusted_video_dir.mkdir(exist_ok=True)

            self.logger.info(f"开始视频速度调整: {silent_video_path}")

            # 调用视频速度调整功能
            result = process_video_speed_adjustment_core(
                silent_video_path, processed_subtitle_path, aligned_srt_path
            )

            if result and not result.get("success", True):
                return ProcessResult(
                    success=False,
                    message="视频速度调整失败",
                    error=result.get("error", "未知错误"),
                )

            # 构建输出文件路径（根据函数行为推断）
            video_name = Path(task.video_path).stem
            video_ext = Path(task.video_path).suffix
            speed_adjusted_video_path = (
                adjusted_video_dir
                / f"final_speed_adjusted_{video_name}_silent{video_ext}"
            )

            # 更新任务路径信息
            task.paths.update(
                {
                    "adjusted_video_dir": str(adjusted_video_dir),
                    "speed_adjusted_video": str(speed_adjusted_video_path),
                }
            )

            self.logger.info("视频速度调整完成")

            return ProcessResult(
                success=True,
                message="视频速度调整完成",
                data={
                    "speed_adjusted_video_path": str(speed_adjusted_video_path),
                    "adjustment_info": result if result else {},
                },
            )

        except Exception as e:
            self.logger.error(f"视频速度调整异常: {e}", exc_info=True)
            return ProcessResult(
                success=False, message="视频速度调整过程中发生异常", error=str(e)
            )
