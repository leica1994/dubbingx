"""
生成对齐字幕处理器 - 步骤 5

负责生成与对齐音频匹配的字幕文件
"""

from pathlib import Path

from ...subtitle.subtitle_processor import convert_subtitle, sync_srt_timestamps_to_ass
from ..step_processor import StepProcessor
from ..task import ProcessResult, StepProgressDetail, Task
from .audio_align_processor import generate_aligned_srt_core


class GenerateAlignedSrtProcessor(StepProcessor):
    """生成对齐字幕步骤处理器"""

    def __init__(self):
        super().__init__(
            step_id=5,
            step_name="generate_aligned_srt",
            timeout=None,  # 移除超时限制，支持长视频处理
            max_retries=2,
        )

    def _execute_process(
        self, task: Task, step_detail: StepProgressDetail
    ) -> ProcessResult:
        """执行生成对齐字幕"""
        try:
            # 验证依赖文件
            if not task.paths:
                return ProcessResult(
                    success=False, message="缺少路径信息", error="task.paths 为空"
                )

            aligned_results_path = task.paths.get("aligned_results")
            processed_subtitle_path = task.paths.get("processed_subtitle")

            if not aligned_results_path or not Path(aligned_results_path).exists():
                return ProcessResult(
                    success=False,
                    message="对齐结果文件不存在",
                    error=f"文件不存在: {aligned_results_path}",
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

            # 获取输出目录
            output_dir = Path(task.paths.get("output_dir", ""))
            aligned_audio_dir = Path(task.paths.get("aligned_audio_dir", ""))

            aligned_srt_path = aligned_audio_dir / "aligned_tts_generation_aligned.srt"

            self.logger.info(f"开始生成对齐字幕: {aligned_results_path}")

            # 生成对齐后的SRT字幕
            result = generate_aligned_srt_core(
                aligned_results_path, processed_subtitle_path, str(aligned_srt_path)
            )

            # 检查生成结果
            if not result.get("success", False):
                return ProcessResult(
                    success=False,
                    message="生成对齐字幕失败",
                    error=result.get("error", "未知错误"),
                )

            # 更新任务路径信息
            task.paths.update(
                {
                    "aligned_srt": str(aligned_srt_path),
                }
            )

            # 检查原始字幕格式，如果不是SRT则需要转换为相应格式
            original_subtitle_path = task.subtitle_path
            original_subtitle_ext = Path(original_subtitle_path).suffix.lower()

            additional_files = []

            if original_subtitle_ext != ".srt":
                self.logger.info(
                    f"检测到原始字幕格式为 {original_subtitle_ext}，正在转换..."
                )

                if original_subtitle_ext == ".ass":
                    # 对于ASS格式，使用sync_srt_timestamps_to_ass方法同步时间戳
                    video_name = Path(task.video_path).stem
                    aligned_ass_path = output_dir / f"{video_name}_aligned.ass"

                    sync_success = sync_srt_timestamps_to_ass(
                        original_subtitle_path,
                        str(aligned_srt_path),
                        str(aligned_ass_path),
                    )

                    if sync_success:
                        additional_files.append(str(aligned_ass_path))
                        self.logger.info(f"ASS字幕时间戳同步完成: {aligned_ass_path}")

                        # 更新路径信息
                        task.paths["aligned_ass"] = str(aligned_ass_path)
                    else:
                        self.logger.warning("ASS字幕时间戳同步失败")
                else:
                    # 对于其他格式，使用convert_subtitle转换
                    video_name = Path(task.video_path).stem
                    aligned_subtitle_path = (
                        output_dir / f"{video_name}_aligned{original_subtitle_ext}"
                    )

                    convert_success = convert_subtitle(
                        str(aligned_srt_path), str(aligned_subtitle_path)
                    )

                    if convert_success:
                        additional_files.append(str(aligned_subtitle_path))
                        self.logger.info(f"字幕格式转换完成: {aligned_subtitle_path}")

                        # 更新路径信息
                        task.paths[f"aligned_{original_subtitle_ext[1:]}"] = str(
                            aligned_subtitle_path
                        )
                    else:
                        self.logger.warning("字幕格式转换失败")

            self.logger.info("对齐字幕生成完成")

            return ProcessResult(
                success=True,
                message="对齐字幕生成完成",
                data={
                    "aligned_srt_path": str(aligned_srt_path),
                    "additional_files": additional_files,
                    "original_format": original_subtitle_ext,
                },
            )

        except Exception as e:
            self.logger.error(f"生成对齐字幕异常: {e}", exc_info=True)
            return ProcessResult(
                success=False, message="生成对齐字幕过程中发生异常", error=str(e)
            )
