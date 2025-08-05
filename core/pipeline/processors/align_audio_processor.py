"""
音频对齐处理器 - 步骤 4

负责将TTS生成的音频与字幕时间轴对齐
"""

import json
from datetime import datetime
from pathlib import Path

from .audio_align_processor import align_audio_with_subtitles_core
from ..step_processor import StepProcessor
from ..task import ProcessResult, Task, StepProgressDetail


class AlignAudioProcessor(StepProcessor):
    """音频对齐步骤处理器"""

    def __init__(self):
        super().__init__(
            step_id=4,
            step_name="align_audio",
            timeout=300.0,  # 5分钟超时
            max_retries=2,
        )

    def _execute_process(self, task: Task, step_detail: StepProgressDetail) -> ProcessResult:
        """执行音频对齐"""
        try:
            # 验证依赖文件
            if not task.paths:
                return ProcessResult(
                    success=False, message="缺少路径信息", error="task.paths 为空"
                )

            tts_results_path = task.paths.get("tts_results")
            processed_subtitle_path = task.paths.get("processed_subtitle")

            if not tts_results_path or not Path(tts_results_path).exists():
                return ProcessResult(
                    success=False,
                    message="TTS结果文件不存在",
                    error=f"文件不存在: {tts_results_path}",
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
            aligned_audio_dir = output_dir / "aligned_audio"
            aligned_audio_dir.mkdir(exist_ok=True)

            aligned_audio_path = (
                aligned_audio_dir / "aligned_tts_generation_results.wav"
            )

            self.logger.info(f"开始音频对齐: {tts_results_path}")

            # 调用音频对齐功能
            result = align_audio_with_subtitles_core(
                tts_results_path=tts_results_path,
                srt_path=processed_subtitle_path,
                output_path=str(aligned_audio_path),
            )

            if not result.get("success", False):
                return ProcessResult(
                    success=False,
                    message="音频对齐失败",
                    error=result.get("error", "未知错误"),
                )

            # 保存对齐结果到JSON文件
            aligned_results_path = (
                aligned_audio_dir / "aligned_tts_generation_results.json"
            )
            result_copy = result.copy()
            result_copy["saved_at"] = datetime.now().isoformat()

            with open(aligned_results_path, "w", encoding="utf-8") as f:
                json.dump(result_copy, f, ensure_ascii=False, indent=2)

            # 更新任务路径信息
            task.paths.update(
                {
                    "aligned_audio_dir": str(aligned_audio_dir),
                    "aligned_audio": str(aligned_audio_path),
                    "aligned_results": str(aligned_results_path),
                }
            )

            self.logger.info(
                f"音频对齐完成: 总时长 {result.get('total_duration', 0):.2f}s"
            )

            return ProcessResult(
                success=True,
                message=f"音频对齐完成，总时长 {result.get('total_duration', 0):.2f}s",
                data={
                    "output_file": str(aligned_audio_path),
                    "total_duration": result.get("total_duration", 0),
                    "segments_count": result.get("segments_count", 0),
                    "alignment_info": result.get("alignment_info", {}),
                },
            )

        except Exception as e:
            self.logger.error(f"音频对齐异常: {e}", exc_info=True)
            return ProcessResult(
                success=False, message="音频对齐过程中发生异常", error=str(e)
            )
