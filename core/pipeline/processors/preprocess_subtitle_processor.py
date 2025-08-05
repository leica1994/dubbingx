"""
字幕预处理器 - 步骤 0

负责预处理字幕文件，转换格式和清理文本
"""

from pathlib import Path

from core.util import sanitize_filename
from .subtitle_preprocessor import preprocess_subtitle_core
from ..step_processor import StepProcessor
from ..task import ProcessResult, Task, StepProgressDetail


class PreprocessSubtitleProcessor(StepProcessor):
    """字幕预处理步骤处理器"""

    def __init__(self):
        super().__init__(
            step_id=0,
            step_name="preprocess_subtitle",
            timeout=None,  # 移除超时限制，支持长视频处理
            max_retries=3,
        )

    def _execute_process(self, task: Task, step_detail: StepProgressDetail) -> ProcessResult:
        """执行字幕预处理"""
        try:
            # 验证字幕文件路径
            if not task.subtitle_path:
                return ProcessResult(
                    success=False,
                    message="缺少字幕文件路径",
                    error="task.subtitle_path 为空",
                )

            subtitle_path = Path(task.subtitle_path)
            if not subtitle_path.exists():
                return ProcessResult(
                    success=False,
                    message="字幕文件不存在",
                    error=f"文件不存在: {task.subtitle_path}",
                )

            # 获取输出目录，使用统一的文件名清理逻辑
            video_path = Path(task.video_path)
            clean_video_name = sanitize_filename(video_path.stem)
            output_dir = video_path.parent / "outputs" / clean_video_name
            output_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"开始预处理字幕: {task.subtitle_path}")

            # 调用字幕预处理功能
            result = preprocess_subtitle_core(str(task.subtitle_path), str(output_dir))

            if not result.get("success", False):
                return ProcessResult(
                    success=False,
                    message="字幕预处理失败",
                    error=result.get("error", "未知错误"),
                )

            # 保存处理结果到任务路径信息
            if not task.paths:
                task.paths = {}

            # 转换SubtitleEntry对象为字典以支持JSON序列化
            subtitle_entries = result.get("subtitle_entries", [])
            subtitle_entries_dict = [
                entry.to_dict() if hasattr(entry, 'to_dict') else entry 
                for entry in subtitle_entries
            ]

            task.paths.update(
                {
                    "output_dir": str(output_dir),
                    "processed_subtitle": result.get("processed_subtitle_path", ""),
                    "subtitle_entries": subtitle_entries_dict,
                    "total_entries": result.get("total_entries", 0),
                }
            )

            self.logger.info(f"字幕预处理完成: {result.get('total_entries', 0)} 条字幕")

            return ProcessResult(
                success=True,
                message=f"字幕预处理完成，共 {result.get('total_entries', 0)} 条字幕",
                data={
                    "processed_subtitle_path": result.get("processed_subtitle_path"),
                    "total_entries": result.get("total_entries", 0),
                    "format": result.get("format", "srt"),
                },
            )

        except Exception as e:
            self.logger.error(f"字幕预处理异常: {e}", exc_info=True)
            return ProcessResult(
                success=False, message="字幕预处理过程中发生异常", error=str(e)
            )
