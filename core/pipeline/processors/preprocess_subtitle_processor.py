"""
字幕预处理器 - 步骤 0

负责预处理字幕文件，转换格式和清理文本
"""

from pathlib import Path
from typing import Dict, Any

from ..step_processor import StepProcessor
from ..task import ProcessResult, ResourceType, Task
from ...subtitle_preprocessor import preprocess_subtitle


class PreprocessSubtitleProcessor(StepProcessor):
    """字幕预处理步骤处理器"""
    
    def __init__(self):
        super().__init__(
            step_id=0,
            step_name="preprocess_subtitle",
            resource_type=ResourceType.CPU_INTENSIVE,
            timeout=300.0,
            max_retries=3
        )
    
    def _execute_process(self, task: Task) -> ProcessResult:
        """执行字幕预处理"""
        try:
            # 验证字幕文件路径
            if not task.subtitle_path:
                return ProcessResult(
                    success=False,
                    message="缺少字幕文件路径",
                    error="task.subtitle_path 为空"
                )
            
            subtitle_path = Path(task.subtitle_path)
            if not subtitle_path.exists():
                return ProcessResult(
                    success=False,
                    message="字幕文件不存在",
                    error=f"文件不存在: {task.subtitle_path}"
                )
            
            # 获取输出目录
            video_path = Path(task.video_path)
            output_dir = video_path.parent / "outputs" / video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"开始预处理字幕: {task.subtitle_path}")
            
            # 调用字幕预处理功能
            result = preprocess_subtitle(str(task.subtitle_path), str(output_dir))
            
            if not result.get("success", False):
                return ProcessResult(
                    success=False,
                    message="字幕预处理失败",
                    error=result.get("error", "未知错误")
                )
            
            # 保存处理结果到任务路径信息
            if not task.paths:
                task.paths = {}
            
            task.paths.update({
                "output_dir": str(output_dir),
                "processed_subtitle": result.get("processed_subtitle_path", ""),
                "subtitle_entries": result.get("subtitle_entries", []),
                "total_entries": result.get("total_entries", 0),
            })
            
            self.logger.info(
                f"字幕预处理完成: {result.get('total_entries', 0)} 条字幕"
            )
            
            return ProcessResult(
                success=True,
                message=f"字幕预处理完成，共 {result.get('total_entries', 0)} 条字幕",
                data={
                    "processed_subtitle_path": result.get("processed_subtitle_path"),
                    "total_entries": result.get("total_entries", 0),
                    "format": result.get("format", "srt"),
                }
            )
            
        except Exception as e:
            self.logger.error(f"字幕预处理异常: {e}", exc_info=True)
            return ProcessResult(
                success=False,
                message="字幕预处理过程中发生异常",
                error=str(e)
            )