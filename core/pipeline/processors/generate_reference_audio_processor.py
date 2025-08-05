"""
生成参考音频处理器 - 步骤 2

负责从人声音频和字幕中生成参考音频片段
"""

from pathlib import Path

from ..step_processor import StepProcessor
from ..task import ProcessResult, ResourceType, Task
from ...media_processor import generate_reference_audio


class GenerateReferenceAudioProcessor(StepProcessor):
    """生成参考音频步骤处理器"""
    
    def __init__(self):
        super().__init__(
            step_id=2,
            step_name="generate_reference_audio",
            resource_type=ResourceType.CPU_INTENSIVE,
            timeout=300.0,  # 5分钟超时
            max_retries=2
        )
    
    def _execute_process(self, task: Task) -> ProcessResult:
        """执行生成参考音频"""
        try:
            # 验证依赖文件
            if not task.paths:
                return ProcessResult(
                    success=False,
                    message="缺少路径信息",
                    error="task.paths 为空"
                )
            
            vocal_audio_path = task.paths.get("vocal_audio_path")
            processed_subtitle_path = task.paths.get("processed_subtitle")
            
            if not vocal_audio_path or not Path(vocal_audio_path).exists():
                return ProcessResult(
                    success=False,
                    message="人声音频文件不存在",
                    error=f"文件不存在: {vocal_audio_path}"
                )
            
            if not processed_subtitle_path or not Path(processed_subtitle_path).exists():
                return ProcessResult(
                    success=False,
                    message="处理后的字幕文件不存在",
                    error=f"文件不存在: {processed_subtitle_path}"
                )
            
            # 获取输出目录
            output_dir = Path(task.paths.get("output_dir", ""))
            reference_audio_dir = output_dir / "reference_audio"
            reference_audio_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"开始生成参考音频: {vocal_audio_path}")
            
            # 调用生成参考音频功能
            result = generate_reference_audio(
                vocal_audio_path,
                processed_subtitle_path,
                str(reference_audio_dir)
            )
            
            if not result.get("success", False):
                return ProcessResult(
                    success=False,
                    message="生成参考音频失败",
                    error=result.get("error", "未知错误")
                )
            
            # 更新任务路径信息
            task.paths.update({
                "reference_audio_dir": str(reference_audio_dir),
                "reference_results": result.get("results_file", ""),
            })
            
            self.logger.info(
                f"参考音频生成完成: {result.get('total_segments', 0)} 个片段"
            )
            
            return ProcessResult(
                success=True,
                message=f"参考音频生成完成，共 {result.get('total_segments', 0)} 个片段",
                data={
                    "results_file": result.get("results_file"),
                    "total_segments": result.get("total_segments", 0),
                    "output_dir": str(reference_audio_dir),
                    "generation_info": result.get("generation_info", {}),
                }
            )
            
        except Exception as e:
            self.logger.error(f"生成参考音频异常: {e}", exc_info=True)
            return ProcessResult(
                success=False,
                message="生成参考音频过程中发生异常",
                error=str(e)
            )