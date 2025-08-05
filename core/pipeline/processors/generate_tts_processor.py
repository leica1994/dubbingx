"""
TTS生成处理器 - 步骤 3

负责基于参考音频生成TTS语音
"""

from pathlib import Path

from ..step_processor import StepProcessor
from ..task import ProcessResult, ResourceType, Task
from ...tts_processor import generate_tts_from_reference


class GenerateTTSProcessor(StepProcessor):
    """TTS生成步骤处理器"""
    
    def __init__(self):
        super().__init__(
            step_id=3,
            step_name="generate_tts",
            resource_type=ResourceType.GPU_INTENSIVE,
            timeout=900.0,  # 15分钟超时
            max_retries=3
        )
    
    def _execute_process(self, task: Task) -> ProcessResult:
        """执行TTS生成"""
        try:
            # 验证依赖文件
            if not task.paths:
                return ProcessResult(
                    success=False,
                    message="缺少路径信息",
                    error="task.paths 为空"
                )
            
            reference_results_path = task.paths.get("reference_results")
            
            if not reference_results_path or not Path(reference_results_path).exists():
                return ProcessResult(
                    success=False,
                    message="参考音频结果文件不存在",
                    error=f"文件不存在: {reference_results_path}"
                )
            
            # 获取输出目录
            output_dir = Path(task.paths.get("output_dir", ""))
            tts_output_dir = output_dir / "tts_output"
            tts_output_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"开始TTS生成: {reference_results_path}")
            
            # 调用TTS生成功能
            result = generate_tts_from_reference(
                reference_results_path,
                str(tts_output_dir)
            )
            
            if not result.get("success", False):
                return ProcessResult(
                    success=False,
                    message="TTS生成失败",
                    error=result.get("error", "未知错误")
                )
            
            # 更新任务路径信息
            task.paths.update({
                "tts_output_dir": str(tts_output_dir),
                "tts_results": str(tts_output_dir / "tts_generation_results.json"),
            })
            
            self.logger.info(
                f"TTS生成完成: {result.get('total_segments', 0)} 个片段"
            )
            
            return ProcessResult(
                success=True,
                message=f"TTS生成完成，共 {result.get('total_segments', 0)} 个片段",
                data={
                    "tts_audio_segments": result.get("tts_audio_segments", []),
                    "total_segments": result.get("total_segments", 0),
                    "total_requested": result.get("total_requested", 0),
                    "output_dir": str(tts_output_dir),
                    "generation_info": result.get("generation_info", {}),
                }
            )
            
        except Exception as e:
            self.logger.error(f"TTS生成异常: {e}", exc_info=True)
            return ProcessResult(
                success=False,
                message="TTS生成过程中发生异常",
                error=str(e)
            )