"""
TTS生成处理器 - 步骤 3

负责基于参考音频生成TTS语音
"""

from pathlib import Path

from core.tts_processor import generate_tts_from_reference
from ..step_processor import StepProcessor
from ..task import ProcessResult, Task, StepProgressDetail, StepStatus


class GenerateTTSProcessor(StepProcessor):
    """TTS生成步骤处理器"""

    def __init__(self):
        super().__init__(
            step_id=3,
            step_name="generate_tts",
            timeout=900.0,  # 15分钟超时
            max_retries=3,
        )

    def _execute_process(self, task: Task, step_detail: StepProgressDetail) -> ProcessResult:
        """执行TTS生成"""
        try:
            # 验证依赖文件
            if not task.paths:
                return ProcessResult(
                    success=False, 
                    message="缺少路径信息", 
                    error="task.paths 为空",
                    step_detail=step_detail
                )

            reference_results = task.paths.get("reference_results")
            output_dir = task.paths.get("output_dir") or task.paths.get("tts_output_dir")

            if not reference_results or not Path(reference_results).exists():
                return ProcessResult(
                    success=False,
                    message="参考音频结果文件不存在",
                    error=f"文件不存在: {reference_results}",
                    step_detail=step_detail
                )

            if not output_dir:
                return ProcessResult(
                    success=False,
                    message="输出目录未指定",
                    error=f"output_dir 为空，可用路径: {list(task.paths.keys())}",
                    step_detail=step_detail
                )

            # 创建TTS输出目录
            tts_output_dir = Path(output_dir) / "tts_output"
            tts_output_dir.mkdir(exist_ok=True)

            self.logger.info(f"开始TTS生成: {reference_results}")

            # 使用TTSProcessor批量生成TTS
            result = self._generate_tts_batch(
                task, step_detail, reference_results, str(tts_output_dir)
            )

            if result.get("success", False):
                # 更新任务路径信息
                task.paths.update({
                    "tts_output_dir": str(tts_output_dir),
                    "tts_results": result.get("results_file", ""),
                })

                successful = result.get("successful_segments", 0)
                failed = result.get("failed_segments", 0)
                total = result.get("total_segments", 0)

                self.logger.info(f"TTS生成完成: {successful}/{total} 成功, {failed} 失败")

                # 只有全部成功才返回success=True
                is_fully_successful = (successful == total and failed == 0)
                
                return ProcessResult(
                    success=is_fully_successful,
                    message=f"TTS生成完成: {successful}/{total} 成功, {failed} 失败",
                    data={
                        "total_segments": total,
                        "successful_segments": successful,
                        "failed_segments": failed,
                        "results_file": result.get("results_file", ""),
                        "output_dir": str(tts_output_dir),
                    },
                    step_detail=step_detail,
                    partial_success=(successful > 0 and failed > 0)  # 标记为部分成功
                )
            else:
                return ProcessResult(
                    success=False,
                    message="TTS生成失败",
                    error=result.get("error", "未知错误"),
                    step_detail=step_detail
                )

        except Exception as e:
            self.logger.error(f"TTS生成异常: {e}", exc_info=True)
            step_detail.status = StepStatus.FAILED
            return ProcessResult(
                success=False, 
                message="TTS生成过程中发生异常", 
                error=str(e),
                step_detail=step_detail
            )

    def _generate_tts_batch(self, task: Task, step_detail: StepProgressDetail, 
                          reference_results_path: str, output_dir: str) -> dict:
        """
        使用TTSProcessor批量生成TTS
        
        Args:
            task: 任务对象
            step_detail: 步骤详细信息
            reference_results_path: 参考音频结果文件路径
            output_dir: 输出目录
            
        Returns:
            生成结果字典
        """
        try:
            self.logger.info(f"开始批量生成TTS: {reference_results_path}")
            
            # 直接调用TTSProcessor的批量处理方法
            result = generate_tts_from_reference(
                reference_results_path=reference_results_path,
                output_dir=output_dir
            )
            
            if result.get("success", False):
                # 获取生成统计信息
                total_segments = result.get("total_segments", 0)
                successful_segments = result.get("successful_segments", 0)
                failed_segments = result.get("failed_segments", 0)
                
                # 更新步骤详情
                step_detail.total_items = total_segments
                step_detail.current_item = successful_segments
                step_detail.update_progress(successful_segments, total_segments)
                
                if successful_segments == total_segments:
                    # 全部成功
                    step_detail.status = StepStatus.COMPLETED
                elif successful_segments > 0:
                    # 部分成功 
                    step_detail.status = StepStatus.PROCESSING
                else:
                    # 全部失败
                    step_detail.status = StepStatus.FAILED
                
                self.logger.info(f"TTS生成完成: {successful_segments}/{total_segments} 成功, {failed_segments} 失败")
                
                return {
                    "success": successful_segments > 0,  # 有成功的就算成功
                    "total_segments": total_segments,
                    "successful_segments": successful_segments,
                    "failed_segments": failed_segments,
                    "results_file": result.get("results_file", ""),
                    "output_dir": output_dir,
                }
            else:
                # 生成失败
                error_msg = result.get("error", "TTS生成失败")
                self.logger.error(f"TTS生成失败: {error_msg}")
                
                step_detail.status = StepStatus.FAILED
                
                return {
                    "success": False,
                    "error": error_msg,
                    "total_segments": 0,
                    "successful_segments": 0,
                    "failed_segments": 0,
                }
                
        except Exception as e:
            error_msg = f"TTS生成异常: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            step_detail.status = StepStatus.FAILED
            
            return {
                "success": False,
                "error": error_msg,
                "total_segments": 0,
                "successful_segments": 0,
                "failed_segments": 0,
            }

    def _resume_from_interruption(self, task: Task, step_detail: StepProgressDetail) -> ProcessResult:
        """
        从中断处恢复TTS生成
        
        由于TTSProcessor是批量处理的，我们直接重新执行
        它内部会处理已经存在的文件
        """
        self.logger.info("TTS生成从中断处恢复，重新执行批量生成")
        
        # 重新执行，TTSProcessor会自动跳过已存在的文件
        return self._execute_process(task, step_detail)