"""
TTS生成处理器 - 步骤 3

负责基于参考音频生成TTS语音
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.tts_processor import generate_tts_from_reference

from ..step_processor import StepProcessor
from ..task import ProcessResult, StepProgressDetail, StepStatus, Task


def retry_failed_segments(
    reference_results_path: str,
    failed_indices: List[int],
    output_dir: str
) -> Dict[str, Any]:
    """
    精确重试失败的TTS片段
    
    Args:
        reference_results_path: 参考音频结果文件路径
        failed_indices: 失败片段的索引列表
        output_dir: 输出目录
        
    Returns:
        包含重试结果的字典
    """
    import logging
    from core.tts_processor import TTSProcessor
    
    logger = logging.getLogger(__name__)
    
    if not failed_indices:
        return {
            "success": True,
            "retry_successful_segments": 0,
            "failed_segment_indices": [],
            "message": "没有需要重试的片段"
        }
    
    try:
        # 加载参考音频结果
        with open(reference_results_path, "r", encoding="utf-8") as f:
            reference_data = json.load(f)
        
        reference_segments = reference_data.get("reference_audio_segments", [])
        if not reference_segments:
            return {
                "success": False,
                "error": "参考音频片段为空",
                "retry_successful_segments": 0,
                "failed_segment_indices": failed_indices
            }
        
        # 初始化TTS处理器
        tts_processor = TTSProcessor()
        
        # 筛选出需要重试的片段
        segments_to_retry = [
            seg for seg in reference_segments 
            if seg.get("index") in failed_indices
        ]
        
        logger.info(f"准备重试 {len(segments_to_retry)} 个失败片段: {failed_indices}")
        
        # 重试处理
        successful_count = 0
        still_failed_indices = []
        output_dir = Path(output_dir)
        
        for segment in segments_to_retry:
            try:
                # 生成单个TTS片段
                tts_result = tts_processor._generate_single_tts(segment, output_dir)
                
                if tts_result:
                    successful_count += 1
                    logger.debug(f"重试成功: 片段 {segment['index']}")
                else:
                    still_failed_indices.append(segment['index'])
                    logger.warning(f"重试失败: 片段 {segment['index']}")
                    
            except Exception as e:
                still_failed_indices.append(segment['index'])
                logger.error(f"重试片段 {segment['index']} 时出错: {str(e)}")
        
        # 构建重试结果
        is_all_successful = len(still_failed_indices) == 0
        
        logger.info(
            f"重试完成: {successful_count}/{len(segments_to_retry)} 成功, "
            f"剩余失败: {still_failed_indices}"
        )
        
        return {
            "success": is_all_successful,
            "retry_successful_segments": successful_count,
            "failed_segment_indices": still_failed_indices,
            "message": f"重试 {successful_count}/{len(segments_to_retry)} 个片段成功"
        }
        
    except Exception as e:
        logger.error(f"重试失败片段时发生异常: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "retry_successful_segments": 0,
            "failed_segment_indices": failed_indices
        }


class GenerateTTSProcessor(StepProcessor):
    """TTS生成步骤处理器"""

    def __init__(self):
        super().__init__(
            step_id=3,
            step_name="generate_tts",
            timeout=None,  # 移除超时限制，支持长视频处理
            max_retries=3,
        )

    def _execute_process(
        self, task: Task, step_detail: StepProgressDetail
    ) -> ProcessResult:
        """执行TTS生成"""
        try:
            # 验证依赖文件
            if not task.paths:
                return ProcessResult(
                    success=False,
                    message="缺少路径信息",
                    error="task.paths 为空",
                    step_detail=step_detail,
                )

            reference_results = task.paths.get("reference_results")
            output_dir = task.paths.get("output_dir") or task.paths.get(
                "tts_output_dir"
            )

            if not reference_results or not Path(reference_results).exists():
                return ProcessResult(
                    success=False,
                    message="参考音频结果文件不存在",
                    error=f"文件不存在: {reference_results}",
                    step_detail=step_detail,
                )

            if not output_dir:
                return ProcessResult(
                    success=False,
                    message="输出目录未指定",
                    error=f"output_dir 为空，可用路径: {list(task.paths.keys())}",
                    step_detail=step_detail,
                )

            # 创建TTS输出目录
            tts_output_dir = Path(output_dir) / "tts_output"
            tts_output_dir.mkdir(exist_ok=True)

            self.logger.info(f"开始TTS生成: {reference_results}")

            # 使用TTSProcessor批量生成TTS
            result = self._generate_tts_batch(
                task, step_detail, reference_results, str(tts_output_dir)
            )

            if result.get("successful_segments", 0) > 0:
                # 有成功片段时更新任务路径信息
                task.paths.update(
                    {
                        "tts_output_dir": str(tts_output_dir),
                        "tts_results": result.get("results_file", ""),
                    }
                )

                successful = result.get("successful_segments", 0)
                failed = result.get("failed_segments", 0)
                total = result.get("total_segments", 0)

                self.logger.info(
                    f"TTS生成完成: {successful}/{total} 成功, {failed} 失败"
                )

                # 严格判定：只有全部成功才返回success=True
                is_fully_successful = successful == total and failed == 0

                if is_fully_successful:
                    return ProcessResult(
                        success=True,
                        message=f"TTS生成完成: {successful}/{total} 成功",
                        data={
                            "total_segments": total,
                            "successful_segments": successful,
                            "failed_segments": failed,
                            "results_file": result.get("results_file", ""),
                            "output_dir": str(tts_output_dir),
                        },
                        step_detail=step_detail,
                    )
                else:
                    # 有失败片段，标记为失败
                    return ProcessResult(
                        success=False,
                        message=f"TTS生成不完整: {successful}/{total} 成功, {failed} 失败",
                        error=f"有 {failed} 个片段生成失败",
                        data={
                            "total_segments": total,
                            "successful_segments": successful,
                            "failed_segments": failed,
                            "results_file": result.get("results_file", ""),
                            "output_dir": str(tts_output_dir),
                        },
                        step_detail=step_detail,
                    )
            else:
                return ProcessResult(
                    success=False,
                    message="TTS生成失败",
                    error=result.get("error", "未知错误"),
                    step_detail=step_detail,
                )

        except Exception as e:
            self.logger.error(f"TTS生成异常: {e}", exc_info=True)
            step_detail.status = StepStatus.FAILED
            return ProcessResult(
                success=False,
                message="TTS生成过程中发生异常",
                error=str(e),
                step_detail=step_detail,
            )

    def _generate_tts_batch(
        self,
        task: Task,
        step_detail: StepProgressDetail,
        reference_results_path: str,
        output_dir: str,
    ) -> dict:
        """
        使用TTSProcessor批量生成TTS - 带两阶段重试机制

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

            # 第一阶段：首次尝试，失败的立即重试5次
            result = self._attempt_tts_generation_with_retries(
                reference_results_path, output_dir, max_retries=5, stage="第一阶段"
            )

            if result.get("success", False):
                failed_segments = result.get("failed_segments", 0)
                total_segments = result.get("total_segments", 0)

                # 如果有失败的片段，进行第二阶段重试
                if failed_segments > 0:
                    self.logger.info(f"第二阶段：对失败的 {failed_segments} 个片段进行最终重试")
                    
                    # 第二阶段：对失败片段最终重试5次
                    retry_result = self._attempt_tts_generation_with_retries(
                        reference_results_path, output_dir, max_retries=5, stage="第二阶段"
                    )

                    if retry_result.get("success", False):
                        # 更新最终统计
                        result["successful_segments"] = retry_result.get("successful_segments", 0)
                        result["failed_segments"] = retry_result.get("failed_segments", 0)
                        result["results_file"] = retry_result.get("results_file", "")

                        self.logger.info(
                            f"第二阶段重试完成: {result['successful_segments']}/{total_segments} 成功"
                        )

                # 更新步骤详情
                step_detail.total_items = result.get("total_segments", 0)
                step_detail.current_item = result.get("successful_segments", 0)
                step_detail.update_progress(
                    result.get("successful_segments", 0), 
                    result.get("total_segments", 0)
                )

                # 严格判定：只有全部成功才标记为完成，否则就是失败
                if result.get("successful_segments", 0) == result.get("total_segments", 0):
                    # 全部成功
                    step_detail.status = StepStatus.COMPLETED
                else:
                    # 有任何失败片段就标记为失败，不继续后续步骤
                    step_detail.status = StepStatus.FAILED
                    self.logger.error(
                        f"TTS生成不完整，失败片段: {result.get('failed_segments', 0)}，标记为失败"
                    )

                return {
                    "success": result.get("successful_segments", 0) == result.get("total_segments", 0),  # 只有全部成功才算成功
                    "total_segments": result.get("total_segments", 0),
                    "successful_segments": result.get("successful_segments", 0),
                    "failed_segments": result.get("failed_segments", 0),
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

    def _attempt_tts_generation_with_retries(
        self, reference_results_path: str, output_dir: str, max_retries: int, stage: str = ""
    ) -> dict:
        """
        尝试TTS生成并进行精确片段级重试

        Args:
            reference_results_path: 参考音频结果文件路径
            output_dir: 输出目录
            max_retries: 最大重试次数
            stage: 阶段标识

        Returns:
            生成结果字典
        """
        self.logger.info(f"{stage}开始TTS生成，最大重试次数: {max_retries}")

        # 首次尝试：处理所有片段
        result = generate_tts_from_reference(
            reference_results_path=reference_results_path, output_dir=output_dir
        )

        if not result.get("success", False):
            return result

        # 检查是否有失败片段
        failed_indices = result.get("failed_segment_indices", [])
        if not failed_indices:
            self.logger.info(f"{stage}首次尝试完全成功")
            return result

        original_total = result.get("total_segments", 0)
        cumulative_successful = result.get("successful_segments", 0)

        # 重试机制：只重试失败的片段
        for retry_count in range(1, max_retries + 1):
            if not failed_indices:
                break  # 没有失败片段，无需重试

            self.logger.warning(
                f"{stage}第 {retry_count} 次重试，目标失败片段: {failed_indices}"
            )

            try:
                # 精确重试：只处理失败的片段
                retry_result = retry_failed_segments(
                    reference_results_path=reference_results_path, 
                    failed_indices=failed_indices,
                    output_dir=output_dir
                )

                if retry_result.get("success", False):
                    # 重试成功，更新统计
                    retry_successful = retry_result.get("retry_successful_segments", 0)
                    cumulative_successful += retry_successful
                    failed_indices = []  # 全部成功
                    
                    self.logger.info(f"{stage}第 {retry_count} 次重试完全成功，重试成功 {retry_successful} 个片段")
                    break
                else:
                    # 部分重试成功
                    retry_successful = retry_result.get("retry_successful_segments", 0)
                    cumulative_successful += retry_successful
                    failed_indices = retry_result.get("failed_segment_indices", [])
                    
                    if retry_successful > 0:
                        self.logger.info(
                            f"{stage}第 {retry_count} 次重试部分成功，成功 {retry_successful} 个，剩余失败: {failed_indices}"
                        )
                    else:
                        self.logger.warning(f"{stage}第 {retry_count} 次重试完全失败")

            except Exception as e:
                self.logger.warning(f"{stage}第 {retry_count} 次重试异常: {str(e)}")

        # 构建最终结果
        final_failed_count = len(failed_indices)
        
        if final_failed_count > 0:
            self.logger.error(
                f"{stage}经过 {max_retries} 次重试后仍有 {final_failed_count} 个片段失败: {failed_indices}"
            )

        final_result = {
            "success": final_failed_count == 0,
            "total_segments": original_total,
            "successful_segments": cumulative_successful,
            "failed_segments": final_failed_count,
            "failed_segment_indices": failed_indices,
            "results_file": result.get("results_file", ""),
        }

        return final_result

    def _resume_from_interruption(
        self, task: Task, step_detail: StepProgressDetail
    ) -> ProcessResult:
        """
        从中断处恢复TTS生成

        由于TTSProcessor是批量处理的，我们直接重新执行
        它内部会处理已经存在的文件
        """
        self.logger.info("TTS生成从中断处恢复，重新执行批量生成")

        # 重新执行，TTSProcessor会自动跳过已存在的文件
        return self._execute_process(task, step_detail)
