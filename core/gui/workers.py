"""GUI工作线程"""

from typing import Any, Dict, List, Tuple

from PySide6.QtCore import QThread, Signal

from .pipeline import GUIStreamlinePipeline


class StreamlineBatchDubbingWorkerThread(QThread):
    """流水线批量配音处理工作线程 - 使用 process_batch_streamline 方法"""

    # 信号定义
    progress_update = Signal(int, int, str)  # 当前进度, 总数, 当前处理文件名
    batch_finished = Signal(bool, str, dict)  # 是否成功, 消息, 结果详情
    log_message = Signal(str)  # 日志消息

    def __init__(
        self,
        video_subtitle_pairs: List[Tuple[str, str]],
        resume_from_cache: bool = True,
        enable_vocal_separation: bool = False,
        audio_quality_level: int = 0,  # 0=最高质量, 1=高质量, 2=标准质量
        video_quality_level: int = 0,  # 0=无损画质, 1=超高画质, 2=平衡质量, 3=高效压缩
    ):
        super().__init__()
        self.pairs = video_subtitle_pairs
        self.resume_from_cache = resume_from_cache
        self.enable_vocal_separation = enable_vocal_separation
        self.audio_quality_level = audio_quality_level
        self.video_quality_level = video_quality_level
        self.is_cancelled = False

        # 创建流水线管道
        self.pipeline = GUIStreamlinePipeline()

        # 连接流水线日志信号到工作线程信号
        self.pipeline.log_message.connect(self.log_message.emit)

    def cancel(self):
        """取消处理"""
        self.is_cancelled = True

    def run(self):
        """执行并行批量处理"""
        try:
            # 注意：StreamlinePipeline暂不支持进度回调
            # 进度更新需要通过其他方式获取

            # 执行流水线批量处理
            result = self.pipeline.process_batch_streamline(
                self.pairs,
                self.resume_from_cache,
                self.enable_vocal_separation,
                self.audio_quality_level,
                self.video_quality_level,
            )

            # 发送完成信号
            self.batch_finished.emit(result["success"], result["message"], result)

        except Exception as e:
            self.batch_finished.emit(False, f"流水线批量处理失败: {str(e)}", {})
