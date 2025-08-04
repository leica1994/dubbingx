"""
ASR (Automatic Speech Recognition) 模块

基于WhisperX的语音识别模块，专注于音频到文字的转换
"""

from .simple_asr import VoiceRecognizer, recognize_audio, audio_to_prompt
from .config import ASRConfig

__all__ = ['VoiceRecognizer', 'recognize_audio', 'audio_to_prompt', 'ASRConfig']

# 版本信息
__version__ = '1.0.0'

# 便捷函数（保持向后兼容）
def create_recognizer(model_name: str = None, language: str = None, device: str = None) -> VoiceRecognizer:
    """
    创建语音识别器实例的便捷函数
    
    Args:
        model_name: 模型名称，默认large-v2
        language: 语言代码，None为自动检测
        device: 设备，None为自动选择
        
    Returns:
        VoiceRecognizer实例
    """
    return VoiceRecognizer(model_name=model_name, language=language, device=device)

# 向后兼容的别名
SimpleASR = VoiceRecognizer
create_asr = create_recognizer
transcribe_audio = recognize_audio