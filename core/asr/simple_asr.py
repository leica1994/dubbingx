"""
Fast-Whisper ASR封装
专注于音频到文字的转换，简化版本
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
import gc
from faster_whisper import WhisperModel
from modelscope.hub.snapshot_download import snapshot_download

from .config import ASRConfig


@dataclass
class ASRSegment:
    """ASR片段数据结构"""
    text: str
    start_time: float
    end_time: float
    
    def __post_init__(self):
        # 清理文本
        if self.text:
            self.text = self._clean_text(self.text)
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除开头和结尾的空白
        text = text.strip()
        # 移除重复的标点符号
        text = re.sub(r'([。！？,.!?])\1+', r'\1', text)
        
        return text


class ASRResult:
    """ASR结果类"""
    
    def __init__(self, segments: List[ASRSegment], language: str = "zh"):
        self.segments = segments
        self.language = language
    
    @property
    def text(self) -> str:
        """获取完整文本"""
        return ' '.join([segment.text for segment in self.segments if segment.text.strip()])
    
    @property
    def word_count(self) -> int:
        """获取词数"""
        return len(self.text.split()) if self.text else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'text': self.text,
            'language': self.language,
            'segments': [
                {
                    'text': seg.text,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time
                }
                for seg in self.segments
            ],
            'word_count': self.word_count
        }


class VoiceRecognizer:
    """Fast-Whisper语音识别器（单例模式）"""
    
    _instances = {}
    
    def __new__(cls, model_name: str = None, language: str = None, device: str = None):
        # 创建唯一标识符
        key = (model_name or ASRConfig.DEFAULT_MODEL, language or ASRConfig.DEFAULT_LANGUAGE)
        
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        
        return cls._instances[key]
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 language: Optional[str] = None,
                 device: Optional[str] = None):
        """初始化ASR实例（单例模式，重复调用不会重复初始化）"""
        # 避免重复初始化
        if hasattr(self, 'model'):
            return
            
        # 初始化配置
        ASRConfig.initialize()
        
        # 配置参数
        self.model_name = ASRConfig.validate_model(model_name or ASRConfig.DEFAULT_MODEL)
        self.language = language or ASRConfig.DEFAULT_LANGUAGE
        self.device = device or ASRConfig.DEVICE
        
        # 模型实例
        self.model = None
        
        # 设置日志级别为WARNING减少输出
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        
        # 初始化时立即加载模型
        self._load_model()
    
    def _try_modelscope_download(self, model_name: str) -> Optional[str]:
        """尝试使用ModelScope下载模型"""
        if not ASRConfig.USE_MODELSCOPE:
            return None
        
        model_id = ASRConfig.get_modelscope_model_id(model_name)
        model_dir = ASRConfig.MODELS_DIR / "modelscope" / model_id.replace("/", "--")
        
        # 如果已经下载过，直接返回路径
        if model_dir.exists() and any(model_dir.iterdir()):
            return str(model_dir)
        
        # 下载模型
        snapshot_download(
            model_id=model_id,
            cache_dir=str(ASRConfig.MODELS_DIR / "modelscope"),
            local_dir=str(model_dir)
        )
        
        return str(model_dir)
    
    def _load_model(self):
        """延迟加载模型"""
        if self.model is not None:
            return
        
        # 优先尝试ModelScope下载
        ms_model_path = self._try_modelscope_download(self.model_name)
        
        if ms_model_path and ASRConfig.FORCE_MODELSCOPE_ONLY:
            # 强制只使用ModelScope模型
            model_config = ASRConfig.get_model_config(self.model_name)
            self.model = WhisperModel(
                model_size_or_path=ms_model_path,
                device=model_config["device"],
                compute_type=model_config["compute_type"]
            )
            return
        
        # 如果强制使用ModelScope失败，或者不是强制模式，回退到标准加载
        if not ASRConfig.FORCE_MODELSCOPE_ONLY:
            model_config = ASRConfig.get_model_config(self.model_name)
            self.model = WhisperModel(
                model_size_or_path=self.model_name,
                device=model_config["device"],
                compute_type=model_config["compute_type"],
                download_root=str(ASRConfig.MODELS_DIR)
            )
        else:
            raise Exception("强制ModelScope模式下无法加载模型")
    
    def transcribe_to_text(self, audio_path: str, clean_text: bool = True) -> str:
        """将音频文件转录为纯文本"""
        # 验证文件存在
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # 获取模型配置
        model_config = ASRConfig.get_model_config(self.model_name)
        
        # 进行转录
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=model_config["beam_size"],
            best_of=model_config["best_of"],
            patience=model_config["patience"],
            temperature=0.0,
            vad_filter=ASRConfig.USE_VAD,
            without_timestamps=False
        )
        
        # 转换为内部格式
        asr_segments = []
        for segment in segments:
            asr_segments.append(ASRSegment(
                text=segment.text,
                start_time=segment.start,
                end_time=segment.end
            ))
        
        # 创建结果对象
        result = ASRResult(asr_segments, info.language)
        
        # 清理文本
        full_text = self._clean_text(result.text) if clean_text else result.text
        
        # 清理GPU内存
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        
        return full_text
    
    def transcribe_with_details(self, audio_path: str) -> Dict[str, Any]:
        """转录音频并返回详细信息"""
        # 验证文件存在
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 获取模型配置
        model_config = ASRConfig.get_model_config(self.model_name)
        
        # 进行转录
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=model_config["beam_size"],
            best_of=model_config["best_of"],
            patience=model_config["patience"],
            temperature=0.0,
            vad_filter=ASRConfig.USE_VAD,
            without_timestamps=False
        )
        
        # 转换为内部格式
        asr_segments = []
        for segment in segments:
            asr_segments.append(ASRSegment(
                text=segment.text,
                start_time=segment.start,
                end_time=segment.end
            ))
        
        # 创建结果对象
        result = ASRResult(asr_segments, info.language)
        
        # 清理GPU内存
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        
        return result.to_dict()
    
    def _clean_text(self, text: str) -> str:
        """清理识别出的文本"""
        if not text:
            return ""
        
        # 移除多余的空白字符和换行符
        text = re.sub(r'\s+', ' ', text).replace('\n', ' ')
        # 移除开头和结尾的空白
        text = text.strip()
        # 移除重复的标点符号
        text = re.sub(r'([。！？,.!?])\1+', r'\1', text)
        
        return text
    
    def get_supported_languages(self) -> list:
        """获取支持的语言列表"""
        return [
            'zh', 'en', 'fr', 'de', 'es', 'it', 'ja', 'ko', 
            'nl', 'uk', 'pt', 'ru', 'ar', 'hi', 'th', 'vi'
        ]
    
    def cleanup(self):
        """清理资源（单例模式，实际不清理以保持性能）"""
        pass
    
    @classmethod
    def cleanup_all(cls):
        """清理所有实例资源"""
        for instance in cls._instances.values():
            if hasattr(instance, 'model') and instance.model is not None:
                del instance.model
                instance.model = None
        
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        
        cls._instances.clear()


def _get_recognizer(model_name: str = None, language: str = None) -> VoiceRecognizer:
    """获取单例识别器实例"""
    # 直接使用VoiceRecognizer的单例模式
    return VoiceRecognizer(model_name=model_name, language=language)

def recognize_audio(audio_path: str, model_name: str = None, language: str = None, clean_text: bool = True) -> str:
    """便捷函数：直接识别音频文件获取文本"""
    recognizer = _get_recognizer(model_name, language)
    return recognizer.transcribe_to_text(audio_path, clean_text=clean_text)

def audio_to_prompt(audio_path: str) -> str:
    """便捷函数：将音频转换为prompt text"""
    return recognize_audio(audio_path, clean_text=True)