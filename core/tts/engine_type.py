"""
TTS引擎类型枚举
"""

from enum import Enum
from typing import Any, Dict


class TTSEngineType(Enum):
    """TTS引擎类型枚举"""
    
    INDEX_TTS = "index_tts"
    COSYVOICE = "cosyvoice"
    
    def __str__(self) -> str:
        """返回字符串值"""
        return self.value
    
    @classmethod
    def from_string(cls, value: str) -> 'TTSEngineType':
        """从字符串创建枚举实例"""
        for engine_type in cls:
            if engine_type.value == value:
                return engine_type
        raise ValueError(f"不支持的TTS引擎类型: {value}")
    
    @property
    def default_api_url(self) -> str:
        """获取默认API URL"""
        default_urls = {
            TTSEngineType.INDEX_TTS: "http://127.0.0.1:7860",
            TTSEngineType.COSYVOICE: "http://127.0.0.1:7860"
        }
        return default_urls.get(self, "")
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        display_names = {
            TTSEngineType.INDEX_TTS: "Index-TTS",
            TTSEngineType.COSYVOICE: "CosyVoice"
        }
        return display_names.get(self, self.value)
    
    @property
    def description(self) -> str:
        """获取引擎描述"""
        descriptions = {
            TTSEngineType.INDEX_TTS: "基于Gradio客户端的Index-TTS声音克隆引擎",
            TTSEngineType.COSYVOICE: "支持多种推理模式的CosyVoice声音克隆引擎"
        }
        return descriptions.get(self, "未知TTS引擎")
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "api_url": self.default_api_url,
            "timeout": 120,
            "ssl_verify": False
        }
    
    @classmethod
    def list_engines(cls) -> list['TTSEngineType']:
        """列出所有支持的引擎类型"""
        return list(cls)
    
    @classmethod
    def get_engine_info(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有引擎的详细信息"""
        return {
            engine_type.value: {
                "enum": engine_type,
                "display_name": engine_type.display_name,
                "description": engine_type.description,
                "default_api_url": engine_type.default_api_url,
                "default_config": engine_type.get_default_config()
            }
            for engine_type in cls.list_engines()
        }