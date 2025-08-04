"""
Fast-Whisper ASR配置文件
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any


class ASRConfig:
    """Fast-Whisper ASR配置类"""
    
    # 项目配置
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # 模型配置
    DEFAULT_MODEL = "large-v2"
    AVAILABLE_MODELS = [
        "tiny", "base", "small", "medium", 
        "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo"
    ]
    
    # 设备配置 - 强制使用CPU避免cuDNN问题
    FORCE_CPU = True
    DEVICE = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    COMPUTE_TYPE = "float32" if FORCE_CPU else ("float16" if torch.cuda.is_available() else "float32")
    
    # 识别配置
    DEFAULT_LANGUAGE = "en"
    SAMPLE_RATE = 16000
    USE_VAD = True
    BEAM_SIZE = 3  # 降低beam_size提升速度
    BEST_OF = 3    # 降低best_of提升速度
    PATIENCE = 0.5 # 降低patience提升速度
    
    # 下载配置
    USE_MODELSCOPE = True
    FORCE_MODELSCOPE_ONLY = True
    MODELSCOPE_MODELS = {
        "tiny": "pengzhendong/faster-whisper-tiny",
        "base": "pengzhendong/faster-whisper-base", 
        "small": "pengzhendong/faster-whisper-small",
        "medium": "pengzhendong/faster-whisper-medium",
        "large": "pengzhendong/faster-whisper-large",
        "large-v1": "pengzhendong/faster-whisper-large-v1",
        "large-v2": "pengzhendong/faster-whisper-large-v2",
        "large-v3": "pengzhendong/faster-whisper-large-v3",
        "large-v3-turbo": "pengzhendong/faster-whisper-large-v3-turbo"
    }
    
    @classmethod
    def initialize(cls):
        """初始化配置"""
        # 强制CPU模式
        if cls.FORCE_CPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["CT2_USE_CUDA"] = "0"
        
        # ModelScope配置
        if cls.USE_MODELSCOPE:
            os.environ["MODELSCOPE_CACHE"] = str(cls.MODELS_DIR / "modelscope")
        
        # 强制离线模式
        if cls.FORCE_MODELSCOPE_ONLY:
            os.environ.update({
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_EVALUATE_OFFLINE": "1",
                "HF_TOKENIZERS_PARALLELISM": "false"
            })
        
        # 确保目录存在
        cls.MODELS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate_model(cls, model_name: str) -> str:
        """验证模型名称"""
        return model_name if model_name in cls.AVAILABLE_MODELS else cls.DEFAULT_MODEL
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """获取模型配置"""
        config = {
            "device": cls.DEVICE,
            "compute_type": cls.COMPUTE_TYPE,
            "beam_size": cls.BEAM_SIZE,
            "best_of": cls.BEST_OF,
            "patience": cls.PATIENCE,
        }
        
        # 优化小模型参数
        if model_name in ["tiny", "base"]:
            config.update({"beam_size": 3, "best_of": 3})
        elif model_name in ["large-v3-turbo"]:
            config.update({"beam_size": 1, "best_of": 1})
        
        return config
    
    @classmethod
    def get_modelscope_model_id(cls, model_name: str) -> str:
        """获取ModelScope模型ID"""
        return cls.MODELSCOPE_MODELS.get(model_name, f"pengzhendong/faster-whisper-{model_name}")
    
    @classmethod
    def get_device_info(cls) -> str:
        """获取设备信息"""
        return "CPU" if cls.FORCE_CPU else f"CUDA: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"