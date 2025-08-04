"""
TTS处理器
支持Index-TTS和CosyVoice多引擎的统一TTS处理接口
"""

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .tts import IndexTTSEngine, CosyVoiceTTSEngine, TTSEngineType


class TTSProcessor:
    """多引擎TTS处理器，支持Index-TTS和CosyVoice"""

    def __init__(self, engine_type: TTSEngineType = TTSEngineType.INDEX_TTS, engine_config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        
        # 处理engine_type参数，支持枚举
        if isinstance(engine_type, TTSEngineType):
            self.engine_type = engine_type
        else:
            raise TypeError(f"engine_type必须是TTSEngineType枚举，得到: {type(engine_type)}")
            
        self.engine_config = engine_config or {}
        self.engine = None
        
        # 初始化TTS引擎
        self._initialize_engine()

    def _initialize_engine(self):
        """初始化TTS引擎"""
        try:
            if self.engine_type == TTSEngineType.INDEX_TTS:
                api_url = self.engine_config.get("api_url", self.engine_type.default_api_url)
                self.engine = IndexTTSEngine(api_url)
                self.logger.info(f"{self.engine_type.display_name}引擎初始化成功: {api_url}")
            elif self.engine_type == TTSEngineType.COSYVOICE:
                api_url = self.engine_config.get("api_url", self.engine_type.default_api_url)
                self.engine = CosyVoiceTTSEngine(api_url)
                self.logger.info(f"{self.engine_type.display_name}引擎初始化成功: {api_url}")
            else:
                raise ValueError(f"不支持的TTS引擎类型: {self.engine_type}")
                
        except Exception as e:
            self.logger.error(f"TTS引擎初始化失败: {str(e)}")
            self.engine = None

    def generate_tts_from_reference(
            self, reference_results_path: str, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        根据参考音频结果生成TTS语音

        Args:
            reference_results_path: 参考音频结果JSON文件路径
            output_dir: TTS输出目录，默认为JSON文件所在目录下的tts_output

        Returns:
            包含TTS生成结果的字典
        """
        try:
            # 验证输入文件
            if not os.path.exists(reference_results_path):
                return {
                    "success": False,
                    "error": f"参考音频结果文件不存在: {reference_results_path}",
                }

            # 加载参考音频结果
            reference_data = self._load_reference_results(reference_results_path)
            if not reference_data:
                return {"success": False, "error": "参考音频结果文件加载失败"}

            # 设置输出目录
            if output_dir is None:
                output_dir = Path(reference_results_path).parent.parent / "tts_output"
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 检查缓存
            cache_results = self._check_tts_cache(
                reference_results_path, str(output_dir)
            )
            if cache_results:
                self.logger.info("找到有效的TTS缓存，直接返回结果")
                return cache_results

            self.logger.info(f"开始TTS语音生成: {reference_results_path}")

            # 获取参考音频片段
            reference_segments = reference_data.get("reference_audio_segments", [])
            if not reference_segments:
                return {"success": False, "error": "参考音频片段为空"}

            self.logger.info(f"找到 {len(reference_segments)} 个参考音频片段")

            # 生成TTS音频片段
            tts_segments = []
            successful_segments = 0

            for segment in reference_segments:
                try:
                    tts_result = self._generate_single_tts(segment, output_dir)
                    if tts_result:
                        tts_segments.append(tts_result)
                        successful_segments += 1
                        self.logger.debug(f"TTS生成成功: 片段 {segment['index']}")
                    else:
                        self.logger.warning(f"TTS生成失败: 片段 {segment['index']}")

                except Exception as e:
                    self.logger.error(f"处理片段 {segment['index']} 时出错: {str(e)}")
                    continue

            self.logger.info(
                f"TTS生成完成: {successful_segments}/{len(reference_segments)} 个片段成功"
            )

            # 构建结果
            results = {
                "success": True,
                "tts_audio_segments": tts_segments,
                "output_dir": str(output_dir),
                "total_segments": successful_segments,
                "total_requested": len(reference_segments),
                "reference_file": reference_results_path,
                "generation_info": {
                    "device": "api_client",
                    "model_type": self._get_model_info(),
                    "engine_type": str(self.engine_type),
                    "engine_display_name": self.engine_type.display_name,
                    "api_url": self.engine_config.get("api_url", ""),
                    "engine_available": self.engine is not None,
                    "segments_generated": successful_segments,
                    "segments_failed": len(reference_segments) - successful_segments,
                },
            }

            # 保存TTS结果
            self._save_tts_results(results, output_dir)

            return results

        except Exception as e:
            self.logger.error(f"TTS生成失败: {str(e)}")
            return {"success": False, "error": str(e)}

    def _load_reference_results(self, json_path: str) -> Optional[Dict[str, Any]]:
        """加载参考音频结果"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not data.get("success", False):
                self.logger.error("参考音频结果显示处理失败")
                return None

            return data

        except Exception as e:
            self.logger.error(f"加载参考音频结果失败: {str(e)}")
            return None

    def _generate_single_tts(
            self, reference_segment: Dict[str, Any], output_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """生成单个TTS音频片段"""
        try:
            index = reference_segment["index"]
            text = reference_segment.get("text", "").strip()
            reference_file = reference_segment.get("file_path", "")
            # 生成TTS音频文件名
            tts_path = output_dir / f"tts_{index:04d}.wav"

            # 如果文本为空，生成静音片段
            if not text:
                self.logger.debug(f"片段 {index} 文本为空，生成静音片段")

                # 生成与参考音频同等时长的静音
                duration = reference_segment.get("duration", 0)
                if duration > 0:
                    silence_result = self._generate_silence_audio(
                        str(tts_path), duration
                    )

                    if silence_result and silence_result.get("success"):
                        return {
                            "index": index,
                            "text": "",
                            "reference_file": reference_file,
                            "tts_file": str(tts_path),
                            "duration": duration,
                            "is_empty": True,
                            "tts_generated": True,
                            "silence_generated": True,
                        }

                # 如果生成失败，返回无音频的结果
                return {
                    "index": index,
                    "text": "",
                    "reference_file": reference_file,
                    "tts_file": None,
                    "duration": reference_segment.get("duration", 0),
                    "is_empty": True,
                    "tts_generated": False,
                }

            # 调用TTS生成
            tts_result = self._call_tts_generation(text, str(tts_path), reference_file)

            if tts_result and tts_path.exists():
                # 获取生成的音频信息
                audio_info = self._get_audio_info(str(tts_path))

                # 构建返回结果
                result = {
                    "index": index,
                    "text": text,
                    "reference_file": reference_file,
                    "tts_file": str(tts_path),
                    "duration": audio_info.get("duration", 0),
                    "sample_rate": audio_info.get("sample_rate", 22050),
                    "is_empty": False,
                    "tts_generated": True,
                    "file_size": audio_info.get("file_size", 0),
                }

                # 添加TTS引擎特定的信息
                if isinstance(tts_result, dict):
                    result.update(
                        {
                            "tts_engine": tts_result.get("engine_type", "mock"),
                            "voice_cloned": tts_result.get("voice_cloned", False),
                            "generated_silence": tts_result.get(
                                "generated_silence", False
                            ),
                            "metadata": tts_result.get("metadata", {}),
                        }
                    )

                return result
            else:
                self.logger.warning(f"TTS生成失败: 片段 {index}")
                return None

        except Exception as e:
            self.logger.error(f"生成TTS片段失败: {str(e)}")
            return None

    def _call_tts_generation(
            self, text: str, output_path: str, reference_file: str = None, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """调用TTS模型生成音频"""
        try:
            if not self.engine:
                self.logger.error("TTS引擎未初始化")
                return None
                
            return self.engine.generate_audio(text, output_path, reference_file, **kwargs)
        except Exception as e:
            self.logger.error(f"TTS生成调用失败: {str(e)}")
            return None

    def _generate_silence_audio(
            self, output_path: str, duration: float
    ) -> Dict[str, Any]:
        """生成静音音频"""
        try:
            if not self.engine:
                self.logger.error("TTS引擎未初始化")
                return {"success": False, "error": "TTS引擎未初始化"}
                
            return self.engine.generate_silence_audio(output_path, duration)
        except Exception as e:
            self.logger.error(f"静音音频生成失败: {str(e)}")
            return {"success": False, "error": str(e)}

    def _get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """获取音频文件信息"""
        try:
            if not self.engine:
                return {"duration": 0, "sample_rate": 22050, "channels": 1, "file_size": 0}
                
            return self.engine.get_audio_info(audio_path)
        except Exception as e:
            self.logger.warning(f"获取音频信息失败: {str(e)}")
            return {"duration": 0, "sample_rate": 22050, "channels": 1, "file_size": 0}

    def _get_model_info(self) -> str:
        """获取模型信息"""
        if self.engine:
            return self.engine.get_model_info()
        else:
            return "unknown_tts_model"

    def _check_tts_cache(
            self, reference_path: str, output_dir: str
    ) -> Optional[Dict[str, Any]]:
        """检查TTS缓存"""
        try:
            # 构建缓存文件路径
            cache_filename = "tts_generation_results.json"
            cache_path = Path(output_dir) / cache_filename

            if not cache_path.exists():
                return None

            # 加载缓存
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)

            # 验证缓存有效性
            return cached_data

        except Exception as e:
            self.logger.warning(f"检查TTS缓存失败: {str(e)}")
            return None

    def _save_tts_results(self, results: Dict[str, Any], output_dir: Path):
        """保存TTS结果"""
        try:
            # 添加保存时间戳
            results_copy = results.copy()
            results_copy["saved_at"] = datetime.datetime.now().isoformat()
            results_copy["file_version"] = "2.0"

            # 保存结果文件
            result_filename = "tts_generation_results.json"
            result_path = output_dir / result_filename

            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(results_copy, f, ensure_ascii=False, indent=2)

            self.logger.info(f"TTS结果已保存: {result_path}")

        except Exception as e:
            self.logger.error(f"保存TTS结果失败: {str(e)}")

    def clear_cache(self):
        """清理TTS模型缓存"""
        if self.engine:
            self.engine.clear_cache()

    def __del__(self):
        """析构时清理资源"""
        try:
            self.clear_cache()
        except Exception:
            pass  # 析构时忽略错误


# 单例实例（按引擎类型缓存）
_tts_processor_instances = {}


def get_tts_processor(engine_type: TTSEngineType = TTSEngineType.INDEX_TTS, engine_config: Optional[Dict[str, Any]] = None) -> TTSProcessor:
    """获取 TTSProcessor 单例实例"""
    global _tts_processor_instances
    
    # 只按引擎类型缓存，忽略配置差异
    engine_type_str = engine_type.value
    
    if engine_type_str not in _tts_processor_instances:
        _tts_processor_instances[engine_type_str] = TTSProcessor(engine_type, engine_config)
    
    return _tts_processor_instances[engine_type_str]


# 便捷函数
def generate_tts_from_reference(
        reference_results_path: str,
        output_dir: Optional[str] = None,
        engine_type: TTSEngineType = TTSEngineType.INDEX_TTS,
        engine_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    便捷函数：根据参考音频结果生成TTS语音

    Args:
        reference_results_path: 参考音频结果JSON文件路径
        output_dir: TTS输出目录，默认为JSON文件所在目录下的tts_output
        engine_type: TTS引擎类型 (TTSEngineType枚举)
        engine_config: TTS引擎配置

    Returns:
        包含TTS生成结果的字典
    """
    return get_tts_processor(engine_type, engine_config).generate_tts_from_reference(
        reference_results_path, output_dir
    )


def clear_tts_cache(engine_type: TTSEngineType = TTSEngineType.INDEX_TTS, engine_config: Optional[Dict[str, Any]] = None):
    """
    便捷函数：清理TTS模型缓存

    Args:
        engine_type: TTS引擎类型 (TTSEngineType枚举)
        engine_config: TTS引擎配置
    """
    get_tts_processor(engine_type, engine_config).clear_cache()


def list_available_engines() -> Dict[str, Dict[str, Any]]:
    """
    便捷函数：列出所有可用的TTS引擎信息

    Returns:
        包含所有引擎详细信息的字典
    """
    return TTSEngineType.get_engine_info()


def get_engine_default_config(engine_type: TTSEngineType) -> Dict[str, Any]:
    """
    便捷函数：获取指定引擎的默认配置

    Args:
        engine_type: TTS引擎类型

    Returns:
        引擎的默认配置
    """
    engine_type_enum = engine_type
    
    return engine_type_enum.get_default_config()