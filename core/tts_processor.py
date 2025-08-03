"""
TTS处理器
直接基于Gradio客户端实现Index-TTS声音克隆功能
"""

import datetime
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
from gradio_client import Client, handle_file


class TTSProcessor:
    """独立TTS处理器，直接基于Gradio客户端实现声音克隆"""

    def __init__(self, api_url: str = "http://127.0.0.1:7860"):
        self.logger = logging.getLogger(__name__)

        # Index-TTS API配置
        self.api_url = api_url
        self.gradio_client = None

        # 获取Gradio临时目录
        self.gradio_temp_dir = TTSProcessor._get_gradio_temp_dir()

        # 初始化Gradio客户端
        self._initialize_gradio_client()

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
                    "api_url": self.api_url,
                    "gradio_available": True,
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

    def _initialize_gradio_client(self):
        """初始化Gradio客户端"""

        try:
            self.logger.info("初始化Gradio客户端...")

            # 创建Gradio客户端，使用动态临时目录
            self.gradio_client = Client(
                self.api_url,
                httpx_kwargs={"timeout": 120, "proxy": None},
                ssl_verify=False,
                download_files=str(self.gradio_temp_dir),
            )

            self.logger.info(f"Gradio客户端初始化成功: {self.api_url}")
            self.logger.info(f"Gradio临时目录: {self.gradio_temp_dir}")

        except Exception as e:
            self.logger.error(f"Gradio客户端初始化失败: {str(e)}")
            self.gradio_client = None

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
            self, text: str, output_path: str, reference_file: str = None
    ) -> Optional[Dict[str, Any]]:
        """调用TTS模型生成音频"""
        try:
            return self._call_gradio_tts(text, output_path, reference_file)
        except Exception as e:
            self.logger.error(f"TTS生成调用失败: {str(e)}")
            return None

    def _call_gradio_tts(
            self, text: str, output_path: str, reference_file: str = None
    ) -> Optional[Dict[str, Any]]:
        """调用Gradio Index-TTS API生成音频"""
        try:
            self.logger.debug(
                f"调用Gradio Index-TTS: '{text[:30]}...' -> {output_path}"
            )

            # 验证参考音频文件存在
            if not reference_file or not os.path.exists(reference_file):
                self.logger.error(f"参考音频文件不存在: {reference_file}")
                return None

            # 准备Index-TTS参数
            params = {
                "infer_mode": "普通推理",
                "max_text_tokens_per_sentence": 120,
                "sentences_bucket_max_size": 4,
                "do_sample": True,
                "top_p": 0.8,
                "top_k": 30,
                "temperature": 1.0,
                "length_penalty": 0.0,
                "num_beams": 3,
                "repetition_penalty": 10.0,
                "max_mel_tokens": 600,
            }

            # 调用Index-TTS API
            result = self.gradio_client.predict(
                handle_file(reference_file),  # prompt (参考音频)
                text,  # text (待合成文本)
                params["infer_mode"],  # infer_mode
                int(
                    params["max_text_tokens_per_sentence"]
                ),  # max_text_tokens_per_sentence
                int(params["sentences_bucket_max_size"]),  # sentences_bucket_max_size
                params["do_sample"],  # do_sample
                float(params["top_p"]),  # top_p
                int(params["top_k"]) if int(params["top_k"]) > 0 else 0,  # top_k
                float(params["temperature"]),  # temperature
                float(params["length_penalty"]),  # length_penalty
                int(params["num_beams"]),  # num_beams
                float(params["repetition_penalty"]),  # repetition_penalty
                int(params["max_mel_tokens"]),  # max_mel_tokens
                api_name="/gen_single",
            )

            if result:
                # 处理返回的音频文件
                audio_data = self._process_gradio_result(result, output_path)
                if audio_data:
                    self.logger.debug(f"Gradio Index-TTS生成成功: {output_path}")
                    return {
                        "engine_type": "index_tts",
                        "voice_cloned": True,
                        "generated_silence": False,
                        "metadata": {"original_text": text, "method": "gradio_api"},
                    }

            self.logger.error("Gradio Index-TTS API调用失败")
            return None

        except Exception as e:
            self.logger.error(f"Gradio Index-TTS调用失败: {str(e)}")
            return None

    def _process_gradio_result(self, result: Any, output_path: str) -> bool:
        """处理Gradio API返回的结果"""
        try:
            # 处理不同格式的返回结果
            if isinstance(result, dict) and "value" in result:
                audio_path = result["value"]
            elif isinstance(result, str):
                audio_path = result
            elif hasattr(result, "name"):
                audio_path = result.name
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                audio_path = result[0]
            else:
                self.logger.error(f"无法识别的Gradio结果格式: {type(result)}")
                return False

            # 复制音频文件到目标路径
            if os.path.exists(audio_path):
                shutil.copy2(audio_path, output_path)
                return True
            else:
                self.logger.error(f"音频文件不存在: {audio_path}")
                return False

        except Exception as e:
            self.logger.error(f"处理Gradio结果失败: {str(e)}")
            return False

    @staticmethod
    def _get_gradio_temp_dir() -> Path:
        """获取Gradio临时目录路径"""
        # 优先使用环境变量GRADIO_TEMP_DIR
        gradio_temp = os.environ.get("GRADIO_TEMP_DIR")
        if gradio_temp:
            return Path(gradio_temp)

        # 否则使用系统临时目录下的gradio子目录
        return Path(tempfile.gettempdir()) / "gradio"

    def _generate_silence_audio(
            self, output_path: str, duration: float
    ) -> Dict[str, Any]:
        """生成静音音频

        Args:
            output_path: 输出文件路径
            duration: 静音时长（秒）

        Returns:
            生成结果字典
        """
        try:
            # 使用指定的时长
            duration = max(0.1, duration)  # 最小0.1秒
            sample_rate = 22050
            samples = int(sample_rate * duration)

            # 生成静音音频
            silence_data = np.zeros(samples, dtype=np.float32)

            # 保存音频文件
            sf.write(output_path, silence_data, sample_rate)

            return {
                "success": True,
                "engine_type": "silence",
                "voice_cloned": False,
                "generated_silence": True,
                "file_path": output_path,
                "metadata": {
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "samples": samples,
                    "method": "silence_generation",
                },
            }

        except Exception as e:
            self.logger.error(f"静音音频生成失败: {str(e)}")
            return {"success": False, "error": str(e)}

    def _get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """获取音频文件信息"""
        try:
            # 使用soundfile获取音频信息
            info = sf.info(audio_path)
            file_size = os.path.getsize(audio_path)

            return {
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "file_size": file_size,
            }

        except Exception as e:
            self.logger.warning(f"获取音频信息失败: {str(e)}")
            return {"duration": 0, "sample_rate": 22050, "channels": 1, "file_size": 0}

    def _get_model_info(self) -> str:
        """获取模型信息"""
        if self.gradio_client:
            return f"Index-TTS@{self.api_url}"
        else:
            return "mock_tts_model_v1.0"

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
        # 清理Gradio临时文件
        self._cleanup_gradio_temp_files()

    def _cleanup_gradio_temp_files(self):
        """清理Gradio临时文件"""
        try:
            # 使用动态获取的Gradio临时目录
            if self.gradio_temp_dir.exists():
                folder_count = 0
                for item in self.gradio_temp_dir.iterdir():
                    if item.is_dir():
                        try:
                            shutil.rmtree(item, ignore_errors=True)
                            if not item.exists():
                                folder_count += 1
                        except Exception as e:
                            self.logger.warning(f"清理临时目录失败: {e}")

                if folder_count > 0:
                    self.logger.info(f"清理了 {folder_count} 个Gradio临时目录")

        except Exception as e:
            self.logger.error(f"Gradio临时文件清理失败: {str(e)}")

    def __del__(self):
        """析构时清理资源"""
        try:
            self.clear_cache()
        except Exception:
            pass  # 析构时忽略错误


# 单例实例
_tts_processor_instance = None


def get_tts_processor(api_url: str = "http://127.0.0.1:7860") -> TTSProcessor:
    """获取 TTSProcessor 单例实例"""
    global _tts_processor_instance
    if _tts_processor_instance is None:
        _tts_processor_instance = TTSProcessor(api_url)
    return _tts_processor_instance


# 便捷函数
def generate_tts_from_reference(
        reference_results_path: str,
        output_dir: Optional[str] = None,
        api_url: str = "http://127.0.0.1:7860",
) -> Dict[str, Any]:
    """
    便捷函数：根据参考音频结果生成TTS语音

    Args:
        reference_results_path: 参考音频结果JSON文件路径
        output_dir: TTS输出目录，默认为JSON文件所在目录下的tts_output
        api_url: TTS API URL

    Returns:
        包含TTS生成结果的字典
    """
    return get_tts_processor(api_url).generate_tts_from_reference(
        reference_results_path, output_dir
    )


def clear_tts_cache(api_url: str = "http://127.0.0.1:7860"):
    """
    便捷函数：清理TTS模型缓存

    Args:
        api_url: TTS API URL
    """
    get_tts_processor(api_url).clear_cache()
