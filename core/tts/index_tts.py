"""
Index-TTS引擎实现
基于Gradio客户端的声音克隆功能
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
from gradio_client import Client, handle_file


class IndexTTSEngine:
    """Index-TTS引擎实现"""

    def __init__(self, api_url: str = "http://127.0.0.1:7860"):
        self.logger = logging.getLogger(__name__)
        self.api_url = api_url
        self.gradio_client = None
        self.gradio_temp_dir = self._get_gradio_temp_dir()
        self._initialize_gradio_client()

    def _get_gradio_temp_dir(self) -> Path:
        """获取Gradio临时目录路径"""
        gradio_temp = os.environ.get("GRADIO_TEMP_DIR")
        if gradio_temp:
            return Path(gradio_temp)
        return Path(tempfile.gettempdir()) / "gradio"

    def _initialize_gradio_client(self):
        """初始化Gradio客户端"""
        try:
            self.logger.info("初始化Index-TTS Gradio客户端...")
            self.gradio_client = Client(
                self.api_url,
                httpx_kwargs={"timeout": 120, "proxy": None},
                ssl_verify=False,
                download_files=str(self.gradio_temp_dir),
            )
            self.logger.info(f"Index-TTS客户端初始化成功: {self.api_url}")
        except Exception as e:
            self.logger.error(f"Index-TTS客户端初始化失败: {str(e)}")
            self.gradio_client = None

    def generate_audio(
        self, text: str, output_path: str, reference_file: str = None, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """生成TTS音频"""
        try:
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
                int(params["max_text_tokens_per_sentence"]),  # max_text_tokens_per_sentence
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
                if self._process_gradio_result(result, output_path):
                    self.logger.debug(f"Index-TTS生成成功: {output_path}")
                    return {
                        "engine_type": "index_tts",
                        "voice_cloned": True,
                        "generated_silence": False,
                        "metadata": {"original_text": text, "method": "gradio_api"},
                    }

            self.logger.error("Index-TTS API调用失败")
            return None

        except Exception as e:
            self.logger.error(f"Index-TTS生成失败: {str(e)}")
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

    def generate_silence_audio(self, output_path: str, duration: float) -> Dict[str, Any]:
        """生成静音音频"""
        try:
            duration = max(0.1, duration)  # 最小0.1秒
            sample_rate = 22050
            samples = int(sample_rate * duration)

            # 生成静音音频
            silence_data = np.zeros(samples, dtype=np.float32)
            sf.write(output_path, silence_data, sample_rate)

            return {
                "success": True,
                "engine_type": "index_tts_silence",
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

    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """获取音频文件信息"""
        try:
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

    def get_model_info(self) -> str:
        """获取模型信息"""
        if self.gradio_client:
            return f"Index-TTS@{self.api_url}"
        else:
            return "index_tts_mock_v1.0"

    def clear_cache(self):
        """清理缓存"""
        self._cleanup_gradio_temp_files()

    def _cleanup_gradio_temp_files(self):
        """清理Gradio临时文件"""
        try:
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
            pass