"""
CosyVoice TTS引擎实现
基于CosyVoice的声音克隆功能
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


class CosyVoiceTTSEngine:
    """CosyVoice TTS引擎实现"""

    def __init__(self, api_url: str = "http://127.0.0.1:8000"):
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
            self.logger.info("初始化CosyVoice Gradio客户端...")
            self.gradio_client = Client(
                self.api_url,
                httpx_kwargs={"timeout": 120, "proxy": None},
                ssl_verify=False,
                download_files=str(self.gradio_temp_dir),
            )
            self.logger.info(f"CosyVoice客户端初始化成功: {self.api_url}")
        except Exception as e:
            self.logger.error(f"CosyVoice客户端初始化失败: {str(e)}")
            self.gradio_client = None

    def generate_audio(
        self, 
        text: str, 
        output_path: str, 
        reference_file: str = None, 
        prompt_text: str = "",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """生成TTS音频
        
        Args:
            text: 要合成的文本
            output_path: 输出文件路径
            reference_file: 参考音频文件（可选）
            prompt_text: 提示文本（可选，用于3s极速复刻模式）
            **kwargs: 其他参数
                - mode: 推理模式，默认为"3s极速复刻"
                - sft_dropdown: 预训练音色选择
                - instruct_text: 指令文本（用于自然语言控制模式）
                - seed: 随机种子
                - stream: 是否流式推理
                - speed: 语速调节
        """
        try:
            # 设置默认参数
            mode = kwargs.get("mode", "3s极速复刻")
            sft_dropdown = kwargs.get("sft_dropdown", "")
            instruct_text = kwargs.get("instruct_text", "")
            seed = kwargs.get("seed", 0)
            stream = kwargs.get("stream", False)
            speed = kwargs.get("speed", 1.0)

            # 验证输入
            if not text.strip():
                self.logger.error("合成文本不能为空")
                return None

            # 根据模式调用不同的API
            if mode == "预训练音色":
                if not sft_dropdown:
                    self.logger.error("预训练音色模式需要选择音色")
                    return None
                return self._generate_sft_audio(text, output_path, sft_dropdown, stream, speed, seed)
            
            elif mode == "3s极速复刻":
                if not reference_file or not os.path.exists(reference_file):
                    self.logger.error("3s极速复刻模式需要参考音频文件")
                    return None
                if not prompt_text.strip():
                    self.logger.error("3s极速复刻模式需要提示文本")
                    return None
                return self._generate_zero_shot_audio(text, output_path, reference_file, prompt_text, stream, speed, seed)
            
            elif mode == "跨语种复刻":
                if not reference_file or not os.path.exists(reference_file):
                    self.logger.error("跨语种复刻模式需要参考音频文件")
                    return None
                return self._generate_cross_lingual_audio(text, output_path, reference_file, stream, speed, seed)
            
            elif mode == "自然语言控制":
                if not sft_dropdown:
                    self.logger.error("自然语言控制模式需要选择预训练音色")
                    return None
                if not instruct_text.strip():
                    self.logger.error("自然语言控制模式需要指令文本")
                    return None
                return self._generate_instruct_audio(text, output_path, sft_dropdown, instruct_text, stream, speed, seed)
            
            else:
                self.logger.error(f"不支持的推理模式: {mode}")
                return None

        except Exception as e:
            self.logger.error(f"CosyVoice生成失败: {str(e)}")
            return None

    def _generate_sft_audio(
        self, text: str, output_path: str, sft_dropdown: str, stream: bool, speed: float, seed: int
    ) -> Optional[Dict[str, Any]]:
        """预训练音色模式"""
        try:
            result = self.gradio_client.predict(
                text,  # tts_text
                "预训练音色",  # mode_checkbox_group
                sft_dropdown,  # sft_dropdown
                "",  # prompt_text
                None,  # prompt_wav_upload
                None,  # prompt_wav_record
                "",  # instruct_text
                seed,  # seed
                stream,  # stream
                speed,  # speed
                api_name="/generate_audio"
            )

            if self._process_gradio_result(result, output_path):
                self.logger.debug(f"CosyVoice SFT模式生成成功: {output_path}")
                return {
                    "engine_type": "cosyvoice_sft",
                    "voice_cloned": False,
                    "generated_silence": False,
                    "metadata": {
                        "original_text": text,
                        "mode": "预训练音色",
                        "sft_dropdown": sft_dropdown,
                        "seed": seed,
                        "stream": stream,
                        "speed": speed,
                    },
                }
            return None

        except Exception as e:
            self.logger.error(f"CosyVoice SFT模式生成失败: {str(e)}")
            return None

    def _generate_zero_shot_audio(
        self, text: str, output_path: str, reference_file: str, prompt_text: str, 
        stream: bool, speed: float, seed: int
    ) -> Optional[Dict[str, Any]]:
        """3s极速复刻模式"""
        try:
            result = self.gradio_client.predict(
                text,  # tts_text
                "3s极速复刻",  # mode_checkbox_group
                "",  # sft_dropdown
                prompt_text,  # prompt_text
                handle_file(reference_file),  # prompt_wav_upload
                None,  # prompt_wav_record
                "",  # instruct_text
                seed,  # seed
                stream,  # stream
                speed,  # speed
                api_name="/generate_audio"
            )

            if self._process_gradio_result(result, output_path):
                self.logger.debug(f"CosyVoice 3s极速复刻模式生成成功: {output_path}")
                return {
                    "engine_type": "cosyvoice_zero_shot",
                    "voice_cloned": True,
                    "generated_silence": False,
                    "metadata": {
                        "original_text": text,
                        "mode": "3s极速复刻",
                        "prompt_text": prompt_text,
                        "reference_file": reference_file,
                        "seed": seed,
                        "stream": stream,
                        "speed": speed,
                    },
                }
            return None

        except Exception as e:
            self.logger.error(f"CosyVoice 3s极速复刻模式生成失败: {str(e)}")
            return None

    def _generate_cross_lingual_audio(
        self, text: str, output_path: str, reference_file: str, 
        stream: bool, speed: float, seed: int
    ) -> Optional[Dict[str, Any]]:
        """跨语种复刻模式"""
        try:
            result = self.gradio_client.predict(
                text,  # tts_text
                "跨语种复刻",  # mode_checkbox_group
                "",  # sft_dropdown
                "",  # prompt_text
                handle_file(reference_file),  # prompt_wav_upload
                None,  # prompt_wav_record
                "",  # instruct_text
                seed,  # seed
                stream,  # stream
                speed,  # speed
                api_name="/generate_audio"
            )

            if self._process_gradio_result(result, output_path):
                self.logger.debug(f"CosyVoice 跨语种复刻模式生成成功: {output_path}")
                return {
                    "engine_type": "cosyvoice_cross_lingual",
                    "voice_cloned": True,
                    "generated_silence": False,
                    "metadata": {
                        "original_text": text,
                        "mode": "跨语种复刻",
                        "reference_file": reference_file,
                        "seed": seed,
                        "stream": stream,
                        "speed": speed,
                    },
                }
            return None

        except Exception as e:
            self.logger.error(f"CosyVoice 跨语种复刻模式生成失败: {str(e)}")
            return None

    def _generate_instruct_audio(
        self, text: str, output_path: str, sft_dropdown: str, instruct_text: str,
        stream: bool, speed: float, seed: int
    ) -> Optional[Dict[str, Any]]:
        """自然语言控制模式"""
        try:
            result = self.gradio_client.predict(
                text,  # tts_text
                "自然语言控制",  # mode_checkbox_group
                sft_dropdown,  # sft_dropdown
                "",  # prompt_text
                None,  # prompt_wav_upload
                None,  # prompt_wav_record
                instruct_text,  # instruct_text
                seed,  # seed
                stream,  # stream
                speed,  # speed
                api_name="/generate_audio"
            )

            if self._process_gradio_result(result, output_path):
                self.logger.debug(f"CosyVoice 自然语言控制模式生成成功: {output_path}")
                return {
                    "engine_type": "cosyvoice_instruct",
                    "voice_cloned": False,
                    "generated_silence": False,
                    "metadata": {
                        "original_text": text,
                        "mode": "自然语言控制",
                        "sft_dropdown": sft_dropdown,
                        "instruct_text": instruct_text,
                        "seed": seed,
                        "stream": stream,
                        "speed": speed,
                    },
                }
            return None

        except Exception as e:
            self.logger.error(f"CosyVoice 自然语言控制模式生成失败: {str(e)}")
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
                "engine_type": "cosyvoice_silence",
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
            return f"CosyVoice@{self.api_url}"
        else:
            return "cosyvoice_mock_v1.0"

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