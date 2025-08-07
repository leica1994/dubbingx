"""
媒体处理核心功能
优化版本 - 集成到流水线系统中
"""

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import ffmpeg
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio


class MediaProcessorCore:
    """媒体处理核心类，专注于音视频分离"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.use_gpu = self._detect_gpu_capabilities()

        # 内存缓存系统 - 充分利用48GB内存
        import psutil

        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)

        # 根据可用内存动态设置缓存大小
        if available_memory_gb > 32:
            # 48GB+ 内存配置 - 使用更大的缓存
            self.cache_size_limit = min(
                8, int(available_memory_gb * 0.15)
            )  # 使用15%内存作为缓存
            self.audio_cache_max_size = 20  # 缓存更多音频文件
            self.enable_aggressive_caching = True
            self.logger.info(
                f"检测到大内存({available_memory_gb:.1f}GB)，启用增强缓存模式"
            )
        elif available_memory_gb > 16:
            # 16-32GB 内存配置
            self.cache_size_limit = 4
            self.audio_cache_max_size = 10
            self.enable_aggressive_caching = False
        else:
            # 低内存配置
            self.cache_size_limit = 1
            self.audio_cache_max_size = 3
            self.enable_aggressive_caching = False

        # 缓存存储
        self._audio_cache = {}  # 音频文件缓存
        self._model_cache = {}  # 模型缓存
        self._result_cache = {}  # 计算结果缓存
        self._cache_usage = 0  # 当前缓存使用量(GB)

    def _detect_gpu_capabilities(self) -> bool:
        """检测GPU硬件解码能力，包括PyTorch CUDA和FFmpeg CUDA支持
        
        Returns:
            bool: 是否可以使用GPU硬件加速
        """
        try:
            # 1. 检查PyTorch CUDA支持
            if not torch.cuda.is_available():
                self.logger.info("PyTorch CUDA不可用，使用CPU处理")
                return False
            
            # 2. 检查FFmpeg CUDA硬件解码支持
            if not self._check_ffmpeg_cuda_support():
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.warning(f"检测到GPU {gpu_name}，但FFmpeg不支持CUDA硬件解码，使用CPU处理")
                return False
            
            # 3. GPU信息输出
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.logger.info(f"使用GPU加速: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
            
        except Exception as e:
            self.logger.warning(f"GPU检测失败，使用CPU处理: {str(e)}")
            return False
    
    def _check_ffmpeg_cuda_support(self) -> bool:
        """检查FFmpeg是否支持CUDA硬件解码
        
        Returns:
            bool: FFmpeg是否支持CUDA
        """
        try:
            import subprocess
            
            # 检查FFmpeg是否支持cuda硬件加速
            result = subprocess.run(
                ["ffmpeg", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and "cuda" in result.stdout.lower():
                self.logger.debug("FFmpeg支持CUDA硬件解码")
                return True
            else:
                self.logger.debug("FFmpeg不支持CUDA硬件解码")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            self.logger.debug(f"FFmpeg CUDA检测失败: {str(e)}")
            return False
    
    def _analyze_gpu_decode_failure(self, error_msg: str) -> str:
        """分析GPU解码失败的可能原因
        
        Args:
            error_msg: 错误消息
            
        Returns:
            失败原因的描述
        """
        error_lower = error_msg.lower()
        
        # 常见GPU解码失败原因分析
        if "out of memory" in error_lower or "cuda out of memory" in error_lower:
            return "显存不足"
        elif "no cuda capable" in error_lower or "cuda device" in error_lower:
            return "CUDA设备不可用"
        elif "unsupported format" in error_lower or "codec not supported" in error_lower:
            return "视频编码格式不支持硬件解码"
        elif "invalid argument" in error_lower or "invalid data" in error_lower:
            return "视频文件格式或参数无效"
        elif "decoder" in error_lower and ("not found" in error_lower or "unavailable" in error_lower):
            return "找不到合适的硬件解码器"
        elif "permission denied" in error_lower:
            return "权限不足"
        elif "busy" in error_lower or "device busy" in error_lower:
            return "GPU设备繁忙"
        elif "hardware accelerated" in error_lower or "hwaccel" in error_lower:
            return "硬件加速初始化失败"
        elif "format not supported" in error_lower or "pixel format" in error_lower:
            return "像素格式不支持"
        elif "cannot initialize" in error_lower or "initialization failed" in error_lower:
            return "CUDA初始化失败"
        elif "nvenc" in error_lower or "nvdec" in error_lower:
            return "NVIDIA编解码器问题"
        elif "driver" in error_lower:
            return "显卡驱动问题"
        elif "timeout" in error_lower:
            return "GPU处理超时"
        elif "connection refused" in error_lower or "device not found" in error_lower:
            return "GPU设备连接失败"
        elif "resource temporarily unavailable" in error_lower:
            return "GPU资源暂时不可用"
        elif "impossible to convert between" in error_lower or "auto_scale" in error_lower:
            return "像素格式转换失败(需要移除hwaccel_output_format参数)"
        elif "function not implemented" in error_lower:
            return "GPU硬件解码功能未实现或不支持"
        elif "ffmpeg error" in error_lower and len(error_msg) < 50:
            return "FFmpeg执行错误(可能是兼容性问题)"
        else:
            # 尝试从错误消息中提取更有用的信息
            if len(error_msg) > 100:
                return f"GPU解码错误: {error_msg[:100]}..."
            else:
                return f"GPU解码错误: {error_msg}"

    def separate_media(
            self,
            video_path: str,
            output_dir: Optional[str] = None,
            enable_vocal_separation: bool = False,
            audio_quality_level: int = 0,
            video_quality_level: int = 0,
    ) -> Dict[str, Any]:
        """
        根据视频路径分离人声、无声视频和背景音乐

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录，默认为视频文件所在目录
            enable_vocal_separation: 是否启用人声分离，False时只分离音视频
            audio_quality_level: 音频质量级别 (0=最高质量, 1=高质量, 2=标准质量)
            video_quality_level: 视频质量级别 (0=最高质量, 1=高质量压缩, 2=极速压缩)

        Returns:
            包含分离结果的字典
        """
        try:
            # 验证输入文件
            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "error": f"视频文件不存在: {video_path}",
                }

            # 设置输出目录
            if output_dir is None:
                output_dir = Path(video_path).parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

            video_name = Path(video_path).stem
            video_ext = '.mp4'

            # 定义输出文件路径
            silent_video_path = output_dir / f"{video_name}_silent{video_ext}"
            raw_audio_path = output_dir / f"{video_name}_raw_audio.wav"
            vocal_audio_path = output_dir / f"{video_name}_vocal.wav"
            background_audio_path = output_dir / f"{video_name}_background.wav"

            self.logger.info(f"开始处理视频: {video_path}")

            # 1. 提取原始音频
            self._extract_audio(video_path, str(raw_audio_path), audio_quality_level)

            # 2. 创建无声视频
            self._create_silent_video(video_path, str(silent_video_path), video_quality_level)

            # 3. 根据选项决定是否分离音频（人声和背景音乐）
            if enable_vocal_separation:
                self.logger.info("使用Demucs进行完整音频分离（人声/背景音乐）")
                self._separate_audio(
                    str(raw_audio_path),
                    str(vocal_audio_path),
                    str(background_audio_path),
                )
            else:
                self.logger.info(
                    "快速模式：直接使用原始音频作为人声，跳过Demucs分离和背景音乐处理"
                )
                # 直接复制原始音频作为人声
                import shutil

                shutil.copy2(str(raw_audio_path), str(vocal_audio_path))

                # 快速模式下不创建背景音乐文件

            # 4. 清理临时文件
            if raw_audio_path.exists():
                raw_audio_path.unlink()

            self.logger.info("媒体分离完成")

            return {
                "success": True,
                "silent_video_path": str(silent_video_path),
                "vocal_audio_path": str(vocal_audio_path),
                "background_audio_path": (
                    str(background_audio_path) if enable_vocal_separation else ""
                ),
                "separation_info": {
                    "video_name": video_name,
                    "video_format": video_ext,
                    "output_directory": str(output_dir),
                    "gpu_acceleration": self.use_gpu,
                    "vocal_separation_enabled": enable_vocal_separation,
                    "separation_mode": (
                        "完整分离" if enable_vocal_separation else "快速模式"
                    ),
                },
            }

        except Exception as e:
            self.logger.error(f"媒体分离失败: {str(e)}")
            return {"success": False, "error": str(e)}

    def generate_reference_audio(
            self,
            audio_path: str,
            subtitle_path: str,
            output_dir: Optional[str] = None,
            normalize_volume: bool = True,
    ) -> Dict[str, Any]:
        """
        根据字幕分隔音频生成参考音频

        Args:
            audio_path: 音频文件路径
            subtitle_path: 字幕文件路径
            output_dir: 输出目录
            normalize_volume: 是否进行音量一致性增强

        Returns:
            包含参考音频生成结果的字典
        """
        try:
            # 验证输入文件
            if not os.path.exists(audio_path):
                return {
                    "success": False,
                    "error": f"音频文件不存在: {audio_path}",
                }

            if not os.path.exists(subtitle_path):
                return {
                    "success": False,
                    "error": f"字幕文件不存在: {subtitle_path}",
                }

            # 设置输出目录
            if output_dir is None:
                output_dir = Path(audio_path).parent / "reference_audio"
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info("开始处理参考音频生成...")

            # 解析字幕文件
            subtitle_entries = self._parse_subtitle_file(subtitle_path)
            if not subtitle_entries:
                return {"success": False, "error": "字幕文件解析失败或为空"}

            # 加载音频文件
            audio_data, sample_rate = self._load_audio(audio_path)

            # 生成参考音频片段
            reference_segments = []
            successful_segments = 0
            audio_name = Path(audio_path).stem

            # 设置音量处理策略
            if normalize_volume:
                target_rms = 0.15  # 舒适听感音量
                self.logger.info(
                    f"启用音量一致性增强，目标RMS音量: {target_rms:.6f}（舒适听感标准）"
                )
            else:
                target_rms = None
                self.logger.info("音量一致性增强已禁用，保持原始音量")

            # 处理和保存音频片段
            for i, entry in enumerate(subtitle_entries):
                try:
                    start_time = entry["start_time"]
                    end_time = entry["end_time"]
                    text = entry["text"]

                    # 验证时间戳
                    if start_time >= end_time or start_time < 0:
                        self.logger.warning(
                            f"跳过无效时间戳的字幕片段 {i + 1}: {start_time}-{end_time}"
                        )
                        continue

                    # 提取音频片段（包括空字幕对应的音频片段）
                    segment_audio = self._extract_audio_segment(
                        audio_data, sample_rate, start_time, end_time
                    )

                    # 静音检测
                    is_silence = self._is_silence_segment(segment_audio)

                    # 质量检查
                    quality_score = self._calculate_audio_quality(
                        segment_audio, sample_rate
                    )

                    # 音量一致性增强（可选）
                    if normalize_volume and target_rms and not is_silence:
                        segment_audio = self._normalize_audio_volume(
                            segment_audio, target_rms
                        )

                    # 保存音频片段
                    segment_filename = f"{audio_name}_ref_{i + 1:04d}.wav"
                    segment_path = output_dir / segment_filename

                    sf.write(str(segment_path), segment_audio, sample_rate)

                    # 记录片段信息
                    segment_info = {
                        "index": i + 1,
                        "text": text,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": round(end_time - start_time, 3),
                        "file_path": str(segment_path),
                        "quality_score": quality_score,
                        "sample_rate": sample_rate,
                        "audio_length": len(segment_audio),
                        "is_silence": is_silence,
                    }
                    reference_segments.append(segment_info)
                    successful_segments += 1

                except Exception as e:
                    self.logger.warning(f"处理字幕片段 {i + 1} 失败: {str(e)}")
                    continue

            self.logger.info(
                f"参考音频生成完成: {successful_segments}/{len(subtitle_entries)} 个片段成功"
            )

            # 静音片段处理
            silence_count = sum(
                1 for seg in reference_segments if seg.get("is_silence", False)
            )
            if silence_count > 0:
                self.logger.info(f"检测到 {silence_count} 个静音片段，开始替换处理...")
                reference_segments = self._replace_silence_segments(reference_segments)

            # 构建返回结果
            results = {
                "success": True,
                "reference_audio_segments": reference_segments,
                "output_dir": str(output_dir),
                "total_segments": successful_segments,
                "total_requested": len(subtitle_entries),
                "volume_normalization": {
                    "enabled": normalize_volume,
                    "target_rms": target_rms if normalize_volume else None,
                    "target_db": -16.5 if normalize_volume else None,
                },
                "silence_detection": {
                    "silence_segments_found": silence_count,
                    "segments_replaced": sum(
                        1 for seg in reference_segments if seg.get("replaced_with")
                    ),
                },
                "audio_info": {
                    "sample_rate": sample_rate,
                    "duration": len(audio_data) / sample_rate,
                    "channels": (
                        1 if len(audio_data.shape) == 1 else audio_data.shape[1]
                    ),
                },
            }

            # 保存结果到JSON文件
            json_file_path = self._save_results_to_json(
                results, str(output_dir), audio_name
            )
            if json_file_path:
                results["results_json_file"] = json_file_path

            return results

        except Exception as e:
            self.logger.error(f"参考音频生成失败: {str(e)}")
            return {"success": False, "error": str(e)}

    def merge_audio_video(
            self,
            video_path: str,
            audio_path: str,
            output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        合并音频和视频文件，自动处理时长差异

        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径
            output_path: 输出文件路径

        Returns:
            包含合并结果的字典
        """
        try:
            # 验证输入文件
            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "error": f"视频文件不存在: {video_path}",
                }

            if not os.path.exists(audio_path):
                return {
                    "success": False,
                    "error": f"音频文件不存在: {audio_path}",
                }

            # 设置输出路径
            if output_path is None:
                video_name = Path(video_path).stem
                video_parent_dir = Path(video_path).parent.parent
                output_dir = video_parent_dir / "final_video"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{video_name}_merged.mp4"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"开始合并音视频: 视频={video_path}, 音频={audio_path}")

            # 获取音视频时长
            video_duration = self._get_video_duration(video_path)
            audio_duration = self._get_audio_duration(audio_path)

            self.logger.info(
                f"视频时长: {video_duration:.2f}s, 音频时长: {audio_duration:.2f}s"
            )

            # 确定最终时长和需要的调整
            final_duration = max(video_duration, audio_duration)
            video_extended = audio_duration > video_duration
            audio_extended = video_duration > audio_duration

            # 使用ffmpeg合并音视频，带时长调整
            self._merge_audio_video_with_duration_adjustment(
                video_path,
                audio_path,
                str(output_path),
                video_duration,
                audio_duration,
                final_duration,
            )

            self.logger.info(f"音视频合并完成: {output_path}")

            return {
                "success": True,
                "output_path": str(output_path),
                "video_duration": video_duration,
                "audio_duration": audio_duration,
                "final_duration": final_duration,
                "adjustments": {
                    "video_extended": video_extended,
                    "audio_extended": audio_extended,
                },
            }

        except Exception as e:
            self.logger.error(f"音视频合并失败: {str(e)}")
            return {"success": False, "error": str(e)}

    def _log_ffmpeg_command(self, ffmpeg_stream, description: str = ""):
        """记录FFmpeg命令到日志
        
        Args:
            ffmpeg_stream: ffmpeg-python流对象
            description: 命令描述
        """
        try:
            # 使用ffmpeg-python的compile()方法获取准确的命令
            full_command = " ".join(ffmpeg_stream.compile())
            desc_text = f" ({description})" if description else ""
            self.logger.info(f"执行FFmpeg命令{desc_text}: {full_command}")

        except Exception as e:
            self.logger.debug(f"FFmpeg命令日志记录失败: {e}")

    def _extract_audio(self, video_path: str, audio_path: str, audio_quality_level: int = 0):
        """从视频中提取音频，使用GPU硬件加速

        Args:
            video_path: 输入视频路径
            audio_path: 输出音频路径
            audio_quality_level: 音频质量级别 (0=最高质量, 1=高质量, 2=标准质量)
        """
        quality_desc = None
        try:
            # RTX 4060Ti GPU加速配置
            input_args = {}

            # 根据质量级别设置音频输出参数
            if audio_quality_level == 0:  # 最高质量 (PCM无损)
                output_args = {"acodec": "pcm_s16le", "ar": 44100, "ac": 2}
                quality_desc = "最高质量(PCM无损)"
            elif audio_quality_level == 1:  # 高质量音频 (MP3 192kbps)
                output_args = {"acodec": "libmp3lame", "b:a": "192k", "ar": 44100, "ac": 2}
                quality_desc = "高质量音频(MP3 192kbps)"
            elif audio_quality_level == 2:  # 高效压缩 (AAC 128kbps)
                output_args = {"acodec": "aac", "b:a": "128k", "ar": 32000, "ac": 2}
                quality_desc = "高效压缩(AAC 128kbps)"
            else:
                # 默认回退到最高质量
                output_args = {"acodec": "pcm_s16le", "ar": 44100, "ac": 2}
                quality_desc = "最高质量(默认)"

            self.logger.info(f"提取音频使用{quality_desc}")

            # 启用GPU硬件解码加速（音频处理兼容性优化）
            if self.use_gpu:
                input_args["hwaccel"] = "cuda"
                # 音频处理不需要hwaccel_output_format，让FFmpeg自动处理
                self.logger.debug("使用CUDA硬件加速解码（音频提取）")

            # 创建FFmpeg流并记录命令到日志
            ffmpeg_stream = (
                ffmpeg.input(video_path, **input_args)
                .output(audio_path, **output_args)
                .overwrite_output()
            )
            self._log_ffmpeg_command(ffmpeg_stream, f"音频提取-{quality_desc}")

            # 执行FFmpeg命令
            ffmpeg_stream.run(quiet=True)
            self.logger.debug(f"音频提取完成: {audio_path}")
        except Exception as e:
            # 首先检查是否是MP3编码器问题，如果是则回退到AAC
            if audio_quality_level == 1 and "libmp3lame" in str(e).lower():
                try:
                    self.logger.warning(f"MP3编码器不可用，回退到AAC 192k: {str(e)}")
                    fallback_args = {"acodec": "aac", "b:a": "192k", "ar": 44100, "ac": 2}
                    input_args_fallback = {}
                    if self.use_gpu:
                        input_args_fallback["hwaccel"] = "cuda"
                        # 移除hwaccel_output_format以提高兼容性

                    # 创建回退FFmpeg流并记录命令到日志
                    fallback_stream = (
                        ffmpeg.input(video_path, **input_args_fallback)
                        .output(audio_path, **fallback_args)
                        .overwrite_output()
                    )
                    self._log_ffmpeg_command(fallback_stream, f"MP3回退到AAC-{quality_desc}")

                    # 执行回退FFmpeg命令
                    fallback_stream.run(quiet=True)
                    self.logger.debug(f"AAC回退编码完成: {audio_path}")
                    return
                except Exception:
                    pass  # 继续到下一个回退选项

            # GPU加速失败，回退到CPU处理
            try:
                # 根据质量级别设置CPU回退参数
                if audio_quality_level == 1:  # MP3级别回退到AAC
                    fallback_args = {"acodec": "aac", "b:a": "192k", "ar": 44100, "ac": 2}
                elif audio_quality_level == 2:  # AAC级别保持AAC
                    fallback_args = {"acodec": "aac", "b:a": "128k", "ar": 32000, "ac": 2}
                else:  # PCM级别或默认
                    fallback_args = {"acodec": "pcm_s16le", "ar": 44100, "ac": 2}

                # 创建CPU回退FFmpeg流并记录命令到日志
                cpu_fallback_stream = (
                    ffmpeg.input(video_path)
                    .output(audio_path, **fallback_args)
                    .overwrite_output()
                )
                self._log_ffmpeg_command(cpu_fallback_stream, f"CPU回退处理-{quality_desc}")

                # 执行CPU回退FFmpeg命令
                cpu_fallback_stream.run(quiet=True)
                self.logger.debug(f"CPU回退处理完成: {audio_path}")
            except Exception as cpu_e:
                raise Exception(f"音频提取失败: {str(cpu_e)}")

    def _create_silent_video(self, video_path: str, silent_video_path: str, video_quality_level: int = 0):
        """创建无声视频，使用统一的libx264编码配置
        
        Args:
            video_path: 输入视频路径
            silent_video_path: 输出无声视频路径  
            video_quality_level: 视频质量级别 (0=无损画质, 1=超高画质, 2=平衡质量, 3=高效压缩)
        """
        try:
            # 统一的libx264编码配置
            input_args = {}
            output_params = {}

            # 根据视频质量级别设置libx264编码参数 - 采用完善的FFmpeg参数格式
            if video_quality_level == 0:  # 无损画质 (CRF 0)
                quality_desc = "无损画质(CRF 0)"
                output_params.update({
                    "an": None,  # 明确禁用音频流
                    "c:v": "libx264",  # 使用现代语法
                    "preset": "medium",
                    "crf": "0",
                    "movflags": "+faststart"  # 优化文件结构
                })
            elif video_quality_level == 1:  # 超高画质 (CRF 18 slow)
                quality_desc = "超高画质(CRF 18 slow)"
                output_params.update({
                    "an": None,  # 明确禁用音频流
                    "c:v": "libx264",  # 使用现代语法
                    "preset": "slow",
                    "crf": "18",
                    "movflags": "+faststart"  # 优化文件结构
                })
            elif video_quality_level == 2:  # 平衡质量 (CRF 20 slow)
                quality_desc = "平衡质量(CRF 20 slow)"
                output_params.update({
                    "an": None,  # 明确禁用音频流
                    "c:v": "libx264",  # 使用现代语法
                    "preset": "slow",
                    "crf": "20",
                    "movflags": "+faststart"  # 优化文件结构
                })
            elif video_quality_level == 3:  # 高效压缩 (CRF 20 veryslow)
                quality_desc = "高效压缩(CRF 20 veryslow)"
                output_params.update({
                    "an": None,  # 明确禁用音频流
                    "c:v": "libx264",  # 使用现代语法
                    "preset": "veryslow",
                    "crf": "20",
                    "movflags": "+faststart"  # 优化文件结构
                })
            else:
                # 默认回退到无损画质
                quality_desc = "无损画质(默认)"
                output_params.update({
                    "an": None,  # 明确禁用音频流
                    "c:v": "libx264",  # 使用现代语法
                    "preset": "medium",
                    "crf": "0",
                    "movflags": "+faststart"  # 优化文件结构
                })

            self.logger.info(f"创建无声视频使用{quality_desc}")

            # GPU硬件解码优化（修复像素格式转换问题）
            if self.use_gpu:
                # 方案1：只使用CUDA解码，不使用hwaccel_output_format，让FFmpeg自动转换到系统内存
                input_args["hwaccel"] = "cuda"
                # 不设置 hwaccel_output_format，让FFmpeg自动转换到兼容格式
                self.logger.debug("使用CUDA硬件解码（自动格式转换）")
            else:
                self.logger.debug("使用CPU解码")

            # 创建FFmpeg流并记录命令到日志
            silent_video_stream = (
                ffmpeg.input(video_path, **input_args)
                .video.output(silent_video_path, **output_params)
                .overwrite_output()
            )
            self._log_ffmpeg_command(silent_video_stream, f"创建无声视频-{quality_desc}")

            try:
                # 执行FFmpeg命令并捕获详细错误信息
                silent_video_stream.run(quiet=False, capture_stdout=True, capture_stderr=True)
                self.logger.debug(f"无声视频创建完成: {silent_video_path}")
            except ffmpeg.Error as e:
                if self.use_gpu:
                    # 从ffmpeg.Error中提取stderr信息
                    stderr_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
                    
                    # GPU解码失败，分析失败原因并回退到CPU解码
                    failure_reason = self._analyze_gpu_decode_failure(stderr_output)
                    self.logger.warning(f"GPU解码失败，回退到CPU: {failure_reason}")
                    self.logger.debug(f"GPU解码错误详情: {stderr_output}")

                    # 创建CPU回退FFmpeg流并记录命令到日志
                    cpu_silent_video_stream = (
                        ffmpeg.input(video_path)
                        .video.output(silent_video_path, **output_params)
                        .overwrite_output()
                    )
                    self._log_ffmpeg_command(cpu_silent_video_stream, f"CPU回退处理-{quality_desc}")

                    try:
                        # 执行CPU回退FFmpeg命令
                        cpu_silent_video_stream.run(quiet=False, capture_stdout=True, capture_stderr=True)
                        self.logger.info(f"CPU回退处理成功: {silent_video_path}")
                    except ffmpeg.Error as cpu_e:
                        cpu_stderr = cpu_e.stderr.decode('utf-8', errors='ignore') if cpu_e.stderr else str(cpu_e)
                        self.logger.error(f"CPU回退处理也失败: {cpu_stderr}")
                        raise Exception(f"GPU和CPU处理都失败: GPU错误={stderr_output[:200]}..., CPU错误={cpu_stderr[:200]}...")
                else:
                    raise
            except Exception as e:
                if self.use_gpu:
                    # 处理其他类型的异常
                    error_msg = str(e)
                    failure_reason = self._analyze_gpu_decode_failure(error_msg)
                    self.logger.warning(f"GPU解码失败，回退到CPU: {failure_reason}")
                    self.logger.debug(f"GPU解码错误详情: {error_msg}")

                    # 创建CPU回退FFmpeg流并记录命令到日志
                    cpu_silent_video_stream = (
                        ffmpeg.input(video_path)
                        .video.output(silent_video_path, **output_params)
                        .overwrite_output()
                    )
                    self._log_ffmpeg_command(cpu_silent_video_stream, f"CPU回退处理-{quality_desc}")

                    try:
                        # 执行CPU回退FFmpeg命令
                        cpu_silent_video_stream.run(quiet=False, capture_stdout=True, capture_stderr=True)
                        self.logger.info(f"CPU回退处理成功: {silent_video_path}")
                    except Exception as cpu_e:
                        self.logger.error(f"CPU回退处理也失败: {str(cpu_e)}")
                        raise Exception(f"GPU和CPU处理都失败: GPU错误={error_msg}, CPU错误={str(cpu_e)}")
                else:
                    raise

        except Exception as e:
            raise Exception(f"无声视频创建失败: {str(e)}")

    def _separate_audio(self, audio_path: str, vocal_path: str, background_path: str):
        """使用Demucs分离人声和背景音乐，针对RTX 4060Ti 16GB显存优化"""
        try:
            # 加载和优化模型
            if self.model is None:
                self.logger.info("加载Demucs模型...")
                self.model = pretrained.get_model("mdx_extra_q")
                if self.use_gpu:
                    self.model = self.model.cuda()
                    # 启用混合精度推理以节省显存
                    self.model = self.model.half()  # 转换为FP16
                    self.logger.info("启用FP16混合精度推理，节省显存")
                self.model.eval()

            # 获取模型参数
            model_sample_rate = 44100
            model_channels = 2

            # 加载音频
            waveform, sample_rate = torchaudio.load(audio_path)

            # 转换音频格式以匹配模型要求
            waveform = convert_audio(
                waveform, sample_rate, model_sample_rate, model_channels
            )

            # 计算音频长度并决定是否分块处理
            audio_duration = waveform.shape[-1] / model_sample_rate
            max_chunk_duration = 300  # 5分钟为一块，充分利用16GB显存

            if audio_duration > max_chunk_duration and self.use_gpu:
                # 长音频分块处理，避免显存溢出
                self.logger.info(
                    f"长音频({audio_duration:.1f}s)将分块处理，充分利用16GB显存"
                )
                vocals_chunks = []
                background_chunks = []

                chunk_samples = int(max_chunk_duration * model_sample_rate)
                total_samples = waveform.shape[-1]

                for start_idx in range(0, total_samples, chunk_samples):
                    end_idx = min(start_idx + chunk_samples, total_samples)
                    chunk = waveform[:, start_idx:end_idx]

                    if self.use_gpu:
                        chunk = chunk.cuda().half()  # 使用FP16

                    # 处理块
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(
                                enabled=self.use_gpu
                        ):  # 自动混合精度
                            sources = apply_model(self.model, chunk.unsqueeze(0))[0]

                    # 提取人声和背景音乐
                    vocals_chunk = sources[3].cpu().float()  # 转回FP32保存
                    background_chunk = (
                        (sources[0] + sources[1] + sources[2]).cpu().float()
                    )

                    vocals_chunks.append(vocals_chunk)
                    background_chunks.append(background_chunk)

                    # 立即清理块内存
                    del sources, chunk
                    if self.use_gpu:
                        torch.cuda.empty_cache()

                # 合并所有块
                vocals = torch.cat(vocals_chunks, dim=-1)
                background = torch.cat(background_chunks, dim=-1)

                # 清理块列表
                del vocals_chunks, background_chunks

            else:
                # 短音频或CPU处理，直接处理整个音频
                if self.use_gpu:
                    waveform = waveform.cuda()
                    if hasattr(self.model, "half"):  # 确保模型支持FP16
                        waveform = waveform.half()

                # 应用模型进行源分离
                with torch.no_grad():
                    if self.use_gpu:
                        with torch.cuda.amp.autocast():  # 自动混合精度
                            sources = apply_model(self.model, waveform.unsqueeze(0))[0]
                    else:
                        sources = apply_model(self.model, waveform.unsqueeze(0))[0]

                # 提取人声和背景音乐
                vocals = sources[3].cpu().float()  # 转回FP32保存
                background = (sources[0] + sources[1] + sources[2]).cpu().float()

                # 立即释放中间张量
                del sources
                if self.use_gpu:
                    del waveform

            # 保存分离的音频
            sf.write(vocal_path, vocals.T.numpy(), model_sample_rate)
            sf.write(background_path, background.T.numpy(), model_sample_rate)

            # 最终显存清理
            del vocals, background
            if self.use_gpu:
                torch.cuda.empty_cache()
                # 强制垃圾回收
                import gc

                gc.collect()

            self.logger.debug(
                f"音频分离完成: 人声={vocal_path}, 背景={background_path}"
            )

        except Exception as e:
            raise Exception(f"音频分离失败: {str(e)}")
        finally:
            # 确保显存清理
            if self.use_gpu:
                torch.cuda.empty_cache()

    def _parse_subtitle_file(self, subtitle_path: str) -> List[Dict[str, Any]]:
        """解析字幕文件"""
        import re

        try:
            subtitle_format = Path(subtitle_path).suffix.lower()

            if subtitle_format == ".srt":
                return self._parse_srt_file(subtitle_path)
            else:
                self.logger.warning(f"暂不支持的字幕格式: {subtitle_format}")
                return []

        except Exception as e:
            self.logger.error(f"字幕文件解析失败: {str(e)}")
            return []

    def _parse_srt_file(self, srt_path: str) -> List[Dict[str, Any]]:
        """解析SRT字幕文件"""
        import re

        entries = []

        try:
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 分割字幕块
            subtitle_blocks = re.split(r"\n\s*\n", content.strip())

            for block in subtitle_blocks:
                if not block.strip():
                    continue

                lines = block.strip().split("\n")
                if len(lines) < 2:
                    continue

                # 解析时间戳
                time_line = lines[1]
                time_pattern = (
                    r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> "
                    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
                )
                time_match = re.match(time_pattern, time_line)

                if not time_match:
                    continue

                # 计算时间戳（秒）
                start_h, start_m, start_s, start_ms = map(int, time_match.groups()[:4])
                end_h, end_m, end_s, end_ms = map(int, time_match.groups()[4:])

                start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000.0
                end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000.0

                # 合并文本行
                if len(lines) > 2:
                    text = "\n".join(lines[2:])
                    text = self._clean_subtitle_text(text)
                else:
                    text = ""

                # 创建字幕条目
                entry = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": text,
                    "duration": round(end_time - start_time, 3),
                }
                entries.append(entry)

        except Exception as e:
            self.logger.error(f"SRT文件解析失败: {str(e)}")

        return entries

    def _clean_subtitle_text(self, text: str) -> str:
        """基本清理字幕文本"""
        import re

        if not text:
            return ""

        # 移除HTML标签
        text = re.sub(r"<[^>]*>", "", text)

        # 移除特殊字符（保留中文、英文、数字、基本标点）
        text = re.sub(
            r"[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef.,!?;:\'\"()-]",
            "",
            text,
        )

        # 移除多余的空白字符
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def _load_audio(self, audio_path: str):
        """加载音频文件，使用内存缓存优化"""
        import hashlib
        import os

        try:
            # 创建文件缓存键 - 基于文件路径和修改时间
            file_stat = os.stat(audio_path)
            cache_key = f"{audio_path}_{file_stat.st_mtime}_{file_stat.st_size}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]

            # 检查缓存
            if self.enable_aggressive_caching and cache_hash in self._audio_cache:
                self.logger.debug(f"从缓存加载音频: {os.path.basename(audio_path)}")
                return self._audio_cache[cache_hash]

            # 加载音频文件
            try:
                # 优先使用librosa
                audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=True)
            except Exception:
                # 备选方案：使用soundfile
                audio_data, sample_rate = sf.read(audio_path)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)  # 转换为单声道

            # 计算音频数据大小并决定是否缓存
            if self.enable_aggressive_caching:
                # 估算内存使用 (音频数据 + 元数据)
                audio_size_gb = audio_data.nbytes / (1024 ** 3)

                if (
                        audio_size_gb < 0.5
                        and self._cache_usage + audio_size_gb < self.cache_size_limit
                ):
                    # 只缓存小于500MB的音频文件，避免内存溢出
                    if len(self._audio_cache) >= self.audio_cache_max_size:
                        # 清理最老的缓存项（简单的FIFO策略）
                        oldest_key = next(iter(self._audio_cache))
                        old_data = self._audio_cache.pop(oldest_key)
                        if isinstance(old_data, tuple) and len(old_data) == 2:
                            old_audio_size = old_data[0].nbytes / (1024 ** 3)
                            self._cache_usage -= old_audio_size

                    # 添加到缓存
                    self._audio_cache[cache_hash] = (audio_data.copy(), sample_rate)
                    self._cache_usage += audio_size_gb
                    self.logger.debug(
                        f"音频已缓存: {os.path.basename(audio_path)} "
                        f"({audio_size_gb:.2f}GB, 总缓存: {self._cache_usage:.2f}GB)"
                    )

            return audio_data, sample_rate

        except Exception as e:
            raise Exception(f"无法加载音频文件: {str(e)}")

    def _extract_audio_segment(
            self,
            audio_data: np.ndarray,
            sample_rate: int,
            start_time: float,
            end_time: float,
    ) -> np.ndarray:
        """提取音频片段"""
        try:
            # 计算样本索引
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # 边界检查
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)

            if start_sample >= end_sample:
                return np.array([])

            # 提取片段
            segment = audio_data[start_sample:end_sample]

            # 音频预处理：去除开头和结尾的静音
            segment = self._trim_silence(segment)

            return segment

        except Exception as e:
            self.logger.warning(f"提取音频片段失败: {str(e)}")
            return np.array([])

    def _trim_silence(
            self, audio_data: np.ndarray, threshold: float = 0.01
    ) -> np.ndarray:
        """去除音频开头和结尾的静音"""
        if len(audio_data) == 0:
            return audio_data

        # 计算音频的绝对值
        abs_audio = np.abs(audio_data)

        # 找到非静音部分的开始和结束
        non_silent = abs_audio > threshold

        if not np.any(non_silent):
            return audio_data

        # 找到第一个和最后一个非静音样本
        non_silent_indices = np.where(non_silent)[0]
        start_idx = non_silent_indices[0]
        end_idx = non_silent_indices[-1] + 1

        return audio_data[start_idx:end_idx]

    def _calculate_audio_quality(
            self, audio_data: np.ndarray, sample_rate: int
    ) -> float:
        """计算音频质量分数"""
        if len(audio_data) == 0:
            return 0.0

        try:
            # 计算多个音频质量指标
            duration = len(audio_data) / sample_rate
            if 0.5 <= duration <= 3.0:
                duration_score = 1.0
            elif duration < 0.5:
                duration_score = duration / 0.5
            else:
                duration_score = max(0.1, 3.0 / duration)

            # 音量评分
            rms = np.sqrt(np.mean(audio_data ** 2))
            volume_score = min(1.0, rms / 0.1)

            # 动态范围评分
            dynamic_range = np.max(np.abs(audio_data)) - np.mean(np.abs(audio_data))
            range_score = min(1.0, dynamic_range / 0.5)

            # 静音比例评分
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(audio_data) < silence_threshold) / len(
                audio_data
            )
            silence_score = max(0.1, 1.0 - silence_ratio)

            # 综合评分
            total_score = (
                    duration_score * 0.3
                    + volume_score * 0.3
                    + range_score * 0.2
                    + silence_score * 0.2
            )

            return min(1.0, total_score)

        except Exception:
            return 0.5

    def _is_silence_segment(
            self,
            audio_data: np.ndarray,
            silence_threshold: float = 0.01,
            silence_ratio_threshold: float = 0.8,
    ) -> bool:
        """检测音频片段是否为静音"""
        if len(audio_data) == 0:
            return True

        try:
            # 计算RMS能量
            rms = np.sqrt(np.mean(audio_data ** 2))

            # 计算静音比例
            silence_ratio = np.sum(np.abs(audio_data) < silence_threshold) / len(
                audio_data
            )

            # 如果RMS过低或静音比例过高，认为是静音片段
            is_silence = rms < 0.005 or silence_ratio > silence_ratio_threshold

            return is_silence

        except Exception as e:
            self.logger.warning(f"静音检测失败: {str(e)}")
            return False

    def _replace_silence_segments(self, reference_segments: list) -> list:
        """替换静音片段为质量最高的片段"""
        if not reference_segments:
            return reference_segments

        try:
            # 找出非静音且质量最高的片段作为替换源
            non_silence_segments = []
            silence_segments = []

            for segment in reference_segments:
                if segment.get("is_silence", False):
                    silence_segments.append(segment)
                else:
                    non_silence_segments.append(segment)

            if not non_silence_segments:
                self.logger.warning("没有找到非静音片段，无法替换静音片段")
                return reference_segments

            if not silence_segments:
                return reference_segments

            # 找到质量最高的非静音片段
            best_segment = max(
                non_silence_segments, key=lambda x: x.get("quality_score", 0)
            )

            self.logger.info(
                f"找到最佳替换片段: index={best_segment['index']}, "
                f"quality={best_segment['quality_score']:.3f}"
            )

            # 替换静音片段的音频文件
            import shutil

            best_audio_path = best_segment["file_path"]

            replaced_count = 0
            for silence_segment in silence_segments:
                try:
                    silence_audio_path = silence_segment["file_path"]
                    if silence_audio_path and Path(best_audio_path).exists():
                        shutil.copy2(best_audio_path, silence_audio_path)

                        # 更新片段信息
                        silence_segment["quality_score"] = best_segment["quality_score"]
                        silence_segment["audio_length"] = best_segment["audio_length"]
                        silence_segment["is_silence"] = False
                        silence_segment["replaced_with"] = best_segment["index"]

                        replaced_count += 1

                except Exception as e:
                    self.logger.error(
                        f"替换静音片段 {silence_segment['index']} 失败: {str(e)}"
                    )
                    continue

            self.logger.info(
                f"静音片段替换完成: {replaced_count}/{len(silence_segments)} 个片段成功替换"
            )
            return reference_segments

        except Exception as e:
            self.logger.error(f"静音片段替换处理失败: {str(e)}")
            return reference_segments

    def _save_results_to_json(
            self, results: Dict[str, Any], output_dir: str, audio_name: str
    ) -> str:
        """保存结果到JSON文件"""
        try:
            # 创建JSON文件路径
            json_filename = f"{audio_name}_reference_audio_results.json"
            json_path = Path(output_dir) / json_filename

            # 处理NumPy类型的序列化
            def numpy_serializer(obj):
                if hasattr(obj, "item"):
                    return obj.item()
                elif hasattr(obj, "tolist"):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    raise TypeError(
                        f"Object of type {type(obj)} is not JSON serializable"
                    )

            # 添加保存时间戳
            results_copy = results.copy()
            results_copy["saved_at"] = datetime.datetime.now().isoformat()
            results_copy["file_version"] = "1.0"

            # 保存JSON文件
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    results_copy,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=numpy_serializer,
                )

            self.logger.info(f"结果已保存到JSON文件: {json_path}")
            return str(json_path)

        except Exception as e:
            self.logger.error(f"保存JSON文件失败: {str(e)}")
            return ""

    def _normalize_audio_volume(
            self, audio: np.ndarray, target_rms: float
    ) -> np.ndarray:
        """将音频归一化到固定的舒适听感音量级别"""
        try:
            # 计算当前RMS
            current_rms = librosa.feature.rms(y=audio)[0]
            current_rms_mean = np.mean(current_rms[current_rms > 0])

            if current_rms_mean == 0 or np.isnan(current_rms_mean):
                return audio

            # 计算增益
            gain = target_rms / current_rms_mean

            # 限制增益范围
            gain = np.clip(gain, 0.1, 10.0)

            # 应用增益
            normalized_audio = audio * gain

            # 软限制，防止削波
            normalized_audio = np.tanh(normalized_audio * 0.95) * 0.95

            return normalized_audio

        except Exception as e:
            self.logger.warning(f"音量归一化失败: {str(e)}")
            return audio

    def _get_video_duration(self, video_path: str) -> float:
        """获取视频时长"""
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe["streams"][0]["duration"])
            return duration
        except Exception as e:
            self.logger.warning(f"获取视频时长失败，使用默认值: {str(e)}")
            return 0.0

    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception:
            try:
                info = sf.info(audio_path)
                return info.duration
            except Exception as e:
                self.logger.warning(f"获取音频时长失败，使用默认值: {str(e)}")
                return 0.0

    def _merge_audio_video_with_duration_adjustment(
            self,
            video_path: str,
            audio_path: str,
            output_path: str,
            video_duration: float,
            audio_duration: float,
            final_duration: float,
    ):
        """使用ffmpeg合并音视频，带时长调整和GPU硬件加速"""
        try:
            # RTX 4060Ti GPU硬件加速配置
            input_video_args = {}
            input_audio_args = {}
            output_args = {}

            if self.use_gpu:
                # 使用GPU硬件解码（移除output_format以提高兼容性）
                input_video_args["hwaccel"] = "cuda"

                # 使用GPU硬件编码
                output_args.update(
                    {
                        "vcodec": "h264_nvenc",
                        "acodec": "aac",
                        "preset": "p1",  # NVENC fastest preset
                        "tune": "hq",  # High quality
                        "crf": "20",  # 提高质量，GPU编码可以承受
                        "audio_bitrate": "192k",  # 提高音频质量
                        "maxrate": "10M",  # 利用16GB显存的优势
                        "bufsize": "20M",
                        "t": final_duration,
                    }
                )

                self.logger.debug("使用GPU硬件加速合并音视频")
            else:
                # CPU编码回退配置
                output_args.update(
                    {
                        "vcodec": "libx264",
                        "acodec": "aac",
                        "preset": "fast",
                        "crf": 23,
                        "audio_bitrate": "128k",
                        "t": final_duration,
                    }
                )

            input_video = ffmpeg.input(video_path, **input_video_args)
            input_audio = ffmpeg.input(audio_path, **input_audio_args)

            # 处理视频流
            if audio_duration > video_duration:
                video_stream = input_video["v"].filter(
                    "tpad", stop_mode="clone", stop_duration=final_duration
                )
            else:
                video_stream = input_video["v"]

            # 处理音频流
            if video_duration > audio_duration:
                silence_duration = final_duration - audio_duration
                audio_stream = input_audio["a"].filter("apad", pad_dur=silence_duration)
            else:
                audio_stream = input_audio["a"]

            # 合并音视频流
            output = ffmpeg.output(
                video_stream, audio_stream, output_path, **output_args
            ).overwrite_output()

            # 记录音视频合并FFmpeg命令到日志
            merge_description = f"音视频合并 (视频:{video_duration:.2f}s, 音频:{audio_duration:.2f}s)"

            if self.use_gpu:
                merge_description += " - GPU加速"
            else:
                merge_description += " - CPU处理"

            self._log_ffmpeg_command(output, merge_description)

            try:
                # 执行GPU加速合并
                ffmpeg.run(output, overwrite_output=True, quiet=True)
                self.logger.debug(f"GPU加速合并完成: {output_path}")
            except Exception as e:
                if self.use_gpu:
                    # GPU合并失败，回退到CPU
                    self.logger.warning(f"GPU合并失败，回退到CPU: {str(e)}")

                    input_video_cpu = ffmpeg.input(video_path)
                    input_audio_cpu = ffmpeg.input(audio_path)

                    # 处理视频流
                    if audio_duration > video_duration:
                        video_stream_cpu = input_video_cpu["v"].filter(
                            "tpad", stop_mode="clone", stop_duration=final_duration
                        )
                    else:
                        video_stream_cpu = input_video_cpu["v"]

                    # 处理音频流
                    if video_duration > audio_duration:
                        silence_duration = final_duration - audio_duration
                        audio_stream_cpu = input_audio_cpu["a"].filter(
                            "apad", pad_dur=silence_duration
                        )
                    else:
                        audio_stream_cpu = input_audio_cpu["a"]

                    output_cpu = ffmpeg.output(
                        video_stream_cpu,
                        audio_stream_cpu,
                        output_path,
                        vcodec="libx264",
                        acodec="aac",
                        preset="fast",
                        crf=23,
                        audio_bitrate="128k",
                        t=final_duration,
                    )

                    # 记录CPU回退合并FFmpeg命令到日志
                    self._log_ffmpeg_command(output_cpu, "CPU回退音视频合并")

                    # 执行CPU回退合并FFmpeg命令
                    ffmpeg.run(output_cpu, overwrite_output=True, quiet=True)
                    self.logger.debug(f"CPU回退合并完成: {output_path}")
                else:
                    raise

        except Exception as e:
            raise Exception(f"ffmpeg时长调整合并失败: {str(e)}")


# 单例实例
_media_processor_instance = None


def get_media_processor_core() -> MediaProcessorCore:
    """获取 MediaProcessorCore 单例实例"""
    global _media_processor_instance
    if _media_processor_instance is None:
        _media_processor_instance = MediaProcessorCore()
    return _media_processor_instance


def separate_media_core(
        video_path: str,
        output_dir: Optional[str] = None,
        enable_vocal_separation: bool = False,
        audio_quality_level: int = 0,
        video_quality_level: int = 0,
) -> Dict[str, Any]:
    """便捷函数：分离视频中的人声、无声视频和背景音乐"""
    return get_media_processor_core().separate_media(
        video_path, output_dir, enable_vocal_separation, audio_quality_level, video_quality_level
    )


def generate_reference_audio_core(
        audio_path: str,
        subtitle_path: str,
        output_dir: Optional[str] = None,
        normalize_volume: bool = True,
) -> Dict[str, Any]:
    """便捷函数：根据字幕分隔音频生成参考音频"""
    return get_media_processor_core().generate_reference_audio(
        audio_path, subtitle_path, output_dir, normalize_volume
    )


def merge_audio_video_core(
        video_path: str, audio_path: str, output_path: Optional[str] = None
) -> Dict[str, Any]:
    """便捷函数：合并音频和视频文件"""
    return get_media_processor_core().merge_audio_video(
        video_path, audio_path, output_path
    )
