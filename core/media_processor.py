"""
简化的媒体处理类
专注于核心功能：根据视频路径分离人声、无声视频和背景音乐
"""

import datetime
import json
import logging
import os
import re
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
from numpy import floating


class MediaProcessor:
    """媒体处理器，专注于音视频分离"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.use_gpu = torch.cuda.is_available()

        # 打印GPU状态
        if self.use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"使用GPU加速: {gpu_name}")
        else:
            self.logger.info("使用CPU处理")

    def separate_media(
            self, video_path: str, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        根据视频路径分离人声、无声视频和背景音乐

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录，默认为视频文件所在目录

        Returns:
            包含分离结果的字典：
            {
                'success': bool,
                'silent_video_path': str,  # 无声视频路径
                'vocal_audio_path': str,   # 人声音频路径
                'background_audio_path': str,  # 背景音乐路径
                'error': str  # 错误信息（如果有）
            }
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

            # 定义输出文件路径
            silent_video_path = output_dir / f"{video_name}_silent.mp4"
            raw_audio_path = output_dir / f"{video_name}_raw_audio.wav"
            vocal_audio_path = output_dir / f"{video_name}_vocal.wav"
            background_audio_path = output_dir / f"{video_name}_background.wav"

            self.logger.info(f"开始处理视频: {video_path}")

            # 1. 提取原始音频
            self._extract_audio(video_path, str(raw_audio_path))

            # 2. 创建无声视频
            self._create_silent_video(video_path, str(silent_video_path))

            # 3. 分离音频（人声和背景音乐）
            self._separate_audio(
                str(raw_audio_path),
                str(vocal_audio_path),
                str(background_audio_path),
            )

            # 4. 清理临时文件
            if raw_audio_path.exists():
                raw_audio_path.unlink()

            self.logger.info("媒体分离完成")

            return {
                "success": True,
                "silent_video_path": str(silent_video_path),
                "vocal_audio_path": str(vocal_audio_path),
                "background_audio_path": str(background_audio_path),
            }

        except Exception as e:
            self.logger.error(f"媒体分离失败: {str(e)}")
            return {"success": False, "error": str(e)}

    def _extract_audio(self, video_path: str, audio_path: str):
        """从视频中提取音频"""
        try:
            (
                ffmpeg.input(video_path)
                .output(audio_path, acodec="pcm_s16le", ar=44100, ac=2)
                .overwrite_output()
                .run(quiet=True)
            )
            self.logger.debug(f"音频提取完成: {audio_path}")
        except Exception as e:
            raise Exception(f"音频提取失败: {str(e)}")

    def _create_silent_video(self, video_path: str, silent_video_path: str):
        """创建无声视频"""
        try:
            (
                ffmpeg.input(video_path)
                .video.output(
                    silent_video_path, vcodec="libx264", preset="fast"
                )
                .overwrite_output()
                .run(quiet=True)
            )
            self.logger.debug(f"无声视频创建完成: {silent_video_path}")
        except Exception as e:
            raise Exception(f"无声视频创建失败: {str(e)}")

    def _separate_audio(
            self, audio_path: str, vocal_path: str, background_path: str
    ):
        """使用Demucs分离人声和背景音乐"""
        try:
            # 加载模型
            if self.model is None:
                self.logger.info("加载Demucs模型...")
                self.model = pretrained.get_model("htdemucs")
                if self.use_gpu:
                    self.model = self.model.cuda()
                self.model.eval()

            # 获取模型参数
            model_sample_rate = 44100  # htdemucs默认采样率
            model_channels = 2  # 立体声

            # 加载音频
            waveform, sample_rate = torchaudio.load(audio_path)

            # 转换音频格式以匹配模型要求
            waveform = convert_audio(
                waveform, sample_rate, model_sample_rate, model_channels
            )

            if self.use_gpu:
                waveform = waveform.cuda()

            # 应用模型进行源分离
            with torch.no_grad():
                sources = apply_model(self.model, waveform.unsqueeze(0))[0]

            # 提取人声和背景音乐
            # Demucs输出: [drums, bass, other, vocals]
            vocals = sources[3].cpu()  # 人声
            # 背景音乐 = 鼓 + 贝斯 + 其他乐器
            background = (sources[0] + sources[1] + sources[2]).cpu()

            # 保存分离的音频
            sf.write(vocal_path, vocals.T.numpy(), model_sample_rate)
            sf.write(background_path, background.T.numpy(), model_sample_rate)

            self.logger.debug(
                f"音频分离完成: 人声={vocal_path}, 背景={background_path}"
            )

        except Exception as e:
            raise Exception(f"音频分离失败: {str(e)}")
        finally:
            # 清理GPU内存
            if self.use_gpu:
                torch.cuda.empty_cache()

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
            audio_path: 音频文件路径（通常是人声音频）
            subtitle_path: 字幕文件路径（支持SRT格式）
            output_dir: 输出目录，默认为音频文件所在目录
            normalize_volume: 是否进行音量一致性增强，默认True
                            True - 将所有音频片段归一化到舒适听感音量(-16.5dB)
                            False - 保持原始音量，不进行音量调整

        Returns:
            包含参考音频生成结果的字典：
            {
                'success': bool,
                'reference_audio_segments': List[Dict],  # 参考音频片段信息
                'output_dir': str,  # 输出目录
                'total_segments': int,  # 总片段数
                'error': str  # 错误信息（如果有）
            }
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

            # 检查缓存
            self.logger.info("检查是否存在缓存结果...")
            cached_results = self._load_cached_results(
                audio_path, subtitle_path, str(output_dir)
            )
            if cached_results is not None:
                self.logger.info("使用缓存结果，跳过重新处理")
                return cached_results

            self.logger.info("未找到有效缓存，开始重新处理...")

            # 解析字幕文件
            subtitle_entries = self._parse_subtitle_file(subtitle_path)
            if not subtitle_entries:
                return {"success": False, "error": "字幕文件解析失败或为空"}

            audio_name = Path(audio_path).stem

            self.logger.info(f"开始生成参考音频: {audio_path}")
            self.logger.info(
                f"使用字幕文件: {subtitle_path} ({len(subtitle_entries)}个条目)"
            )

            # 加载音频文件
            audio_data, sample_rate = self._load_audio(audio_path)

            # 生成参考音频片段
            reference_segments = []
            successful_segments = 0

            # 用于音量分析的音频片段收集
            audio_segments_for_analysis = []

            # 第一遍：提取所有音频片段用于检查
            for i, entry in enumerate(subtitle_entries):
                try:
                    start_time = entry["start_time"]
                    end_time = entry["end_time"]
                    text = entry["text"]

                    # 验证时间戳
                    if start_time >= end_time or start_time < 0:
                        continue

                    # 提取音频片段
                    segment_audio = self._extract_audio_segment(
                        audio_data, sample_rate, start_time, end_time
                    )

                    # 静音检测
                    is_silence = self._is_silence_segment(
                        segment_audio, sample_rate
                    )

                    if not is_silence:  # 只收集非静音片段用于统计
                        audio_segments_for_analysis.append(segment_audio)

                except Exception:
                    continue

            # 设置音量处理策略
            if normalize_volume:
                # 设置固定的舒适听感音量（RMS约0.15，相当于-16.5dB，人声对话的舒适音量）
                target_rms = 0.15
                self.logger.info(
                    f"启用音量一致性增强，目标RMS音量: {target_rms:.6f}（舒适听感标准）"
                )
            else:
                target_rms = None
                self.logger.info("音量一致性增强已禁用，保持原始音量")

            # 第二遍：处理和保存音频片段
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
                    is_silence = self._is_silence_segment(
                        segment_audio, sample_rate
                    )

                    # 质量检查
                    quality_score = self._calculate_audio_quality(
                        segment_audio, sample_rate
                    )

                    # 音量一致性增强（可选）
                    if normalize_volume and target_rms and not is_silence:
                        segment_audio = self._normalize_audio_volume(
                            segment_audio, target_rms
                        )
                        self.logger.debug(
                            f"片段 {i + 1} 音量已归一化到舒适听感标准"
                        )
                    elif not normalize_volume:
                        self.logger.debug(f"片段 {i + 1} 保持原始音量")

                    # 保存音频片段
                    segment_filename = f"{audio_name}_ref_{i + 1:04d}.wav"
                    segment_path = output_dir / segment_filename

                    sf.write(str(segment_path), segment_audio, sample_rate)

                    # 记录片段信息
                    segment_info = {
                        "index": i + 1,
                        "text": text,  # 可能为空字符串
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": round(
                            end_time - start_time, 3
                        ),  # 保留3位小数精度
                        "file_path": str(segment_path),
                        "quality_score": quality_score,
                        "sample_rate": sample_rate,
                        "audio_length": len(segment_audio),
                        "is_silence": is_silence,  # 添加静音标记
                    }
                    reference_segments.append(segment_info)
                    successful_segments += 1

                    text_preview = text[:20] if text else "[空字幕]"
                    duration = end_time - start_time
                    silence_info = "[静音]" if is_silence else ""
                    self.logger.debug(
                        f"生成参考音频片段 {i + 1}: {text_preview}... "
                        f"({duration:.2f}s) {silence_info}"
                    )

                except Exception as e:
                    self.logger.warning(f"处理字幕片段 {i + 1} 失败: {str(e)}")
                    continue

            self.logger.info(
                f"参考音频生成完成: {successful_segments}/"
                f"{len(subtitle_entries)} 个片段成功"
            )

            # 静音片段检测和替换
            silence_count = sum(
                1 for seg in reference_segments if seg.get("is_silence", False)
            )
            if silence_count > 0:
                self.logger.info(
                    f"检测到 {silence_count} 个静音片段，开始替换处理..."
                )
                reference_segments = self._replace_silence_segments(
                    reference_segments
                )
            else:
                self.logger.info("未检测到静音片段")

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
                    "description": (
                        "舒适听感标准" if normalize_volume else "保持原始音量"
                    ),
                },
                "silence_detection": {
                    "silence_segments_found": silence_count,
                    "segments_replaced": sum(
                        1
                        for seg in reference_segments
                        if seg.get("replaced_with")
                    ),
                    "best_quality_segment": (
                        max(
                            reference_segments,
                            key=lambda x: x.get("quality_score", 0),
                        )["index"]
                        if reference_segments
                        else None
                    ),
                },
                "subtitle_file": subtitle_path,
                "audio_info": {
                    "sample_rate": sample_rate,
                    "duration": len(audio_data) / sample_rate,
                    "channels": (
                        1
                        if len(audio_data.shape) == 1
                        else audio_data.shape[1]
                    ),
                },
            }

            # 保存结果到JSON文件
            audio_name = Path(audio_path).stem
            json_file_path = self._save_results_to_json(
                results, str(output_dir), audio_name
            )
            if json_file_path:
                results["results_json_file"] = json_file_path

            return results

        except Exception as e:
            self.logger.error(f"参考音频生成失败: {str(e)}")
            return {"success": False, "error": str(e)}

    def _parse_subtitle_file(self, subtitle_path: str) -> List[Dict[str, Any]]:
        """解析字幕文件（目前支持SRT格式）"""
        try:
            subtitle_format = Path(subtitle_path).suffix.lower()

            if subtitle_format == ".srt":
                return self._parse_srt_file(subtitle_path)
            else:
                self.logger.warning(
                    f"暂不支持的字幕格式: {subtitle_format}，请使用SRT格式"
                )
                return []

        except Exception as e:
            self.logger.error(f"字幕文件解析失败: {str(e)}")
            return []

    def _parse_srt_file(self, srt_path: str) -> List[Dict[str, Any]]:
        """解析SRT字幕文件"""
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
                if len(lines) < 2:  # 至少需要序号和时间戳
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
                start_h, start_m, start_s, start_ms = map(
                    int, time_match.groups()[:4]
                )
                end_h, end_m, end_s, end_ms = map(int, time_match.groups()[4:])

                start_time = (
                        start_h * 3600 + start_m * 60 + start_s + start_ms / 1000.0
                )
                end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000.0

                # 合并文本行并基本清理（如果有文本行的话）
                if len(lines) > 2:
                    text = "\n".join(lines[2:])
                    text = self._clean_subtitle_text(text)
                else:
                    text = ""  # 空字幕

                # 创建字幕条目（包括空文本条目，保持索引连续性）
                entry = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": text,  # 可能为空字符串
                    "duration": round(
                        end_time - start_time, 3
                    ),  # 保留3位小数精度
                }
                entries.append(entry)

        except Exception as e:
            self.logger.error(f"SRT文件解析失败: {str(e)}")

        return entries

    def _clean_subtitle_text(self, text: str) -> str:
        """基本清理字幕文本"""
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

    def _load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        """加载音频文件"""
        try:
            # 优先使用librosa
            audio_data, sample_rate = librosa.load(
                audio_path, sr=None, mono=True
            )
            return audio_data, sample_rate
        except Exception:
            # 备选方案：使用soundfile
            try:
                audio_data, sample_rate = sf.read(audio_path)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)  # 转换为单声道
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
            return audio_data  # 如果全是静音，返回原始数据

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

            # 1. 音频长度评分（0.5-3秒为最佳）
            duration = len(audio_data) / sample_rate
            if 0.5 <= duration <= 3.0:
                duration_score = 1.0
            elif duration < 0.5:
                duration_score = duration / 0.5
            else:
                duration_score = max(0.1, 3.0 / duration)

            # 2. 音量评分
            rms = np.sqrt(np.mean(audio_data ** 2))
            volume_score = min(1.0, rms / 0.1)  # 假设0.1为理想RMS

            # 3. 动态范围评分
            dynamic_range = np.max(np.abs(audio_data)) - np.mean(
                np.abs(audio_data)
            )
            range_score = min(1.0, dynamic_range / 0.5)

            # 4. 静音比例评分（静音越少越好）
            silence_threshold = 0.01
            silence_ratio = np.sum(
                np.abs(audio_data) < silence_threshold
            ) / len(audio_data)
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
            return 0.5  # 默认中等质量

    def _is_silence_segment(
            self,
            audio_data: np.ndarray,
            sample_rate: int,
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
            silence_ratio = np.sum(
                np.abs(audio_data) < silence_threshold
            ) / len(audio_data)

            # 如果RMS过低或静音比例过高，认为是静音片段
            is_silence = rms < 0.005 or silence_ratio > silence_ratio_threshold

            if is_silence:
                self.logger.debug(
                    f"检测到静音片段: RMS={rms:.6f}, 静音比例={silence_ratio:.2f}"
                )

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
                self.logger.info("没有检测到静音片段")
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
                    # 复制最佳片段的音频文件来替换静音片段
                    silence_audio_path = silence_segment["file_path"]
                    if silence_audio_path and Path(best_audio_path).exists():
                        shutil.copy2(best_audio_path, silence_audio_path)

                        # 更新片段信息
                        silence_segment["quality_score"] = best_segment[
                            "quality_score"
                        ]
                        silence_segment["audio_length"] = best_segment[
                            "audio_length"
                        ]
                        silence_segment["is_silence"] = False
                        silence_segment["replaced_with"] = best_segment[
                            "index"
                        ]

                        replaced_count += 1
                        self.logger.info(
                            f"替换静音片段 {silence_segment['index']} "
                            f"为片段 {best_segment['index']}"
                        )

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
                if hasattr(obj, "item"):  # NumPy标量
                    return obj.item()
                elif hasattr(obj, "tolist"):  # NumPy数组
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

    def _load_cached_results(
            self, audio_path: str, subtitle_path: str, output_dir: str
    ) -> Optional[Dict[str, Any]]:
        """加载缓存的结果"""
        try:
            # 构建JSON文件路径
            audio_name = Path(audio_path).stem
            json_filename = f"{audio_name}_reference_audio_results.json"
            json_path = Path(output_dir) / json_filename

            if not json_path.exists():
                self.logger.debug(f"缓存文件不存在: {json_path}")
                return None

            # 读取JSON文件
            with open(json_path, "r", encoding="utf-8") as f:
                cached_results = json.load(f)

            self.logger.info(f"找到缓存文件: {json_path}")
            return cached_results

        except Exception as e:
            self.logger.warning(f"加载缓存文件失败: {str(e)}")
            return None

    @staticmethod
    def _time_to_seconds(time_obj) -> float:
        """时间对象转换为秒数"""
        if isinstance(time_obj, datetime.time):
            return (
                    time_obj.hour * 3600
                    + time_obj.minute * 60
                    + time_obj.second
                    + time_obj.microsecond / 1000000
            )
        elif isinstance(time_obj, (int, float)):
            return float(time_obj)
        else:
            return 0.0

    def _analyze_target_volume(
            self, audio_segments: List[np.ndarray]
    ) -> floating[Any] | None:
        """
        分析音频片段，确定目标音量级别
        使用RMS (Root Mean Square) 方法分析音量

        Args:
            audio_segments: 音频片段列表

        Returns:
            目标RMS值
        """
        try:
            if not audio_segments:
                return None

            rms_values = []

            for segment in audio_segments:
                # 计算RMS值
                rms = librosa.feature.rms(y=segment)[0]
                # 取非零部分的均值，避免静音影响
                non_zero_rms = rms[rms > 0]
                if len(non_zero_rms) > 0:
                    rms_values.append(np.mean(non_zero_rms))

            if not rms_values:
                return None

            # 使用75%分位数作为目标音量，既保持音量充足又避免过度放大
            target_rms = np.percentile(rms_values, 75)

            # 设置合理的音量范围 (0.01 到 0.3)
            target_rms = np.clip(target_rms, 0.01, 0.3)

            min_rms = np.min(rms_values)
            max_rms = np.max(rms_values)
            self.logger.debug(
                f"音量分析: RMS范围 {min_rms:.6f} - {max_rms:.6f}, "
                f"目标: {target_rms:.6f}"
            )

            return target_rms

        except Exception as e:
            self.logger.warning(f"音量分析失败: {str(e)}")
            return None

    def _normalize_audio_volume(
            self, audio: np.ndarray, target_rms: float
    ) -> np.ndarray:
        """
        将音频归一化到固定的舒适听感音量级别

        固定音量标准：
        - RMS = 0.15 (约-16.5dB)
        - 相当于正常人声对话的舒适音量
        - 适合长时间聆听，不会造成听觉疲劳

        Args:
            audio: 音频数据
            target_rms: 目标RMS值（0.15，固定舒适听感标准）

        Returns:
            归一化后的音频数据
        """
        try:
            # 计算当前RMS
            current_rms = librosa.feature.rms(y=audio)[0]
            current_rms_mean = np.mean(current_rms[current_rms > 0])

            if current_rms_mean == 0 or np.isnan(current_rms_mean):
                return audio  # 静音片段直接返回

            # 计算增益
            gain = target_rms / current_rms_mean

            # 限制增益范围，避免过度放大或衰减
            gain = np.clip(gain, 0.1, 10.0)

            # 应用增益
            normalized_audio = audio * gain

            # 软限制，防止削波
            normalized_audio = np.tanh(normalized_audio * 0.95) * 0.95

            self.logger.debug(
                f"音量归一化: 当前RMS {current_rms_mean:.6f} -> "
                f"目标RMS {target_rms:.6f}, 增益: {gain:.2f}"
            )

            return normalized_audio

        except Exception as e:
            self.logger.warning(f"音量归一化失败: {str(e)}")
            return audio

    def merge_audio_video(
            self,
            video_path: str,
            audio_path: str,
            output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        合并音频和视频文件，自动处理时长差异

        规则：
        - 如果音频时长 > 视频时长：视频补最后一帧延长到音频时长
        - 如果视频时长 > 音频时长：音频补静音延长到视频时长
        - 最终输出时长为较长者的时长

        Args:
            video_path: 视频文件路径（可以是无声视频）
            audio_path: 音频文件路径（包含新的音轨）
            output_path: 输出文件路径，默认为视频文件目录下的 {video_name}_merged.mp4

        Returns:
            包含合并结果的字典：
            {
                'success': bool,
                'output_path': str,  # 合并后的视频文件路径
                'video_duration': float,  # 原视频时长
                'audio_duration': float,  # 原音频时长
                'final_duration': float,  # 最终输出时长
                'adjustments': {
                    'video_extended': bool,  # 是否延长了视频
                    'audio_extended': bool,  # 是否延长了音频
                },
                'error': str  # 错误信息（如果有）
            }
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
                video_parent_dir = Path(video_path).parent.parent  # 上级目录
                output_dir = video_parent_dir / "final_video"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{video_name}_merged.mp4"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(
                f"开始合并音视频: 视频={video_path}, 音频={audio_path}"
            )

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

            if video_extended:
                self.logger.info(
                    f"音频较长，将视频延长到 {final_duration:.2f}s（补最后一帧）"
                )
            elif audio_extended:
                self.logger.info(
                    f"视频较长，将音频延长到 {final_duration:.2f}s（补静音）"
                )
            else:
                self.logger.info("音视频时长一致，无需调整")

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

    def _merge_audio_video_ffmpeg(
            self, video_path: str, audio_path: str, output_path: str
    ):
        """使用ffmpeg合并音频和视频"""
        try:
            # 使用ffmpeg-python进行合并
            input_video = ffmpeg.input(video_path)
            input_audio = ffmpeg.input(audio_path)

            # 合并音视频流
            output = ffmpeg.output(
                input_video["v"],  # 视频流
                input_audio["a"],  # 音频流
                output_path,
                vcodec="libx264",  # 视频编码器
                acodec="aac",  # 音频编码器
                preset="fast",  # 编码预设
                crf=23,  # 视频质量
                audio_bitrate="128k",  # 音频比特率
            )

            # 执行合并
            ffmpeg.run(output, overwrite_output=True, quiet=True)

            self.logger.debug(f"ffmpeg合并完成: {output_path}")

        except Exception as e:
            raise Exception(f"ffmpeg合并音视频失败: {str(e)}")

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
            # 优先使用 librosa
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception:
            try:
                # 备选方案：使用 soundfile
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
        """使用ffmpeg合并音视频，带时长调整"""
        try:
            input_video = ffmpeg.input(video_path)
            input_audio = ffmpeg.input(audio_path)

            # 处理视频流
            if audio_duration > video_duration:
                # 音频较长，需要延长视频（补最后一帧）
                video_stream = input_video["v"].filter(
                    "tpad", stop_mode="clone", stop_duration=final_duration
                )
                self.logger.debug(
                    f"视频延长：从 {video_duration:.2f}s 到 {final_duration:.2f}s"
                )
            else:
                # 视频不需要延长或等长
                video_stream = input_video["v"]

            # 处理音频流
            if video_duration > audio_duration:
                # 视频较长，需要延长音频（补静音）
                silence_duration = final_duration - audio_duration
                audio_stream = input_audio["a"].filter(
                    "apad", pad_dur=silence_duration
                )
                self.logger.debug(
                    f"音频延长：从 {audio_duration:.2f}s 到 {final_duration:.2f}s"
                )
            else:
                # 音频不需要延长或等长
                audio_stream = input_audio["a"]

            # 合并音视频流，确保最终时长
            output = ffmpeg.output(
                video_stream,
                audio_stream,
                output_path,
                vcodec="libx264",  # 视频编码器
                acodec="aac",  # 音频编码器
                preset="fast",  # 编码预设
                crf=23,  # 视频质量
                audio_bitrate="128k",  # 音频比特率
                t=final_duration,  # 限制输出时长
            )

            # 执行合并
            ffmpeg.run(output, overwrite_output=True, quiet=True)

            self.logger.debug(f"ffmpeg时长调整合并完成: {output_path}")

        except Exception as e:
            raise Exception(f"ffmpeg时长调整合并失败: {str(e)}")


# 单例实例
_media_processor_instance = None


def get_media_processor() -> MediaProcessor:
    """获取 MediaProcessor 单例实例"""
    global _media_processor_instance
    if _media_processor_instance is None:
        _media_processor_instance = MediaProcessor()
    return _media_processor_instance


# 便捷函数
def separate_media(
        video_path: str, output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数：分离视频中的人声、无声视频和背景音乐

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录，默认为视频文件所在目录

    Returns:
        包含分离结果的字典
    """
    return get_media_processor().separate_media(video_path, output_dir)


def generate_reference_audio(
        audio_path: str,
        subtitle_path: str,
        output_dir: Optional[str] = None,
        normalize_volume: bool = True,
) -> Dict[str, Any]:
    """
    便捷函数：根据字幕分隔音频生成参考音频

    Args:
        audio_path: 音频文件路径
        subtitle_path: 字幕文件路径
        output_dir: 输出目录，默认为音频文件所在目录
        normalize_volume: 是否进行音量一致性增强

    Returns:
        包含参考音频生成结果的字典
    """
    return get_media_processor().generate_reference_audio(
        audio_path, subtitle_path, output_dir, normalize_volume
    )


def merge_audio_video(
        video_path: str, audio_path: str, output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数：合并音频和视频文件

    Args:
        video_path: 视频文件路径
        audio_path: 音频文件路径
        output_path: 输出文件路径，默认为视频文件目录下的 {video_name}_merged.mp4

    Returns:
        包含合并结果的字典
    """
    return get_media_processor().merge_audio_video(
        video_path, audio_path, output_path
    )
