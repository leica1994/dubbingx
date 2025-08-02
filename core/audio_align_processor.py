"""
音频对齐处理器
根据TTS生成结果和SRT字幕文件生成合并后的音频文件
"""

import os
import logging
import json
import subprocess
import tempfile
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import librosa
from datetime import datetime

from core.subtitle.subtitle_entry import SubtitleEntry


@dataclass
class AudioSegment:
    """音频片段数据结构"""
    index: int
    file_path: str
    duration: float
    start_time: float
    end_time: float
    is_silence: bool = False
    text: str = ""


@dataclass
class VideoSegment:
    """视频片段数据结构"""
    index: int
    start_time: float
    end_time: float
    duration: float
    segment_type: str  # "subtitle", "gap", "prefix", "suffix"
    text: str = ""
    subtitle_index: Optional[int] = None
    original_duration: Optional[float] = None
    target_duration: Optional[float] = None
    speed_ratio: Optional[float] = None


class AudioAlignProcessor:
    """音频对齐处理器"""

    def __init__(self, sample_rate: int = 22050):
        """初始化音频对齐处理器"""
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate

    def align_audio_with_subtitles(self,
                                   tts_results_path: str,
                                   srt_path: str,
                                   output_path: Optional[str] = None) -> Dict[str, Any]:
        """根据TTS结果和SRT字幕生成对齐的音频文件"""
        try:
            self.logger.info(f"开始音频对齐处理: {tts_results_path}")

            # 跳过缓存检查（已移除缓存功能）
            self.logger.info("开始处理...")

            # 解析TTS结果
            tts_segments = self._parse_tts_results(tts_results_path)
            if not tts_segments:
                return {'success': False, 'error': 'TTS结果解析失败'}

            # 解析SRT字幕
            subtitle_entries = self._parse_srt_subtitles(srt_path)
            if not subtitle_entries:
                return {'success': False, 'error': 'SRT字幕解析失败'}

            # 验证数据一致性
            if not self._validate_data_consistency(tts_segments, subtitle_entries):
                return {'success': False, 'error': 'TTS结果与字幕数据不一致'}

            # 生成默认输出路径
            if output_path is None:
                output_path = self._generate_default_output_path(tts_results_path)

            # 生成音频片段列表
            audio_segments = self._generate_audio_segments(tts_segments, subtitle_entries, srt_path)

            # 计算实际的总时长（基于最后一个音频片段的结束时间）
            total_duration = max(segment.end_time for segment in audio_segments) if audio_segments else (
                subtitle_entries[-1].end_time_seconds() if subtitle_entries else 0)

            # 拼接音频
            merged_audio = self._concatenate_audio_segments(audio_segments, total_duration)

            # 保存合并后的音频
            self._save_merged_audio(merged_audio, output_path)

            # 构建详细的音频片段信息
            aligned_audio_segments = []
            for segment in audio_segments:
                # 获取音频文件信息
                file_info = self._get_audio_file_info(segment.file_path)

                segment_info = {
                    'index': segment.index,
                    'text': segment.text,
                    'start_time': round(segment.start_time, 3),
                    'end_time': round(segment.end_time, 3),
                    'duration': round(segment.duration, 3),
                    'file_path': segment.file_path,
                    'is_silence': segment.is_silence,
                    'sample_rate': file_info.get('sample_rate', self.sample_rate),
                    'audio_length': file_info.get('audio_length', 0),
                    'file_size': file_info.get('file_size', 0)
                }

                # 如果是TTS片段，添加额外信息
                if not segment.is_silence:
                    segment_info['segment_type'] = 'tts_audio'
                else:
                    segment_info['segment_type'] = 'silence_audio'
                    segment_info[
                        'silence_reason'] = 'gap_between_segments' if 'gap' in segment.text else 'missing_tts_segment'

                aligned_audio_segments.append(segment_info)

            # 构建结果
            result = {
                'success': True,
                'output_path': output_path,
                'total_duration': round(total_duration, 3),
                'subtitle_count': len(subtitle_entries),
                'audio_segments': len(audio_segments),
                'silence_segments': sum(1 for seg in audio_segments if seg.is_silence),
                'tts_segments': sum(1 for seg in audio_segments if not seg.is_silence),
                'sample_rate': self.sample_rate,
                'aligned_audio_segments': aligned_audio_segments,
                'processing_info': {
                    'tts_file': tts_results_path,
                    'srt_file': srt_path,
                    'processed_at': datetime.now().isoformat()
                }
            }

            # 保存结果到JSON文件
            json_file_path = self._save_results_to_json(result, tts_results_path)
            if json_file_path:
                result['results_json_file'] = json_file_path

            self.logger.info(f"音频对齐完成: {output_path}")
            return result

        except Exception as e:
            self.logger.error(f"音频对齐处理失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _get_audio_file_info(self, file_path: str) -> Dict[str, Any]:
        """获取音频文件的详细信息"""
        try:
            if not os.path.exists(file_path):
                return {
                    'sample_rate': self.sample_rate,
                    'audio_length': 0,
                    'file_size': 0,
                    'file_exists': False
                }

            # 获取文件大小
            file_size = os.path.getsize(file_path)

            # 获取音频信息
            try:
                audio_data, sr = sf.read(file_path)
                audio_length = len(audio_data)
                sample_rate = sr

                # 如果是立体声，计算实际样本数
                if len(audio_data.shape) > 1:
                    audio_length = audio_data.shape[0]

                return {
                    'sample_rate': sample_rate,
                    'audio_length': audio_length,
                    'file_size': file_size,
                    'file_exists': True,
                    'duration': round(audio_length / sample_rate, 3) if sample_rate > 0 else 0
                }
            except Exception:
                # 如果无法读取音频文件，返回基本信息
                return {
                    'sample_rate': self.sample_rate,
                    'audio_length': 0,
                    'file_size': file_size,
                    'file_exists': True
                }

        except Exception as e:
            self.logger.warning(f"获取音频文件信息失败: {file_path}, 错误: {str(e)}")
            return {
                'sample_rate': self.sample_rate,
                'audio_length': 0,
                'file_size': 0,
                'file_exists': False
            }

    def _parse_tts_results(self, json_path: str) -> List[AudioSegment]:
        """解析TTS生成结果"""
        try:
            if not os.path.exists(json_path):
                self.logger.error(f"TTS结果文件不存在: {json_path}")
                return []

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not data.get('success', False):
                self.logger.error("TTS结果显示处理失败")
                return []

            tts_segments = data.get('tts_audio_segments', [])
            parsed_segments = []

            for segment in tts_segments:
                try:
                    parsed_segment = AudioSegment(
                        index=segment['index'],
                        file_path=segment['tts_file'],
                        duration=segment['duration'],
                        start_time=0,
                        end_time=0,
                        is_silence=segment.get('is_empty', False),
                        text=segment.get('text', '')
                    )
                    parsed_segments.append(parsed_segment)
                except Exception as e:
                    self.logger.warning(f"解析TTS片段失败: {segment.get('index', 'unknown')}")
                    continue

            parsed_segments.sort(key=lambda x: x.index)
            return parsed_segments

        except Exception as e:
            self.logger.error(f"TTS结果解析失败: {str(e)}")
            return []

    def _parse_srt_subtitles(self, srt_path: str) -> List[SubtitleEntry]:
        """解析SRT字幕文件"""
        try:
            if not os.path.exists(srt_path):
                self.logger.error(f"SRT文件不存在: {srt_path}")
                return []

            with open(srt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            entries = []
            i = 0
            while i < len(lines):
                try:
                    line = lines[i].strip()
                    if not line:
                        i += 1
                        continue

                    # 解析序号
                    index = int(line)
                    i += 1

                    # 解析时间行
                    time_line = lines[i].strip()
                    i += 1
                    if '-->' not in time_line:
                        continue

                    start_str, end_str = time_line.split('-->', 1)
                    start_time = self._parse_srt_time(start_str.strip())
                    end_time = self._parse_srt_time(end_str.strip())

                    # 解析文本行
                    text_lines = []
                    while i < len(lines) and lines[i].strip():
                        text_lines.append(lines[i].strip())
                        i += 1
                    text = ' '.join(text_lines)

                    # 创建字幕条目
                    entry = SubtitleEntry(
                        start_time=SubtitleEntry._seconds_to_time(start_time),
                        end_time=SubtitleEntry._seconds_to_time(end_time),
                        text=text,
                        style="Default",
                        actor=""
                    )
                    entries.append(entry)

                except Exception as e:
                    self.logger.warning(f"解析字幕条目失败: {line}")
                    i += 1
                    continue

            entries.sort(key=lambda x: x.start_time_seconds())
            return entries

        except Exception as e:
            self.logger.error(f"SRT字幕解析失败: {str(e)}")
            return []

    def _parse_srt_time(self, time_str: str) -> float:
        """解析SRT时间格式为秒数"""
        try:
            time_part, ms_part = time_str.split(',')
            hours, minutes, seconds = time_part.split(':')

            total_seconds = (int(hours) * 3600 +
                             int(minutes) * 60 +
                             int(seconds) +
                             int(ms_part) / 1000)
            return total_seconds
        except Exception as e:
            self.logger.error(f"时间解析失败: {time_str}")
            return 0.0

    def _validate_data_consistency(self, tts_segments: List[AudioSegment],
                                   subtitle_entries: List[SubtitleEntry]) -> bool:
        """验证TTS结果与字幕数据的一致性"""
        try:
            if len(tts_segments) != len(subtitle_entries):
                self.logger.warning(f"TTS片段数量({len(tts_segments)})与字幕条目数量({len(subtitle_entries)})不匹配")

            for segment in tts_segments:
                if not segment.is_silence and not os.path.exists(segment.file_path):
                    self.logger.error(f"TTS音频文件不存在: {segment.file_path}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"数据一致性验证失败: {str(e)}")
            return False

    def _generate_audio_segments(self, tts_segments: List[AudioSegment],
                                 subtitle_entries: List[SubtitleEntry],
                                 srt_path: str) -> List[AudioSegment]:
        """生成音频片段列表，包括静音片段"""
        try:
            if not subtitle_entries:
                return []

            # 设置输出目录
            srt_file_path = Path(srt_path)
            output_dir = srt_file_path.parent / "silence_segments"
            output_dir.mkdir(parents=True, exist_ok=True)

            audio_segments = []
            current_time = 0.0

            for i, subtitle in enumerate(subtitle_entries):
                # 获取字幕时间信息
                subtitle_start = subtitle.start_time_seconds()
                subtitle_duration = subtitle.duration_seconds()

                # 检查是否有对应的TTS片段（按索引匹配）
                tts_segment = None
                for tts in tts_segments:
                    if tts.index == i + 1:
                        tts_segment = tts
                        break

                # 计算与当前时间的差距
                time_gap = subtitle_start - current_time

                # 如果有时间差距，生成静音片段
                if time_gap > 0.01:
                    silence_path = output_dir / f"silence_{i:04d}_{current_time:.3f}.wav"
                    silence_duration = time_gap

                    if self._generate_silence_audio(str(silence_path), silence_duration):
                        silence_segment = AudioSegment(
                            index=len(audio_segments),
                            file_path=str(silence_path),
                            duration=silence_duration,
                            start_time=current_time,
                            end_time=current_time + silence_duration,
                            is_silence=True,
                            text=f"静音片段 {current_time:.3f}s - {current_time + silence_duration:.3f}s"
                        )
                        audio_segments.append(silence_segment)
                        current_time += silence_duration

                # 添加TTS音频片段
                if tts_segment:
                    # 创建TTS片段的副本以避免修改原始对象
                    tts_copy = AudioSegment(
                        index=len(audio_segments),
                        file_path=tts_segment.file_path,
                        duration=tts_segment.duration,
                        start_time=current_time,
                        end_time=current_time + tts_segment.duration,
                        is_silence=tts_segment.is_silence,
                        text=tts_segment.text
                    )
                    audio_segments.append(tts_copy)
                    current_time += tts_copy.duration
                else:
                    # 如果没有对应的TTS片段，生成静音片段
                    silence_path = output_dir / f"silence_{i:04d}_missing_{current_time:.3f}.wav"
                    silence_duration = subtitle_duration

                    if self._generate_silence_audio(str(silence_path), silence_duration):
                        silence_segment = AudioSegment(
                            index=len(audio_segments),
                            file_path=str(silence_path),
                            duration=silence_duration,
                            start_time=current_time,
                            end_time=current_time + silence_duration,
                            is_silence=True,
                            text=f"缺失TTS片段静音 {current_time:.3f}s - {current_time + silence_duration:.3f}s"
                        )
                        audio_segments.append(silence_segment)
                        current_time += silence_duration

            return audio_segments

        except Exception as e:
            self.logger.error(f"音频片段生成失败: {str(e)}")
            return []

    def _generate_silence_audio(self, output_path: str, duration: float) -> bool:
        """生成静音音频文件"""
        try:
            duration = max(0.01, duration)
            samples = int(self.sample_rate * duration)
            silence_data = np.zeros(samples, dtype=np.float32)
            sf.write(output_path, silence_data, self.sample_rate)
            return True
        except Exception as e:
            self.logger.error(f"静音音频生成失败: {str(e)}")
            return False

    def _generate_default_output_path(self, tts_results_path: str) -> str:
        """基于TTS结果路径生成默认输出路径"""
        try:
            tts_path = Path(tts_results_path)

            if "tts_output" in tts_path.parts:
                tts_output_index = tts_path.parts.index("tts_output")
                output_parts = list(tts_path.parts[:tts_output_index]) + ["aligned_audio"] + [
                    f"aligned_{tts_path.stem}.wav"]
                output_path = Path(*output_parts)
            else:
                output_path = tts_path.parent / f"aligned_{tts_path.stem}.wav"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            return str(output_path)

        except Exception as e:
            self.logger.error(f"默认输出路径生成失败: {str(e)}")
            return "aligned_audio.wav"

    def _concatenate_audio_segments(self, audio_segments: List[AudioSegment],
                                    total_duration: float) -> np.ndarray:
        """拼接音频片段"""
        try:
            if not audio_segments:
                self.logger.warning("没有音频片段需要拼接")
                return np.zeros(int(self.sample_rate * total_duration), dtype=np.float32)

            total_samples = int(self.sample_rate * total_duration)
            merged_audio = np.zeros(total_samples, dtype=np.float32)

            for segment in audio_segments:
                try:
                    # 从文件加载音频数据
                    audio_data, sr = sf.read(segment.file_path)

                    # 如果采样率不一致，进行重采样
                    if sr != self.sample_rate:
                        self.logger.info(f"重采样: {sr} -> {self.sample_rate}")
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)

                    start_sample = int(segment.start_time * self.sample_rate)
                    end_sample = int(segment.end_time * self.sample_rate)

                    if start_sample >= total_samples:
                        continue

                    end_sample = min(end_sample, total_samples)
                    copy_length = min(len(audio_data), end_sample - start_sample)

                    if copy_length > 0:
                        merged_audio[start_sample:start_sample + copy_length] = audio_data[:copy_length]

                except Exception as e:
                    self.logger.error(f"处理音频片段失败: {segment.index} - {str(e)}")
                    continue

            return merged_audio

        except Exception as e:
            self.logger.error(f"音频片段拼接失败: {str(e)}")
            return np.zeros(int(self.sample_rate * total_duration), dtype=np.float32)

    def _save_merged_audio(self, audio_data: np.ndarray, output_path: str):
        """保存合并后的音频"""
        try:
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存音频文件
            sf.write(output_path, audio_data, self.sample_rate)

            self.logger.info(f"合并音频已保存: {output_path}")

        except Exception as e:
            self.logger.error(f"合并音频保存失败: {str(e)}")
            raise

    def get_processing_statistics(self, audio_segments: List[AudioSegment]) -> Dict[str, Any]:
        """获取处理统计信息"""
        try:
            total_duration = sum(seg.duration for seg in audio_segments)
            silence_duration = sum(seg.duration for seg in audio_segments if seg.is_silence)
            tts_duration = sum(seg.duration for seg in audio_segments if not seg.is_silence)

            return {
                'total_segments': len(audio_segments),
                'total_duration': round(total_duration, 3),
                'silence_segments': sum(1 for seg in audio_segments if seg.is_silence),
                'tts_segments': sum(1 for seg in audio_segments if not seg.is_silence),
                'silence_duration': round(silence_duration, 3),
                'tts_duration': round(tts_duration, 3),
                'silence_ratio': round(silence_duration / total_duration, 3) if total_duration > 0 else 0,
                'sample_rate': self.sample_rate
            }

        except Exception as e:
            self.logger.error(f"统计信息获取失败: {str(e)}")
            return {}

    def cleanup_temp_files(self, audio_segments: List[AudioSegment]):
        """清理静音片段文件"""
        try:
            cleaned_count = 0
            for segment in audio_segments:
                if segment.is_silence and segment.file_path and os.path.exists(segment.file_path):
                    try:
                        os.remove(segment.file_path)
                        cleaned_count += 1
                    except Exception as e:
                        self.logger.warning(f"清理静音文件失败: {segment.file_path}")

            if cleaned_count > 0:
                self.logger.info(f"清理了 {cleaned_count} 个静音文件")

        except Exception as e:
            self.logger.error(f"文件清理失败: {str(e)}")

    def _save_results_to_json(self, results: Dict[str, Any], tts_results_path: str) -> str:
        """保存结果到JSON文件"""
        try:
            # 创建JSON文件路径
            tts_path = Path(tts_results_path)
            if "tts_output" in tts_path.parts:
                tts_output_index = tts_path.parts.index("tts_output")
                output_parts = list(tts_path.parts[:tts_output_index]) + ["aligned_audio"]
                output_dir = Path(*output_parts)
            else:
                output_dir = tts_path.parent

            json_filename = f"aligned_{tts_path.stem}_results.json"
            json_path = output_dir / json_filename

            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)

            # 处理NumPy类型和浮点数精度的序列化
            def numpy_float_serializer(obj):
                if isinstance(obj, float):
                    return round(obj, 3)
                elif hasattr(obj, 'item'):  # NumPy标量
                    return round(obj.item(), 3) if isinstance(obj.item(), float) else obj.item()
                elif hasattr(obj, 'tolist'):  # NumPy数组
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return round(obj.item(), 3) if isinstance(obj, np.floating) else obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            # 添加保存时间戳
            results_copy = results.copy()
            results_copy['saved_at'] = datetime.now().isoformat()
            results_copy['file_version'] = "1.0"

            # 保存JSON文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_copy, f, ensure_ascii=False, indent=2, default=numpy_float_serializer)

            self.logger.info(f"结果已保存到JSON文件: {json_path}")
            return str(json_path)

        except Exception as e:
            self.logger.error(f"保存JSON文件失败: {str(e)}")
            return ""

    def _load_cached_results(self, tts_results_path: str, srt_path: str) -> Optional[Dict[str, Any]]:
        """加载缓存的结果（已移除缓存功能）"""
        return None

    def _is_cache_valid(self, cached_results: Dict[str, Any], tts_results_path: str, srt_path: str) -> bool:
        """验证缓存是否有效（已移除缓存功能）"""
        return False

    def generate_aligned_srt(self,
                             aligned_results_path: str,
                             original_srt_path: str,
                             output_srt_path: Optional[str] = None) -> Dict[str, Any]:
        """
        根据对齐结果生成新的SRT字幕文件
        
        Args:
            aligned_results_path: 对齐结果JSON文件路径
            original_srt_path: 原始SRT字幕文件路径
            output_srt_path: 输出SRT文件路径（可选）
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            self.logger.info(f"开始生成对齐后的SRT字幕: {aligned_results_path}")

            # 加载对齐结果
            if not os.path.exists(aligned_results_path):
                return {'success': False, 'error': f'对齐结果文件不存在: {aligned_results_path}'}

            with open(aligned_results_path, 'r', encoding='utf-8') as f:
                aligned_results = json.load(f)

            if not aligned_results.get('success', False):
                return {'success': False, 'error': '对齐结果显示处理失败'}

            # 加载原始SRT字幕
            original_subtitles = self._parse_srt_subtitles(original_srt_path)
            if not original_subtitles:
                return {'success': False, 'error': '原始SRT字幕解析失败'}

            # 获取对齐后的音频片段
            aligned_segments = aligned_results.get('aligned_audio_segments', [])

            # 提取TTS片段（非静音片段）
            tts_segments = [seg for seg in aligned_segments if not seg.get('is_silence', False)]

            # 生成默认输出路径
            if output_srt_path is None:
                output_srt_path = self._generate_default_srt_path(aligned_results_path)

            # 确保输出目录存在
            output_dir = Path(output_srt_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # 生成新的SRT内容
            srt_content = self._generate_srt_content(tts_segments, original_subtitles)

            # 保存SRT文件
            with open(output_srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)

            result = {
                'success': True,
                'output_srt_path': output_srt_path,
                'subtitle_count': len(tts_segments),
                'total_duration': round(aligned_results.get('total_duration', 0), 3),
                'processing_info': {
                    'aligned_results_path': aligned_results_path,
                    'original_srt_path': original_srt_path,
                    'processed_at': datetime.now().isoformat()
                }
            }

            self.logger.info(f"对齐SRT字幕生成完成: {output_srt_path}")
            return result

        except Exception as e:
            self.logger.error(f"生成对齐SRT字幕失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _generate_default_srt_path(self, aligned_results_path: str) -> str:
        """生成默认的SRT输出路径"""
        try:
            aligned_path = Path(aligned_results_path)

            if "aligned_audio" in aligned_path.parts:
                aligned_index = aligned_path.parts.index("aligned_audio")
                output_parts = list(aligned_path.parts[:aligned_index]) + ["aligned_subtitles"] + [
                    f"{aligned_path.stem.replace('_results', '')}_aligned.srt"]
                output_path = Path(*output_parts)
            else:
                output_path = aligned_path.parent / f"{aligned_path.stem}_aligned.srt"

            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)

        except Exception as e:
            self.logger.error(f"默认SRT路径生成失败: {str(e)}")
            return "aligned_subtitles.srt"

    def _generate_srt_content(self, tts_segments: List[Dict[str, Any]],
                              original_subtitles: List[SubtitleEntry]) -> str:
        """生成SRT文件内容"""
        try:
            srt_lines = []

            # 确保字幕数量匹配
            subtitle_count = min(len(tts_segments), len(original_subtitles))

            for i in range(subtitle_count):
                tts_segment = tts_segments[i]
                original_subtitle = original_subtitles[i]

                # 获取对齐后的时间戳
                start_time = tts_segment.get('start_time', 0.0)
                end_time = tts_segment.get('end_time', 0.0)

                # 格式化时间戳
                start_time_str = self._seconds_to_srt_time(start_time)
                end_time_str = self._seconds_to_srt_time(end_time)

                # 使用原始字幕文本
                subtitle_text = original_subtitle.text

                # 构建SRT条目
                srt_lines.append(str(i + 1))  # 序号
                srt_lines.append(f"{start_time_str} --> {end_time_str}")  # 时间戳
                srt_lines.append(subtitle_text)  # 文本
                srt_lines.append("")  # 空行

            return '\n'.join(srt_lines)

        except Exception as e:
            self.logger.error(f"SRT内容生成失败: {str(e)}")
            raise

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """将秒数转换为SRT时间格式"""
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            milliseconds = int((seconds % 1) * 1000)

            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

        except Exception as e:
            self.logger.error(f"时间格式转换失败: {seconds}")
            return "00:00:00,000"

    def process_video_speed_adjustment(self,
                                        silent_video_path: str,
                                        original_srt_path: str,
                                        new_srt_path: str) -> Dict[str, Any]:
        """
        处理视频变速调整：根据原字幕和新字幕进行视频分割、变速处理和拼接
        
        Args:
            silent_video_path: 无声视频文件路径
            original_srt_path: 原字幕文件路径
            new_srt_path: 新字幕文件路径
            
        Returns:
            Dict[str, Any]: 处理结果，包含视频片段信息、变速比例和最终拼接视频
        """
        try:
            self.logger.info(f"开始视频切割分析: {silent_video_path}")

            # 解析原字幕和新字幕
            original_subtitles = self._parse_srt_subtitles(original_srt_path)
            new_subtitles = self._parse_srt_subtitles(new_srt_path)

            if not original_subtitles:
                return {'success': False, 'error': '原字幕解析失败'}
            if not new_subtitles:
                return {'success': False, 'error': '新字幕解析失败'}

            # 获取视频总时长
            video_total_duration = self._get_video_duration(silent_video_path)
            if video_total_duration <= 0:
                return {'success': False, 'error': '无法获取视频时长'}

            # 根据原字幕切割视频片段
            video_segments = self._cut_video_segments_by_subtitles(
                original_subtitles, video_total_duration)

            # 计算变速比例
            speed_adjusted_segments = self._calculate_speed_ratios(
                video_segments, original_subtitles, new_subtitles)

            # 格式化变速片段
            speed_adjusted_segments = self._merge_short_segments(speed_adjusted_segments, min_duration=0.3)

            # 构建完整的处理结果
            processing_result = {
                'success': True,
                'silent_video_path': silent_video_path,
                'original_video_duration': round(video_total_duration, 3),
                'subtitle_count': len(original_subtitles),
                'video_segments': len(speed_adjusted_segments),
                'speed_adjusted_segments': speed_adjusted_segments,
                'processing_info': {
                    'silent_video_path': silent_video_path,
                    'original_srt_path': original_srt_path,
                    'new_srt_path': new_srt_path,
                    'processed_at': datetime.now().isoformat()
                }
            }

            # 根据处理结果进行分割
            self.logger.info("开始实际分割视频片段...")
            updated_segments = self._split_video_segments(silent_video_path, speed_adjusted_segments)
            if updated_segments:
                processing_result['speed_adjusted_segments'] = updated_segments
                self.logger.info(f"视频分割完成，共生成 {len(updated_segments)} 个片段文件")
            else:
                self.logger.warning("视频分割失败，保留原始片段信息")

            # 保存结果到JSON文件
            json_file_path = self._save_video_results_to_json(processing_result, silent_video_path)
            if json_file_path:
                processing_result['results_json_file'] = json_file_path

            self.logger.info(f"视频切割分析完成，结果已保存: {json_file_path}")

            # 对片段进行变速
            if updated_segments:
                self.logger.info("开始对片段进行变速处理...")
                speed_processed_segments = self._apply_speed_to_segments(updated_segments)
                if speed_processed_segments:
                    updated_segments = speed_processed_segments
                    processing_result['speed_adjusted_segments'] = updated_segments
                    self.logger.info(f"片段变速处理完成，共处理 {len(updated_segments)} 个片段")

                    # 变速处理完成后，再次保存更新的结果
                    final_json_file_path = self._save_video_results_to_json(processing_result, silent_video_path)
                    if final_json_file_path:
                        processing_result['results_json_file'] = final_json_file_path
                        self.logger.info(f"变速处理结果已更新保存: {final_json_file_path}")
                else:
                    self.logger.warning("片段变速处理失败，保留原始片段")
            else:
                self.logger.warning("没有有效的片段进行变速处理")

            # 对变速后的片段进行拼接
            if updated_segments and any(seg.get('speed_processing_success', False) for seg in updated_segments):
                self.logger.info("开始拼接变速后的视频片段...")
                final_video_result = self._concatenate_speed_adjusted_segments(updated_segments, silent_video_path)
                if final_video_result.get('success', False):
                    processing_result['final_video_path'] = final_video_result['output_path']
                    processing_result['final_video_duration'] = final_video_result.get('total_duration', 0)
                    processing_result['concatenation_success'] = True
                    self.logger.info(f"视频拼接完成: {final_video_result['output_path']}")

                    # 拼接完成后，最终保存完整结果
                    final_json_file_path = self._save_video_results_to_json(processing_result, silent_video_path)
                    if final_json_file_path:
                        processing_result['results_json_file'] = final_json_file_path
                        self.logger.info(f"最终处理结果已保存: {final_json_file_path}")
                else:
                    processing_result['concatenation_success'] = False
                    processing_result['concatenation_error'] = final_video_result.get('error', '拼接失败')
                    self.logger.warning("视频拼接失败，请检查片段文件")
            else:
                self.logger.warning("没有成功的变速片段进行拼接")

            return processing_result

        except Exception as e:
            self.logger.error(f"视频切割分析失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _get_video_duration(self, video_path: str) -> float:
        """获取视频总时长"""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries',
                'format=duration', '-of', 'csv=p=0', video_path
            ], capture_output=True, text=True)

            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                self.logger.error(f"获取视频时长失败: {result.stderr}")
                return 0.0

        except Exception as e:
            self.logger.error(f"获取视频时长异常: {str(e)}")
            return 0.0

    def _cut_video_segments_by_subtitles(self,
                                         subtitles: List[SubtitleEntry],
                                         video_duration: float) -> List[VideoSegment]:
        """根据字幕时间戳切割视频片段"""
        try:
            segments = []
            segment_index = 0
            current_time = 0.0

            for i, subtitle in enumerate(subtitles):
                subtitle_start = subtitle.start_time_seconds()
                subtitle_end = subtitle.end_time_seconds()

                # 如果当前时间到字幕开始有间隔，添加间隔片段
                if subtitle_start > current_time:
                    gap_duration = subtitle_start - current_time
                    gap_type = "prefix" if i == 0 else "gap"
                    gap_segment = VideoSegment(
                        index=segment_index,
                        start_time=current_time,
                        end_time=subtitle_start,
                        duration=gap_duration,
                        segment_type=gap_type,
                        text=f"间隔片段 {current_time:.3f}s - {subtitle_start:.3f}s"
                    )
                    segments.append(gap_segment)
                    segment_index += 1
                    current_time = subtitle_start

                # 添加字幕片段
                subtitle_duration = subtitle_end - subtitle_start
                subtitle_segment = VideoSegment(
                    index=segment_index,
                    start_time=subtitle_start,
                    end_time=subtitle_end,
                    duration=subtitle_duration,
                    segment_type="subtitle",
                    text=subtitle.text,
                    subtitle_index=i,
                    original_duration=subtitle_duration
                )
                segments.append(subtitle_segment)
                segment_index += 1
                current_time = subtitle_end

            # 如果最后一个字幕到视频结束有间隔，添加后缀片段
            if current_time < video_duration:
                suffix_duration = video_duration - current_time
                suffix_segment = VideoSegment(
                    index=segment_index,
                    start_time=current_time,
                    end_time=video_duration,
                    duration=suffix_duration,
                    segment_type="suffix",
                    text=f"后缀片段 {current_time:.3f}s - {video_duration:.3f}s"
                )
                segments.append(suffix_segment)

            return segments

        except Exception as e:
            self.logger.error(f"视频切割失败: {str(e)}")
            return []

    def _calculate_speed_ratios(self,
                                video_segments: List[VideoSegment],
                                original_subtitles: List[SubtitleEntry],
                                new_subtitles: List[SubtitleEntry]) -> List[Dict[str, Any]]:
        """计算每个视频片段的变速比例"""
        try:
            adjusted_segments = []

            for segment in video_segments:
                segment_info = {
                    'index': segment.index,
                    'start_time': round(segment.start_time, 3),
                    'end_time': round(segment.end_time, 3),
                    'duration': round(segment.duration, 3),
                    'segment_type': segment.segment_type,
                    'text': segment.text,
                    'original_duration': round(segment.original_duration or segment.duration, 3)
                }

                if segment.segment_type == "subtitle" and segment.subtitle_index is not None:
                    # 对于字幕片段，计算变速比例
                    subtitle_idx = segment.subtitle_index

                    if subtitle_idx < len(new_subtitles):
                        new_subtitle = new_subtitles[subtitle_idx]
                        new_duration = new_subtitle.duration_seconds()
                        original_duration = segment.original_duration or segment.duration

                        if original_duration > 0:
                            speed_ratio = original_duration / new_duration
                        else:
                            speed_ratio = 1.0

                        segment_info.update({
                            'target_duration': round(new_duration, 3),
                            'speed_ratio': round(speed_ratio, 3),
                            'subtitle_index': subtitle_idx,
                            'new_subtitle_text': new_subtitle.text
                        })
                    else:
                        # 如果新字幕数量不够，保持原速度
                        segment_info.update({
                            'target_duration': segment_info['original_duration'],
                            'speed_ratio': 1.0,
                            'subtitle_index': subtitle_idx
                        })
                else:
                    # 对于非字幕片段（间隔、前缀、后缀），保持原速度
                    segment_info.update({
                        'target_duration': segment_info['original_duration'],
                        'speed_ratio': 1.0
                    })

                adjusted_segments.append(segment_info)

            return adjusted_segments

        except Exception as e:
            self.logger.error(f"变速比例计算失败: {str(e)}")
            return []

    def _merge_short_segments(self,
                              segments: List[Dict[str, Any]],
                              min_duration: float = 0.3) -> List[Dict[str, Any]]:
        """
        合并时长过短的片段到相邻片段
        
        Args:
            segments: 原始片段列表
            min_duration: 最小时长阈值（秒）
            
        Returns:
            List[Dict[str, Any]]: 合并后的片段列表
        """
        try:
            if not segments:
                return segments

            self.logger.info(f"开始合并时长小于 {min_duration}s 的短片段")

            merged_segments = []
            i = 0

            while i < len(segments):
                current_segment = segments[i].copy()

                # 检查当前片段是否太短
                if current_segment.get('duration', 0) < min_duration:
                    self.logger.debug(
                        f"发现短片段 {current_segment['index']}: {current_segment.get('duration', 0):.3f}s")

                    # 如果有前一个片段，合并到前一个片段
                    if merged_segments:
                        prev_segment = merged_segments[-1]
                        self.logger.debug(f"将片段 {current_segment['index']} 合并到片段 {prev_segment['index']}")

                        # 合并时间信息
                        new_start_time = min(prev_segment['start_time'], current_segment['start_time'])
                        new_end_time = max(prev_segment['end_time'], current_segment['end_time'])
                        new_duration = new_end_time - new_start_time

                        # 更新前一个片段
                        prev_segment.update({
                            'start_time': round(new_start_time, 3),
                            'end_time': round(new_end_time, 3),
                            'duration': round(new_duration, 3),
                            'original_duration': round(new_duration, 3)
                        })

                        # 合并文本信息
                        if current_segment.get('text') and prev_segment.get('text'):
                            if current_segment['segment_type'] == 'subtitle' and prev_segment[
                                'segment_type'] == 'subtitle':
                                # 两个都是字幕片段，合并文本
                                prev_segment['text'] = f"{prev_segment['text']} {current_segment['text']}"
                            elif current_segment['segment_type'] != 'subtitle':
                                # 当前是间隔片段，更新描述
                                prev_segment['text'] = f"合并片段 {new_start_time:.3f}s - {new_end_time:.3f}s"

                        # 更新片段类型（如果合并了不同类型）
                        if prev_segment['segment_type'] != current_segment['segment_type']:
                            if prev_segment['segment_type'] == 'subtitle' or current_segment[
                                'segment_type'] == 'subtitle':
                                prev_segment['segment_type'] = 'subtitle'  # 优先保持字幕类型
                            else:
                                prev_segment['segment_type'] = 'merged'

                        # 更新变速信息（如果都有的话）
                        if 'target_duration' in current_segment and 'target_duration' in prev_segment:
                            # 重新计算目标时长和变速比例
                            if 'new_subtitle_text' in current_segment and 'new_subtitle_text' in prev_segment:
                                # 两个都是字幕片段，需要重新计算
                                new_target_duration = prev_segment.get('target_duration', 0) + current_segment.get(
                                    'target_duration', 0)
                                prev_segment['target_duration'] = round(new_target_duration, 3)

                                if new_target_duration > 0:
                                    prev_segment['speed_ratio'] = round(new_duration / new_target_duration, 3)
                                else:
                                    prev_segment['speed_ratio'] = 1.0

                                # 合并字幕信息
                                if 'new_subtitle_text' in current_segment and 'new_subtitle_text' in prev_segment:
                                    prev_segment[
                                        'new_subtitle_text'] = f"{prev_segment['new_subtitle_text']} {current_segment['new_subtitle_text']}"

                        # 标记为已合并
                        prev_segment['merged_from'] = prev_segment.get('merged_from', []) + [current_segment['index']]

                    # 如果没有前一个片段，且有下一个片段，尝试合并到下一个片段
                    elif i + 1 < len(segments):
                        next_segment = segments[i + 1].copy()
                        self.logger.debug(
                            f"将首个短片段 {current_segment['index']} 合并到下一个片段 {next_segment['index']}")

                        # 合并时间信息
                        new_start_time = min(current_segment['start_time'], next_segment['start_time'])
                        new_end_time = max(current_segment['end_time'], next_segment['end_time'])
                        new_duration = new_end_time - new_start_time

                        # 更新下一个片段信息
                        next_segment.update({
                            'start_time': round(new_start_time, 3),
                            'end_time': round(new_end_time, 3),
                            'duration': round(new_duration, 3),
                            'original_duration': round(new_duration, 3)
                        })

                        # 合并文本
                        if current_segment.get('text') and next_segment.get('text'):
                            if current_segment['segment_type'] == 'subtitle' and next_segment[
                                'segment_type'] == 'subtitle':
                                next_segment['text'] = f"{current_segment['text']} {next_segment['text']}"
                            else:
                                next_segment['text'] = f"合并片段 {new_start_time:.3f}s - {new_end_time:.3f}s"

                        # 更新其他信息
                        if next_segment['segment_type'] != current_segment['segment_type']:
                            if next_segment['segment_type'] == 'subtitle' or current_segment[
                                'segment_type'] == 'subtitle':
                                next_segment['segment_type'] = 'subtitle'
                            else:
                                next_segment['segment_type'] = 'merged'

                        # 更新变速信息（合并到下一个片段）
                        if 'target_duration' in current_segment and 'target_duration' in next_segment:
                            # 重新计算目标时长和变速比例
                            if 'new_subtitle_text' in current_segment and 'new_subtitle_text' in next_segment:
                                # 两个都是字幕片段，需要重新计算
                                new_target_duration = current_segment.get('target_duration', 0) + next_segment.get(
                                    'target_duration', 0)
                                next_segment['target_duration'] = round(new_target_duration, 3)

                                if new_target_duration > 0:
                                    next_segment['speed_ratio'] = round(new_duration / new_target_duration, 3)
                                else:
                                    next_segment['speed_ratio'] = 1.0

                                # 合并字幕信息
                                if 'new_subtitle_text' in current_segment and 'new_subtitle_text' in next_segment:
                                    next_segment[
                                        'new_subtitle_text'] = f"{current_segment['new_subtitle_text']} {next_segment['new_subtitle_text']}"

                        next_segment['merged_from'] = next_segment.get('merged_from', []) + [current_segment['index']]

                        # 添加合并后的片段，跳过下一个原始片段
                        merged_segments.append(next_segment)
                        i += 2  # 跳过当前和下一个
                        continue
                    else:
                        # 只有一个片段且太短，保留它
                        self.logger.warning(f"唯一片段 {current_segment['index']} 时长过短，但无法合并")
                        merged_segments.append(current_segment)
                else:
                    # 片段时长足够，直接添加
                    merged_segments.append(current_segment)

                i += 1

            # 重新分配索引
            for idx, segment in enumerate(merged_segments):
                segment['index'] = idx

            original_count = len(segments)
            merged_count = len(merged_segments)
            self.logger.info(f"片段合并完成: {original_count} -> {merged_count} 个片段")

            return merged_segments

        except Exception as e:
            self.logger.error(f"合并短片段失败: {str(e)}")
            return segments  # 返回原始片段列表

    def _apply_speed_to_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对分割后的视频片段应用变速处理
        
        Args:
            segments: 已分割的视频片段列表（包含segment_file_path）
            
        Returns:
            List[Dict[str, Any]]: 变速处理后的片段列表
        """
        try:
            if not segments:
                return segments

            self.logger.info("开始对视频片段应用变速处理")

            processed_segments = []
            success_count = 0

            # 检测GPU加速选项
            gpu_params = self._detect_gpu_acceleration()

            for segment in segments:
                try:
                    # 检查片段是否有文件路径
                    if not segment.get('segment_file_path') or not os.path.exists(segment['segment_file_path']):
                        self.logger.warning(f"片段 {segment['index']} 没有有效的文件路径，跳过变速处理")
                        processed_segments.append(segment)
                        continue

                    # 检查是否需要变速
                    speed_ratio = segment.get('speed_ratio', 1.0)
                    if abs(speed_ratio - 1.0) < 0.01:  # 变速比例接近1.0，不需要变速
                        self.logger.debug(f"片段 {segment['index']} 变速比例为 {speed_ratio:.3f}，不需要变速")
                        processed_segments.append(segment)
                        success_count += 1
                        continue

                    # 生成变速后的文件路径
                    original_path = Path(segment['segment_file_path'])
                    speed_filename = f"{original_path.stem}_speed_{speed_ratio:.2f}.mp4"
                    speed_file_path = original_path.parent / speed_filename

                    # 构建ffmpeg变速命令
                    cmd = ['ffmpeg', '-y']

                    # 添加GPU加速参数
                    if gpu_params:
                        cmd.extend(gpu_params['decode'])

                    # 输入文件
                    cmd.extend(['-i', str(original_path)])

                    # 变速处理
                    if speed_ratio > 1.0:
                        # 加速：使用setpts滤镜
                        pts_ratio = 1.0 / speed_ratio
                        cmd.extend(['-filter:v', f'setpts={pts_ratio}*PTS'])
                    else:
                        # 减速：可选择使用插值
                        pts_ratio = 1.0 / speed_ratio
                        cmd.extend(['-filter:v', f'setpts={pts_ratio}*PTS'])

                    # 添加GPU编码参数
                    if gpu_params:
                        cmd.extend(gpu_params['encode'])
                    else:
                        cmd.extend(['-c:v', 'libx264', '-preset', 'fast'])

                    # 输出参数
                    cmd.extend([
                        '-avoid_negative_ts', 'make_zero',
                        '-an',  # 移除音频
                        str(speed_file_path)
                    ])

                    # 执行变速处理
                    self.logger.debug(f"变速片段 {segment['index']}: 比例 {speed_ratio:.3f}")
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        # 变速成功，更新片段信息
                        updated_segment = segment.copy()
                        updated_segment['speed_processed_file_path'] = str(speed_file_path)
                        updated_segment['speed_processing_success'] = True

                        # 更新文件大小
                        if speed_file_path.exists():
                            updated_segment['speed_processed_file_size'] = speed_file_path.stat().st_size

                        # 计算实际处理后的时长（理论值）
                        original_duration = segment.get('original_duration', segment.get('duration', 0))
                        target_duration = segment.get('target_duration', original_duration / speed_ratio)
                        updated_segment['actual_output_duration'] = round(target_duration, 3)

                        processed_segments.append(updated_segment)
                        success_count += 1

                        self.logger.debug(f"片段 {segment['index']} 变速成功: {speed_file_path}")

                        # 可选：删除原始分割文件以节省空间
                        # os.remove(original_path)

                    else:
                        self.logger.error(f"片段 {segment['index']} 变速失败: {result.stderr}")
                        # 变速失败，保留原始信息但标记错误
                        updated_segment = segment.copy()
                        updated_segment['speed_processing_success'] = False
                        updated_segment['speed_processing_error'] = result.stderr
                        processed_segments.append(updated_segment)

                except Exception as e:
                    self.logger.error(f"处理片段 {segment.get('index', 'unknown')} 变速时发生异常: {str(e)}")
                    # 保留原始片段信息但标记异常
                    updated_segment = segment.copy()
                    updated_segment['speed_processing_success'] = False
                    updated_segment['speed_processing_error'] = str(e)
                    processed_segments.append(updated_segment)

            self.logger.info(f"片段变速处理完成: 成功 {success_count}/{len(segments)} 个片段")

            return processed_segments

        except Exception as e:
            self.logger.error(f"片段变速处理失败: {str(e)}")
            return segments  # 返回原始片段列表

    def _concatenate_speed_adjusted_segments(self,
                                             segments: List[Dict[str, Any]],
                                             original_video_path: str) -> Dict[str, Any]:
        """
        拼接变速处理后的视频片段
        
        Args:
            segments: 变速处理后的片段列表
            original_video_path: 原始视频路径（用于生成输出路径）
            
        Returns:
            Dict[str, Any]: 拼接结果
        """
        try:
            if not segments:
                return {'success': False, 'error': '没有片段需要拼接'}

            self.logger.info("开始拼接变速后的视频片段")

            # 生成输出路径
            video_file = Path(original_video_path)
            output_dir = video_file.parent / "adjusted_video"
            output_dir.mkdir(parents=True, exist_ok=True)
            final_video_path = output_dir / f"final_speed_adjusted_{video_file.stem}.mp4"

            # 收集有效的视频片段文件
            valid_segments = []
            total_target_duration = 0.0

            for segment in segments:
                # 优先使用变速后的文件，否则使用原始分割文件
                video_file_path = None
                if segment.get('speed_processing_success', False) and segment.get('speed_processed_file_path'):
                    video_file_path = segment['speed_processed_file_path']
                elif segment.get('segment_file_path'):
                    video_file_path = segment['segment_file_path']

                if video_file_path and os.path.exists(video_file_path):
                    valid_segments.append({
                        'index': segment['index'],
                        'file_path': video_file_path,
                        'target_duration': segment.get('actual_output_duration',
                                                       segment.get('target_duration', segment.get('duration', 0))),
                        'is_speed_processed': segment.get('speed_processing_success', False)
                    })
                    total_target_duration += valid_segments[-1]['target_duration']
                else:
                    self.logger.warning(f"片段 {segment['index']} 没有有效的视频文件，跳过")

            if not valid_segments:
                return {'success': False, 'error': '没有有效的视频片段文件'}

            # 使用临时目录处理拼接
            with tempfile.TemporaryDirectory() as temp_dir:
                # 创建文件列表
                list_file_path = Path(temp_dir) / "concat_list.txt"
                with open(list_file_path, 'w', encoding='utf-8') as f:
                    for segment in valid_segments:
                        f.write(f"file '{segment['file_path']}'\n")

                # 检测GPU加速选项
                gpu_params = self._detect_gpu_acceleration()

                # 构建ffmpeg拼接命令
                cmd = ['ffmpeg', '-y']

                # 添加GPU加速参数
                if gpu_params:
                    cmd.extend(['-hwaccel', gpu_params['decode'][1]])  # 仅使用hwaccel参数

                # 拼接参数
                cmd.extend([
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(list_file_path)
                ])

                # 添加GPU编码参数
                if gpu_params:
                    cmd.extend(gpu_params['encode'])
                else:
                    cmd.extend(['-c:v', 'libx264', '-preset', 'fast'])

                # 输出参数
                cmd.extend([
                    '-avoid_negative_ts', 'make_zero',
                    '-an',  # 移除音频
                    str(final_video_path)
                ])

                # 执行拼接
                self.logger.debug(f"拼接命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    # 拼接成功
                    file_size = final_video_path.stat().st_size if final_video_path.exists() else 0

                    concat_result = {
                        'success': True,
                        'output_path': str(final_video_path),
                        'total_duration': round(total_target_duration, 3),
                        'file_size': file_size,
                        'segments_count': len(valid_segments),
                        'speed_processed_count': sum(1 for seg in valid_segments if seg['is_speed_processed']),
                        'concatenation_info': {
                            'segments_used': len(valid_segments),
                            'total_segments': len(segments),
                            'concatenated_at': datetime.now().isoformat()
                        }
                    }

                    self.logger.info(f"视频拼接成功: {final_video_path}")
                    self.logger.info(
                        f"拼接了 {len(valid_segments)} 个片段，其中 {concat_result['speed_processed_count']} 个经过变速处理")

                    return concat_result
                else:
                    self.logger.error(f"视频拼接失败: {result.stderr}")
                    return {'success': False, 'error': f"ffmpeg拼接失败: {result.stderr}"}

        except Exception as e:
            self.logger.error(f"视频拼接处理失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _split_video_segments(self,
                              silent_video_path: str,
                              segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据片段信息实际分割视频文件
        
        Args:
            silent_video_path: 原始无声视频路径
            segments: 视频片段信息列表
            
        Returns:
            List[Dict[str, Any]]: 更新后的片段信息（包含分割文件路径）
        """
        try:
            # 创建输出目录
            video_file = Path(silent_video_path)
            segments_dir = video_file.parent / "adjusted_video"
            segments_dir.mkdir(parents=True, exist_ok=True)

            # 检测GPU加速选项
            gpu_params = self._detect_gpu_acceleration()

            updated_segments = []
            success_count = 0

            for segment in segments:
                try:
                    # 生成输出文件名
                    segment_filename = f"segment_{segment['index']:04d}_{segment['segment_type']}.mp4"
                    segment_path = segments_dir / segment_filename

                    # 构建ffmpeg命令
                    cmd = ['ffmpeg', '-y']

                    # 添加GPU加速参数
                    if gpu_params:
                        cmd.extend(gpu_params['decode'])

                    # 输入文件和时间参数
                    cmd.extend([
                        '-i', silent_video_path,
                        '-ss', str(segment['start_time']),
                        '-t', str(segment['original_duration'])
                    ])

                    # 添加GPU编码参数
                    if gpu_params:
                        cmd.extend(gpu_params['encode'])
                    else:
                        cmd.extend(['-c:v', 'libx264', '-preset', 'fast'])

                    # 输出参数
                    cmd.extend([
                        '-avoid_negative_ts', 'make_zero',
                        '-an',  # 移除音频
                        str(segment_path)
                    ])

                    # 执行分割
                    self.logger.debug(f"分割片段 {segment['index']}: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        # 分割成功，更新片段信息
                        updated_segment = segment.copy()
                        updated_segment['segment_file_path'] = str(segment_path)
                        updated_segment['file_size'] = segment_path.stat().st_size if segment_path.exists() else 0
                        updated_segments.append(updated_segment)
                        success_count += 1

                        self.logger.debug(f"片段 {segment['index']} 分割成功: {segment_path}")
                    else:
                        self.logger.error(f"片段 {segment['index']} 分割失败: {result.stderr}")
                        # 分割失败，保留原始信息但标记错误
                        updated_segment = segment.copy()
                        updated_segment['segment_file_path'] = None
                        updated_segment['split_error'] = result.stderr
                        updated_segments.append(updated_segment)

                except Exception as e:
                    self.logger.error(f"处理片段 {segment.get('index', 'unknown')} 时发生异常: {str(e)}")
                    # 保留原始片段信息但标记异常
                    updated_segment = segment.copy()
                    updated_segment['segment_file_path'] = None
                    updated_segment['split_error'] = str(e)
                    updated_segments.append(updated_segment)

            self.logger.info(f"视频分割完成: 成功 {success_count}/{len(segments)} 个片段")

            # 如果有成功的片段，返回更新后的列表
            if success_count > 0:
                return updated_segments
            else:
                self.logger.error("所有片段分割都失败了")
                return []

        except Exception as e:
            self.logger.error(f"视频分割处理失败: {str(e)}")
            return []

    def _detect_gpu_acceleration(self) -> Optional[Dict[str, List[str]]]:
        """
        检测可用的GPU硬件加速选项
        
        Returns:
            Dict[str, List[str]]: GPU加速参数，包含decode和encode参数
        """
        try:
            # 检测NVIDIA CUDA支持
            if self._check_nvidia_gpu():
                self.logger.info("检测到NVIDIA GPU，使用CUDA硬件加速")
                return {
                    'decode': ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'],
                    'encode': ['-c:v', 'h264_nvenc', '-preset', 'fast']
                }

            # 检测Intel QSV支持
            if self._check_intel_qsv():
                self.logger.info("检测到Intel QSV，使用硬件加速")
                return {
                    'decode': ['-hwaccel', 'qsv'],
                    'encode': ['-c:v', 'h264_qsv', '-preset', 'fast']
                }

            # 检测AMD VAAPI支持（Linux）
            if self._check_vaapi():
                self.logger.info("检测到VAAPI支持，使用硬件加速")
                return {
                    'decode': ['-hwaccel', 'vaapi', '-hwaccel_output_format', 'vaapi'],
                    'encode': ['-c:v', 'h264_vaapi']
                }

            self.logger.info("未检测到GPU硬件加速，使用CPU编码")
            return None

        except Exception as e:
            self.logger.warning(f"GPU加速检测失败: {str(e)}")
            return None

    def _check_nvidia_gpu(self) -> bool:
        """检测NVIDIA GPU和CUDA支持"""
        try:
            # 检查nvidia-smi命令
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                # 检查ffmpeg是否支持CUDA
                result = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True)
                return 'cuda' in result.stdout.lower()
            return False
        except:
            return False

    def _check_intel_qsv(self) -> bool:
        """检测Intel QSV支持"""
        try:
            result = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True)
            return 'qsv' in result.stdout.lower()
        except:
            return False

    def _check_vaapi(self) -> bool:
        """检测VAAPI支持（主要用于Linux）"""
        try:
            if platform.system().lower() != 'linux':
                return False
            result = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True)
            return 'vaapi' in result.stdout.lower()
        except:
            return False

    def _generate_default_video_output_path(self, video_path: str) -> str:
        """生成默认的视频输出路径"""
        try:
            video_file = Path(video_path)
            output_dir = video_file.parent / "adjusted_video"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"speed_adjusted_{video_file.stem}.mp4"
            return str(output_path)

        except Exception as e:
            self.logger.error(f"默认视频输出路径生成失败: {str(e)}")
            return "speed_adjusted_video.mp4"

    def _save_video_results_to_json(self, results: Dict[str, Any], video_path: str) -> str:
        """保存视频处理结果到JSON文件"""
        try:
            video_file = Path(video_path)
            output_dir = video_file.parent / "adjusted_video"
            output_dir.mkdir(parents=True, exist_ok=True)

            json_filename = f"speed_adjusted_{video_file.stem}_results.json"
            json_path = output_dir / json_filename

            # 添加保存时间戳
            results_copy = results.copy()
            results_copy['saved_at'] = datetime.now().isoformat()
            results_copy['file_version'] = "1.0"

            # 处理浮点数精度的序列化函数
            def float_precision_serializer(obj):
                if isinstance(obj, float):
                    return round(obj, 3)
                elif hasattr(obj, 'item'):  # NumPy标量
                    return round(obj.item(), 3) if isinstance(obj.item(), float) else obj.item()
                elif hasattr(obj, 'tolist'):  # NumPy数组
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return round(obj.item(), 3) if isinstance(obj, np.floating) else obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            # 保存JSON文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_copy, f, ensure_ascii=False, indent=2, default=float_precision_serializer)

            self.logger.info(f"视频处理结果已保存到JSON文件: {json_path}")
            return str(json_path)

        except Exception as e:
            self.logger.error(f"保存视频结果JSON文件失败: {str(e)}")
            return ""


# 全局单例实例
_processor_instance = None


def _get_processor() -> AudioAlignProcessor:
    """获取AudioAlignProcessor单例实例"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = AudioAlignProcessor()
    return _processor_instance


def align_audio_with_subtitles(tts_results_path: str,
                               srt_path: str,
                               output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    根据TTS结果和SRT字幕生成对齐的音频文件
    
    Args:
        tts_results_path: TTS生成结果JSON文件路径
        srt_path: 原始SRT字幕文件路径
        output_path: 输出音频文件路径（可选）
        
    Returns:
        Dict[str, Any]: 处理结果
        
    Example:
        >>> result = align_audio_with_subtitles(
        ...     tts_results_path="output/tts_output/tts_generation_results.json",
        ...     srt_path="output/srt.srt",
        ...     output_path="output/aligned_audio/aligned_audio.wav"
        ... )
        >>> if result['success']:
        ...     print(f"对齐音频已生成: {result['output_path']}")
    """
    processor = _get_processor()
    return processor.align_audio_with_subtitles(
        tts_results_path=tts_results_path,
        srt_path=srt_path,
        output_path=output_path
    )


def generate_aligned_srt(aligned_results_path: str,
                         original_srt_path: str,
                         output_srt_path: Optional[str] = None) -> Dict[str, Any]:
    """
    根据对齐结果生成新的SRT字幕文件
    
    Args:
        aligned_results_path: 对齐结果JSON文件路径
        original_srt_path: 原始SRT字幕文件路径
        output_srt_path: 输出SRT文件路径（可选）
        
    Returns:
        Dict[str, Any]: 处理结果
        
    Example:
        >>> result = generate_aligned_srt(
        ...     aligned_results_path="output/aligned_audio/aligned_tts_generation_results_results.json",
        ...     original_srt_path="output/srt.srt",
        ...     output_srt_path="output/aligned_subtitles/aligned_subtitle.srt"
        ... )
        >>> if result['success']:
        ...     print(f"新字幕文件已生成: {result['output_srt_path']}")
    """
    processor = _get_processor()
    return processor.generate_aligned_srt(
        aligned_results_path=aligned_results_path,
        original_srt_path=original_srt_path,
        output_srt_path=output_srt_path
    )


def process_video_speed_adjustment(silent_video_path: str,
                                    original_srt_path: str,
                                    new_srt_path: str) -> Dict[str, Any]:
    """
    处理视频变速调整：根据原字幕和新字幕进行视频分割、变速处理和拼接
    
    Args:
        silent_video_path: 无声视频文件路径
        original_srt_path: 原字幕文件路径
        new_srt_path: 新字幕文件路径
        
    Returns:
        Dict[str, Any]: 处理结果，包含视频片段信息、变速比例和最终拼接视频
        
    Example:
        >>> result = process_video_speed_adjustment(
        ...     silent_video_path="video/silent_video.mp4",
        ...     original_srt_path="subtitles/original.srt",
        ...     new_srt_path="subtitles/new.srt"
        ... )
        >>> if result['success']:
        ...     print(f"处理完成: {result['results_json_file']}")
        ...     print(f"最终视频: {result.get('final_video_path', 'N/A')}")
        ...     print(f"视频片段数: {result['video_segments']}")
        ...     print(f"字幕数量: {result['subtitle_count']}")
    """
    processor = _get_processor()
    return processor.process_video_speed_adjustment(
        silent_video_path=silent_video_path,
        original_srt_path=original_srt_path,
        new_srt_path=new_srt_path
    )
