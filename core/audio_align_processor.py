"""
音频对齐处理器
根据TTS生成结果和SRT字幕文件生成合并后的音频文件
"""

import os
import logging
import json
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
                'total_duration': total_duration,
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
                    'duration': audio_length / sample_rate if sample_rate > 0 else 0
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
                'total_duration': total_duration,
                'silence_segments': sum(1 for seg in audio_segments if seg.is_silence),
                'tts_segments': sum(1 for seg in audio_segments if not seg.is_silence),
                'silence_duration': silence_duration,
                'tts_duration': tts_duration,
                'silence_ratio': silence_duration / total_duration if total_duration > 0 else 0,
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

            # 处理NumPy类型的序列化
            def numpy_serializer(obj):
                if hasattr(obj, 'item'):  # NumPy标量
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # NumPy数组
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
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
                json.dump(results_copy, f, ensure_ascii=False, indent=2, default=numpy_serializer)

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
                'total_duration': aligned_results.get('total_duration', 0),
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
