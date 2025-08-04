"""
字幕处理器 - 专门用于处理各种字幕文件格式

支持格式: ASS、SRT、VTT、LRC、SBV、SAMI、TTML 等格式的转换和处理

主要功能:
    - ASS/SSA 格式解析和转换
    - 多种字幕格式互相转换
    - ASS 文件格式化（处理相同 Style 连续出现）
    - SRT 到 ASS 时间戳同步
    - 字幕文件验证和合并
    - 支持多种编码格式

快速使用:
    from .subtitle_processor import *

    # 1. 多格式转换
    convert_subtitle('input.ass', 'output.srt')
    convert_subtitle('subtitle.vtt', 'subtitle.srt')

    # 2. 获取 ASS 文件中的样式
    styles = get_available_styles('subtitle.ass')

    # 3. 从 ASS 提取指定样式为 SRT
    extract_ass_to_srt('subtitle.ass', 'Default', 'chinese.srt')
    extract_ass_to_srt('bilingual.ass', 'Secondary', 'english.srt')

    # 4. ASS 文件格式化（处理重复 Style）
    format_ass_file('messy.ass', 'clean.ass')
    format_ass_file('input.ass')  # 原地格式化

    # 5. 时间戳同步（SRT -> ASS）
    sync_srt_timestamps_to_ass('bilingual.ass', 'corrected.srt', 'synced.ass')

    # 6. 获取支持的格式
    formats = get_supported_formats()
"""

import datetime
import logging

# 标准库导入
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .subtitle_entry import SubtitleEntry

# ==================== 工具类定义 ====================


class TimeUtils:
    """
    时间处理工具类

    提供各种字幕格式的时间戳解析、格式化和转换功能
    """

    # 时间格式正则表达式
    TIME_PATTERNS = {
        "srt": r"(\d{2}):(\d{2}):(\d{2}),(\d{3})",
        "ass": r"(\d+):(\d{2}):(\d{2})\.(\d{2})",
        "vtt": r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})",
        "lrc": r"\[(\d{2}):(\d{2})\.(\d{2})\]",
        "sbv": r"(\d+):(\d{2}):(\d{2})\.(\d{3})",
    }

    @staticmethod
    def parse_time(time_str: str, format_type: str) -> Optional[datetime.time]:
        """
        解析时间字符串

        Args:
            time_str: 时间字符串
            format_type: 时间格式类型 ('srt', 'ass', 'vtt', 'lrc', 'sbv')

        Returns:
            Optional[datetime.time]: 解析后的时间对象，失败时返回 None
        """
        try:
            pattern = TimeUtils.TIME_PATTERNS.get(format_type)
            if not pattern:
                return None

            match = re.match(pattern, time_str.strip())
            if not match:
                return None

            if format_type == "srt":
                hours, minutes, seconds, milliseconds = map(int, match.groups())
                microseconds = milliseconds * 1000
            elif format_type == "ass":
                hours, minutes, seconds, centiseconds = map(int, match.groups())
                microseconds = centiseconds * 10000
            elif format_type in ["vtt", "sbv"]:
                hours, minutes, seconds, milliseconds = map(int, match.groups())
                microseconds = milliseconds * 1000
            elif format_type == "lrc":
                minutes, seconds, centiseconds = map(int, match.groups())
                hours = 0
                microseconds = centiseconds * 10000
            else:
                return None

            return datetime.time(hours % 24, minutes, seconds, microseconds)

        except (ValueError, AttributeError):
            return None

    @staticmethod
    def format_time(time_obj: datetime.time, format_type: str) -> str:
        """
        格式化时间对象为字符串

        Args:
            time_obj: 时间对象
            format_type: 输出格式类型

        Returns:
            str: 格式化后的时间字符串
        """
        hours = time_obj.hour
        minutes = time_obj.minute
        seconds = time_obj.second
        microseconds = time_obj.microsecond

        if format_type == "srt":
            milliseconds = microseconds // 1000
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
        elif format_type == "ass":
            centiseconds = microseconds // 10000
            return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
        elif format_type in ["vtt", "sbv"]:
            milliseconds = microseconds // 1000
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        elif format_type == "lrc":
            centiseconds = microseconds // 10000
            total_minutes = hours * 60 + minutes
            return f"[{total_minutes:02d}:{seconds:02d}.{centiseconds:02d}]"
        else:
            return str(time_obj)

    @staticmethod
    def time_to_seconds(time_obj: datetime.time) -> float:
        """将时间对象转换为总秒数"""
        return (
            time_obj.hour * 3600
            + time_obj.minute * 60
            + time_obj.second
            + time_obj.microsecond / 1000000
        )

    @staticmethod
    def seconds_to_time(seconds: float) -> datetime.time:
        """将秒数转换为时间对象"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        microseconds = int((seconds % 1) * 1000000)
        return datetime.time(hours % 24, minutes, secs, microseconds)


class AudioUtils:
    """
    音频处理工具类

    提供音频时长获取等功能
    """

    @staticmethod
    def get_audio_duration(audio_file_path: str) -> float:
        """
        获取音频文件时长（秒）

        Args:
            audio_file_path: 音频文件路径

        Returns:
            float: 音频时长（秒）
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_file_path}")

        try:
            # 使用ffprobe获取音频时长
            cmd = [
                "ffprobe",
                "-i",
                audio_file_path,
                "-show_entries",
                "format=duration",
                "-v",
                "quiet",
                "-of",
                "csv=p=0",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            try:
                # 备用方案：使用ffmpeg
                cmd = ["ffmpeg", "-i", audio_file_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                output = result.stderr

                # 解析Duration行
                for line in output.split("\n"):
                    if "Duration" in line:
                        duration_str = line.split("Duration: ")[1].split(",")[0]
                        time_parts = duration_str.split(":")
                        duration = (
                            float(time_parts[0]) * 3600
                            + float(time_parts[1]) * 60
                            + float(time_parts[2])
                        )
                        return duration

                raise ValueError("无法解析音频时长")
            except Exception as e:
                raise RuntimeError(f"无法获取音频时长: {e}")


class TextUtils:
    """
    文本处理工具类

    提供各种字幕格式的文本清理和处理功能
    """

    @staticmethod
    def clean_ass_text(text: str) -> str:
        """
        清理 ASS 文本内容，移除格式标签

        Args:
            text: 原始文本

        Returns:
            str: 清理后的文本
        """
        if not text:
            return ""

        # 移除 ASS 格式标签 {\...}
        text = re.sub(r"\{[^}]*\}", "", text)

        # 移除换行符标记
        text = text.replace("\\N", " ")
        text = text.replace("\\n", " ")
        text = text.replace("\\h", " ")  # 硬空格

        # 移除多余的空格
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def clean_html_tags(text: str) -> str:
        """移除 HTML 标签"""
        if not text:
            return ""
        return re.sub(r"<[^>]*>", "", text).strip()

    @staticmethod
    def clean_vtt_tags(text: str) -> str:
        """清理 VTT 格式标签"""
        if not text:
            return ""

        # 移除 VTT 样式标签
        text = re.sub(r"<[^>]*>", "", text)
        text = re.sub(r"\{[^}]*\}", "", text)

        return text.strip()


# ==================== 主处理器类 ====================


class SubtitleProcessor:
    """
    字幕处理器主类

    提供完整的字幕文件处理功能，包括格式转换、时间戳同步、文件格式化等
    """

    # 支持的字幕格式
    SUPPORTED_FORMATS = {
        ".srt": "SubRip Text",
        ".ass": "Advanced SubStation Alpha",
        ".ssa": "SubStation Alpha",
        ".vtt": "WebVTT",
        ".sub": "MicroDVD/SubViewer",
        ".lrc": "LRC Lyrics",
        ".sbv": "YouTube SBV",
        ".smi": "SAMI",
        ".sami": "SAMI",
        ".ttml": "Timed Text Markup Language",
        ".dfxp": "Distribution Format Exchange Profile",
        ".txt": "Plain Text",
    }

    # 默认编码尝试顺序
    DEFAULT_ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin1"]

    def __init__(self) -> None:
        """初始化字幕处理器"""
        self.time_utils = TimeUtils()
        self.text_utils = TextUtils()
        self.audio_utils = AudioUtils()
        self.logger = logging.getLogger(__name__)

    def get_supported_formats(self) -> Dict[str, str]:
        """获取支持的格式列表"""
        return self.SUPPORTED_FORMATS.copy()

    def is_format_supported(self, file_path: str) -> bool:
        """检查文件格式是否支持"""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_FORMATS

    def detect_encoding(self, file_path: str) -> Optional[str]:
        """检测文件编码"""
        for encoding in self.DEFAULT_ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    f.read()
                return encoding
            except UnicodeDecodeError:
                continue
        return None

    def read_subtitle_file(self, file_path: str) -> List[SubtitleEntry]:
        """
        读取字幕文件

        Args:
            file_path: 字幕文件路径

        Returns:
            List[SubtitleEntry]: 字幕条目列表
        """
        if not os.path.exists(file_path):
            self.logger.error(f" 文件不存在: {file_path}")
            return []

        if not self.is_format_supported(file_path):
            self.logger.error(f" 不支持的文件格式: {Path(file_path).suffix}")
            return []

        encoding = self.detect_encoding(file_path)
        if not encoding:
            self.logger.error(f" 无法检测文件编码: {file_path}")
            return []

        try:
            ext = Path(file_path).suffix.lower()

            if ext == ".srt":
                return self._read_srt(file_path, encoding)
            elif ext in [".ass", ".ssa"]:
                return self._read_ass(file_path, encoding)
            elif ext == ".vtt":
                return self._read_vtt(file_path, encoding)
            elif ext == ".lrc":
                return self._read_lrc(file_path, encoding)
            elif ext == ".sbv":
                return self._read_sbv(file_path, encoding)
            elif ext in [".smi", ".sami"]:
                return self._read_sami(file_path, encoding)
            elif ext in [".ttml", ".dfxp"]:
                return self._read_ttml(file_path, encoding)
            elif ext == ".txt":
                return self._read_txt(file_path, encoding)
            else:
                self.logger.error(f" 暂不支持读取格式: {ext}")
                return []

        except Exception as e:
            self.logger.error(f" 读取文件失败: {str(e)}")
            return []

    def write_subtitle_file(
        self, subtitles: List[SubtitleEntry], file_path: str, format_type: str = None
    ) -> bool:
        """
        写入字幕文件

        Args:
            subtitles: 字幕条目列表
            file_path: 输出文件路径
            format_type: 强制指定格式类型

        Returns:
            bool: 写入是否成功
        """
        if not subtitles:
            self.logger.error(" 字幕列表为空")
            return False

        # 确保输出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 确定输出格式
        if format_type:
            ext = format_type if format_type.startswith(".") else f".{format_type}"
        else:
            ext = Path(file_path).suffix.lower()

        if ext not in self.SUPPORTED_FORMATS:
            self.logger.error(f" 不支持的输出格式: {ext}")
            return False

        try:
            if ext == ".srt":
                return self._write_srt(subtitles, file_path)
            elif ext in [".ass", ".ssa"]:
                return self._write_ass(subtitles, file_path)
            elif ext == ".vtt":
                return self._write_vtt(subtitles, file_path)
            elif ext == ".lrc":
                return self._write_lrc(subtitles, file_path)
            elif ext == ".sbv":
                return self._write_sbv(subtitles, file_path)
            elif ext in [".smi", ".sami"]:
                return self._write_sami(subtitles, file_path)
            elif ext in [".ttml", ".dfxp"]:
                return self._write_ttml(subtitles, file_path)
            elif ext == ".txt":
                return self._write_txt(subtitles, file_path)
            else:
                self.logger.error(f" 暂不支持写入格式: {ext}")
                return False

        except Exception as e:
            self.logger.error(f" 写入文件失败: {str(e)}")
            return False

    def convert_format(
        self, input_path: str, output_path: str, target_format: str = None
    ) -> bool:
        """
        转换字幕格式

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            target_format: 目标格式（可选）

        Returns:
            bool: 转换是否成功
        """
        self.logger.debug(
            f" 开始转换: {Path(input_path).name} -> {Path(output_path).name}"
        )

        # 读取输入文件
        subtitles = self.read_subtitle_file(input_path)
        if not subtitles:
            return False

        # 写入输出文件
        success = self.write_subtitle_file(subtitles, output_path, target_format)

        if success:
            self.logger.info(f"转换成功: {output_path}")
            self.logger.debug(f" 转换了 {len(subtitles)} 条字幕")

        return success

    # ==================== SRT 格式处理 ====================

    def _read_srt(self, file_path: str, encoding: str) -> List[SubtitleEntry]:
        """读取 SRT 格式文件"""
        subtitles = []

        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()

        # 分割字幕块
        blocks = content.strip().split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                try:
                    # 解析时间轴
                    time_line = lines[1]
                    if " --> " in time_line:
                        start_str, end_str = time_line.split(" --> ")
                        start_time = self.time_utils.parse_time(
                            start_str.strip(), "srt"
                        )
                        end_time = self.time_utils.parse_time(end_str.strip(), "srt")

                        if start_time and end_time:
                            text = "\n".join(lines[2:])
                            subtitles.append(
                                SubtitleEntry(
                                    start_time=start_time, end_time=end_time, text=text
                                )
                            )
                except Exception as e:
                    self.logger.warning(f" 跳过无效的 SRT 块: {str(e)}")
                    continue

        return subtitles

    def _write_srt(self, subtitles: List[SubtitleEntry], file_path: str) -> bool:
        """写入 SRT 格式文件"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for i, subtitle in enumerate(subtitles, 1):
                    # 序号
                    f.write(f"{i}\n")

                    # 时间轴
                    start_time = self.time_utils.format_time(subtitle.start_time, "srt")
                    end_time = self.time_utils.format_time(subtitle.end_time, "srt")
                    f.write(f"{start_time} --> {end_time}\n")

                    # 文本内容
                    f.write(f"{subtitle.text}\n")

                    # 空行分隔（最后一个除外）
                    if i < len(subtitles):
                        f.write("\n")

            return True

        except Exception as e:
            self.logger.error(f" 写入 SRT 文件失败: {str(e)}")
            return False

    # ==================== VTT 格式处理 ====================

    def _read_vtt(self, file_path: str, encoding: str) -> List[SubtitleEntry]:
        """读取 WebVTT 格式文件"""
        subtitles = []

        with open(file_path, "r", encoding=encoding) as f:
            lines = f.readlines()

        # 跳过 WEBVTT 头部
        start_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("WEBVTT"):
                start_index = i + 1
                break

        # 解析字幕块
        i = start_index
        while i < len(lines):
            line = lines[i].strip()

            # 跳过空行和注释
            if not line or line.startswith("NOTE"):
                i += 1
                continue

            # 检查是否是时间轴行
            if "-->" in line:
                try:
                    time_parts = line.split(" --> ")
                    if len(time_parts) == 2:
                        start_time = self.time_utils.parse_time(
                            time_parts[0].strip(), "vtt"
                        )
                        end_time = self.time_utils.parse_time(
                            time_parts[1].strip(), "vtt"
                        )

                        if start_time and end_time:
                            # 读取文本内容
                            text_lines = []
                            i += 1
                            while i < len(lines) and lines[i].strip():
                                text_lines.append(lines[i].strip())
                                i += 1

                            if text_lines:
                                text = "\n".join(text_lines)
                                # 清理 VTT 标签
                                text = self.text_utils.clean_vtt_tags(text)

                                subtitles.append(
                                    SubtitleEntry(
                                        start_time=start_time,
                                        end_time=end_time,
                                        text=text,
                                    )
                                )
                except Exception as e:
                    self.logger.warning(f" 跳过无效的 VTT 块: {str(e)}")

            i += 1

        return subtitles

    def _write_vtt(self, subtitles: List[SubtitleEntry], file_path: str) -> bool:
        """写入 WebVTT 格式文件"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                # 写入头部
                f.write("WEBVTT\n\n")

                for subtitle in subtitles:
                    # 时间轴
                    start_time = self.time_utils.format_time(subtitle.start_time, "vtt")
                    end_time = self.time_utils.format_time(subtitle.end_time, "vtt")
                    f.write(f"{start_time} --> {end_time}\n")

                    # 文本内容
                    f.write(f"{subtitle.text}\n\n")

            return True

        except Exception as e:
            self.logger.error(f" 写入 VTT 文件失败: {str(e)}")
            return False

    # ==================== ASS 格式处理 ====================

    def extract_ass_style_to_srt(
        self, ass_file_path: str, style_name: str, srt_output_path: str
    ) -> bool:
        """
        从 ASS 文件中提取指定 Style 的字幕并转换为 SRT 格式

        Args:
            ass_file_path: ASS 文件路径
            style_name: 要提取的 Style 名称
            srt_output_path: SRT 输出文件路径

        Returns:
            bool: 转换是否成功
        """
        try:
            if not os.path.exists(ass_file_path):
                self.logger.error(f" ASS 文件不存在: {ass_file_path}")
                return False

            encoding = self.detect_encoding(ass_file_path)
            if not encoding:
                self.logger.error(f" 无法检测文件编码: {ass_file_path}")
                return False

            with open(ass_file_path, "r", encoding=encoding) as f:
                content = f.read()

            # 解析指定 Style 的字幕
            subtitles = self._parse_ass_style(content, style_name)

            if not subtitles:
                self.logger.error(f" 未找到 Style '{style_name}' 的字幕内容")
                return False

            # 写入 SRT 文件
            success = self._write_srt(subtitles, srt_output_path)

            if success:
                self.logger.info(
                    f"成功提取 Style '{style_name}' 并转换为 SRT: {srt_output_path}"
                )
                self.logger.debug(f" 提取了 {len(subtitles)} 条字幕")

            return success

        except Exception as e:
            self.logger.error(f" 转换失败: {str(e)}")
            return False

    def get_ass_styles(self, ass_file_path: str) -> List[str]:
        """
        获取 ASS 文件中所有可用的 Style 名称

        Args:
            ass_file_path: ASS 文件路径

        Returns:
            List[str]: Style 名称列表
        """
        try:
            if not os.path.exists(ass_file_path):
                return []

            encoding = self.detect_encoding(ass_file_path)
            if not encoding:
                return []

            with open(ass_file_path, "r", encoding=encoding) as f:
                content = f.read()

            styles = set()
            lines = content.split("\n")
            in_events_section = False

            for line in lines:
                line = line.strip()

                if line == "[Events]":
                    in_events_section = True
                    continue

                if in_events_section and line.startswith("[") and line.endswith("]"):
                    break

                if in_events_section and line.startswith("Dialogue:"):
                    # 简单提取 Style 字段（通常是第4个字段）
                    parts = line.split(",")
                    if len(parts) >= 4:
                        style = parts[3].strip()
                        if style:
                            styles.add(style)

            return sorted(list(styles))

        except Exception as e:
            self.logger.error(f" 读取 ASS 文件失败: {str(e)}")
            return []

    def _parse_ass_style(self, content: str, target_style: str) -> List[SubtitleEntry]:
        """解析 ASS 文件中指定 Style 的字幕"""
        subtitles = []
        lines = content.split("\n")

        # 查找 [Events] 部分
        in_events_section = False
        format_line = None

        for line in lines:
            line = line.strip()

            if line == "[Events]":
                in_events_section = True
                continue

            if in_events_section and line.startswith("[") and line.endswith("]"):
                break

            if not in_events_section:
                continue

            # 获取格式行
            if line.startswith("Format:"):
                format_line = line[7:].strip()
                continue

            # 处理对话行
            if line.startswith("Dialogue:"):
                dialogue = self._parse_ass_dialogue(
                    line[9:].strip(), format_line, target_style
                )
                if dialogue:
                    subtitles.append(dialogue)

        return subtitles

    def _parse_ass_dialogue(
        self, dialogue_data: str, format_line: str, target_style: str
    ) -> Optional[SubtitleEntry]:
        """解析单行 ASS 对话数据"""
        if not format_line:
            return None

        try:
            # 解析格式
            format_fields = [field.strip() for field in format_line.split(",")]

            # 分割对话数据，注意 Text 字段可能包含逗号
            dialogue_parts = dialogue_data.split(",", len(format_fields) - 1)

            if len(dialogue_parts) != len(format_fields):
                return None

            # 创建字段映射
            dialogue_dict = {}
            for i, field in enumerate(format_fields):
                dialogue_dict[field] = dialogue_parts[i].strip()

            # 检查是否匹配目标 Style
            if dialogue_dict.get("Style", "").strip() != target_style:
                return None

            # 解析时间
            start_time = self.time_utils.parse_time(
                dialogue_dict.get("Start", ""), "ass"
            )
            end_time = self.time_utils.parse_time(dialogue_dict.get("End", ""), "ass")

            if not start_time or not end_time:
                return None

            # 清理文本内容
            text = self.text_utils.clean_ass_text(dialogue_dict.get("Text", ""))

            return SubtitleEntry(
                start_time=start_time,
                end_time=end_time,
                text=text,
                style=dialogue_dict.get("Style", ""),
                actor=dialogue_dict.get("Name", ""),
            )

        except Exception:
            return None

    # ==================== 简化的其他格式处理 ====================

    def _read_ass(self, file_path: str, encoding: str) -> List[SubtitleEntry]:
        """读取 ASS 文件（所有 Style）"""
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            # 获取所有 Style
            styles = self.get_ass_styles(file_path)
            all_subtitles = []

            # 为每个 Style 提取字幕
            for style in styles:
                subtitles = self._parse_ass_style(content, style)
                all_subtitles.extend(subtitles)

            # 按开始时间排序
            all_subtitles.sort(
                key=lambda x: self.time_utils.time_to_seconds(x.start_time)
            )

            return all_subtitles

        except Exception as e:
            self.logger.error(f" 读取 ASS 文件失败: {str(e)}")
            return []

    def _write_ass(self, subtitles: List[SubtitleEntry], file_path: str) -> bool:
        """写入 ASS 格式文件（简化版）"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                # 写入基本的 ASS 头部
                f.write("[Script Info]\n")
                f.write("Author: Leica\n")
                f.write("ScriptType: v4.00+\n\n")

                f.write("[V4+ Styles]\n")
                f.write(
                    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                )
                f.write(
                    "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n\n"
                )

                f.write("[Events]\n")
                f.write(
                    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                )

                for subtitle in subtitles:
                    start_time = self.time_utils.format_time(subtitle.start_time, "ass")
                    end_time = self.time_utils.format_time(subtitle.end_time, "ass")
                    style = subtitle.style or "Default"
                    actor = subtitle.actor or ""
                    text = subtitle.text.replace("\n", "\\N")  # ASS 换行符

                    f.write(
                        f"Dialogue: 0,{start_time},{end_time},{style},{actor},0,0,0,,{text}\n"
                    )

            return True

        except Exception as e:
            self.logger.error(f" 写入 ASS 文件失败: {str(e)}")
            return False

    def format_ass_file(self, input_ass_path: str, output_ass_path: str = None) -> bool:
        """
        格式化 ASS 文件，处理相同 Style 连续出现的情况

        当检测到上下两行使用相同 Style 时，会在下面那行的对应 Style 位置添加一个空的字幕行

        Args:
            input_ass_path: 输入的 ASS 文件路径
            output_ass_path: 输出的 ASS 文件路径（如果为 None，则覆盖原文件）

        Returns:
            bool: 格式化是否成功
        """
        try:
            # 检查输入文件是否存在
            if not os.path.exists(input_ass_path):
                self.logger.error(f" ASS 文件不存在: {input_ass_path}")
                return False

            # 如果没有指定输出路径，则覆盖原文件
            if output_ass_path is None:
                output_ass_path = input_ass_path

            # 检测编码并读取文件
            encoding = self.detect_encoding(input_ass_path)
            if not encoding:
                self.logger.error(f" 无法检测文件编码: {input_ass_path}")
                return False

            with open(input_ass_path, "r", encoding=encoding) as f:
                content = f.read()

            # 解析并格式化内容
            formatted_content = self._format_ass_content(content)

            if formatted_content is None:
                self.logger.error(" 格式化失败")
                return False

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_ass_path), exist_ok=True)

            # 写入格式化后的内容
            with open(output_ass_path, "w", encoding="utf-8-sig") as f:
                f.write(formatted_content)

            self.logger.info(f"ASS 文件格式化完成: {output_ass_path}")
            return True

        except Exception as e:
            self.logger.error(f" 格式化 ASS 文件失败: {str(e)}")
            return False

    def _format_ass_content(self, content: str) -> Optional[str]:
        """
        格式化 ASS 文件内容

        Args:
            content: ASS 文件内容

        Returns:
            Optional[str]: 格式化后的内容，失败时返回 None
        """
        try:
            lines = content.split("\n")
            formatted_lines = []

            # 查找 [Events] 部分
            in_events_section = False
            format_line = None
            dialogue_lines = []

            for line in lines:
                original_line = line
                line_stripped = line.strip()

                # 检查是否进入 Events 部分
                if line_stripped == "[Events]":
                    in_events_section = True
                    formatted_lines.append(original_line)
                    continue

                # 检查是否离开 Events 部分
                if (
                    in_events_section
                    and line_stripped.startswith("[")
                    and line_stripped.endswith("]")
                ):
                    # 处理收集到的对话行
                    if dialogue_lines:
                        processed_dialogues = self._process_dialogue_lines(
                            dialogue_lines, format_line
                        )
                        formatted_lines.extend(processed_dialogues)
                        dialogue_lines = []

                    in_events_section = False
                    formatted_lines.append(original_line)
                    continue

                if not in_events_section:
                    formatted_lines.append(original_line)
                    continue

                # 在 Events 部分内
                if line_stripped.startswith("Format:"):
                    format_line = line_stripped
                    formatted_lines.append(original_line)
                    continue

                # 收集对话行
                if line_stripped.startswith("Dialogue:"):
                    dialogue_lines.append(original_line)
                else:
                    formatted_lines.append(original_line)

            # 处理文件末尾的对话行
            if dialogue_lines:
                processed_dialogues = self._process_dialogue_lines(
                    dialogue_lines, format_line
                )
                formatted_lines.extend(processed_dialogues)

            return "\n".join(formatted_lines)

        except Exception as e:
            self.logger.error(f" 格式化内容时出错: {str(e)}")
            return None

    def _process_dialogue_lines(
        self, dialogue_lines: List[str], format_line: str
    ) -> List[str]:
        """
        处理对话行，添加必要的空行来分隔相同 Style

        Args:
            dialogue_lines: 对话行列表
            format_line: 格式行

        Returns:
            List[str]: 处理后的对话行列表
        """
        if not dialogue_lines or not format_line:
            return dialogue_lines

        # 解析格式行获取字段位置
        format_fields = [
            field.strip() for field in format_line[7:].split(",")
        ]  # 移除 "Format: "

        try:
            style_index = format_fields.index("Style")
        except ValueError:
            self.logger.warning(" 未找到 Style 字段，跳过格式化")
            return dialogue_lines

        # 提取所有 Style 信息
        dialogue_info = []
        for line in dialogue_lines:
            if line.strip().startswith("Dialogue:"):
                dialogue_data = line.strip()[9:].strip()  # 移除 "Dialogue: "
                parts = dialogue_data.split(",", len(format_fields) - 1)

                if len(parts) > style_index:
                    style = parts[style_index].strip()
                    dialogue_info.append({"line": line, "style": style, "parts": parts})
                else:
                    dialogue_info.append(
                        {"line": line, "style": "Unknown", "parts": parts}
                    )

        # 检查是否有多个不同的 Style
        unique_styles = set(info["style"] for info in dialogue_info)

        if len(unique_styles) <= 1:
            self.logger.info(f" 只有单个 Style ({unique_styles}), 无需格式化")
            return dialogue_lines

        self.logger.debug(
            f"检测到 {len(unique_styles)} 个不同的 Style: {unique_styles}"
        )

        # 处理相同 Style 连续出现的情况 - 合并内容而不是添加空行
        processed_lines = []
        i = 0

        while i < len(dialogue_info):
            current_info = dialogue_info[i]

            # 检查下一行是否存在且为相同 Style
            if i < len(dialogue_info) - 1:
                next_info = dialogue_info[i + 1]

                if (
                    current_info["style"] == next_info["style"]
                    and current_info["style"] != "Unknown"
                ):
                    # 找到相同 Style 连续出现的情况
                    self.logger.debug(
                        f"检测到连续的 Style '{current_info['style']}', 合并内容"
                    )

                    # 找到对应的不同 Style 行来合并重复行的内容
                    target_style = None
                    for other_style in unique_styles:
                        if (
                            other_style != current_info["style"]
                            and other_style != "Unknown"
                        ):
                            target_style = other_style
                            break

                    if target_style:
                        # 查找前面是否有对应的 target_style 行可以合并
                        target_line_index = -1
                        for j in range(len(processed_lines) - 1, -1, -1):
                            line = processed_lines[j]
                            if line.strip().startswith("Dialogue:"):
                                parts = line.strip()[9:].split(
                                    ",", len(format_fields) - 1
                                )
                                if (
                                    len(parts) > style_index
                                    and parts[style_index].strip() == target_style
                                ):
                                    target_line_index = j
                                    break

                        if target_line_index >= 0:
                            # 找到了可以合并的目标行
                            target_line = processed_lines[target_line_index]
                            target_parts = target_line.strip()[9:].split(
                                ",", len(format_fields) - 1
                            )

                            try:
                                # 获取字段索引
                                start_index = format_fields.index("Start")
                                end_index = format_fields.index("End")
                                text_index = format_fields.index("Text")

                                # 扩展目标行的结束时间到重复行的结束时间
                                target_parts[end_index] = next_info["parts"][end_index]

                                # 合并文本内容
                                original_text = (
                                    target_parts[text_index]
                                    if target_parts[text_index]
                                    else ""
                                )
                                repeat_text = (
                                    next_info["parts"][text_index]
                                    if next_info["parts"][text_index]
                                    else ""
                                )

                                if original_text and repeat_text:
                                    merged_text = f"{original_text}, {repeat_text}"
                                elif repeat_text:
                                    merged_text = repeat_text
                                else:
                                    merged_text = original_text

                                target_parts[text_index] = merged_text

                                # 重新构建目标行
                                merged_line = "Dialogue: " + ",".join(target_parts)
                                processed_lines[target_line_index] = merged_line

                                self.logger.debug(f"合并到 Style '{target_style}' 行")
                                self.logger.debug(
                                    f"扩展时间戳到: {next_info['parts'][end_index]}"
                                )
                                self.logger.debug("合并文本内容")

                                # 同时扩展当前行的结束时间戳
                                current_info["parts"][end_index] = next_info["parts"][
                                    end_index
                                ]
                                current_info["line"] = "Dialogue: " + ",".join(
                                    current_info["parts"]
                                )
                                self.logger.debug(
                                    f"同时扩展当前行时间戳到: {next_info['parts'][end_index]}"
                                )

                            except ValueError as e:
                                self.logger.warning(f"字段解析错误: {e}")
                                processed_lines.append(current_info["line"])
                        else:
                            # 没有找到可合并的目标行，保持原样
                            self.logger.warning(
                                f"未找到可合并的 '{target_style}' 行，保持原样"
                            )
                            processed_lines.append(current_info["line"])
                    else:
                        # 没有其他 Style 可用，保持原样
                        processed_lines.append(current_info["line"])

                    # 添加当前行（已扩展时间戳）
                    processed_lines.append(current_info["line"])

                    # 跳过下一行（重复行），因为已经被合并了
                    i += 2
                    continue

            # 没有重复，正常添加
            processed_lines.append(current_info["line"])
            i += 1

        return processed_lines

    def sync_srt_timestamps_to_ass(
        self,
        ass_file_path: str,
        srt_file_path: str,
        output_ass_path: str = None,
        reference_style: str = "Default",
    ) -> bool:
        """
        将 SRT 文件的时间戳同步到 ASS 文件中

        适用场景：ASS 文件是双语多 Style 的，SRT 是从 ASS 的某个 Style 提取的单行版本，
        经过时间戳处理后，需要将新的时间戳同步回 ASS 文件的所有对应行。

        Args:
            ass_file_path: 原始 ASS 文件路径
            srt_file_path: 包含新时间戳的 SRT 文件路径
            output_ass_path: 输出 ASS 文件路径（如果为 None，则覆盖原文件）
            reference_style: 参考的 Style 名称，用于匹配对应关系（默认 "Default"）

        Returns:
            bool: 同步是否成功
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(ass_file_path):
                self.logger.error(f" ASS 文件不存在: {ass_file_path}")
                return False

            if not os.path.exists(srt_file_path):
                self.logger.error(f" SRT 文件不存在: {srt_file_path}")
                return False

            # 如果没有指定输出路径，则覆盖原文件
            if output_ass_path is None:
                output_ass_path = ass_file_path

            self.logger.debug(
                f" 开始同步时间戳: {os.path.basename(srt_file_path)} -> {os.path.basename(output_ass_path)}"
            )

            # 读取 SRT 文件的时间戳
            srt_timestamps = self._read_srt_timestamps(srt_file_path)
            if not srt_timestamps:
                self.logger.error(" 无法读取 SRT 时间戳")
                return False

            self.logger.debug(f" 读取到 {len(srt_timestamps)} 个 SRT 时间戳")

            # 读取并处理 ASS 文件
            success = self._update_ass_timestamps(
                ass_file_path, srt_timestamps, output_ass_path, reference_style
            )

            if success:
                self.logger.info(f"时间戳同步成功: {output_ass_path}")
                return True
            else:
                self.logger.error(" 时间戳同步失败")
                return False

        except Exception as e:
            self.logger.error(f" 同步时间戳失败: {str(e)}")
            return False

    def _read_srt_timestamps(self, srt_file_path: str) -> List[Dict]:
        """
        读取 SRT 文件的时间戳信息

        Args:
            srt_file_path: SRT 文件路径

        Returns:
            List[Dict]: 时间戳信息列表，每个元素包含 start_time, end_time, text
        """
        timestamps = []

        try:
            encoding = self.detect_encoding(srt_file_path)
            if not encoding:
                return []

            with open(srt_file_path, "r", encoding=encoding) as f:
                content = f.read()

            # 分割字幕块
            blocks = content.strip().split("\n\n")

            for block in blocks:
                lines = block.strip().split("\n")
                if len(lines) >= 3:
                    try:
                        # 解析时间轴
                        time_line = lines[1]
                        if " --> " in time_line:
                            start_str, end_str = time_line.split(" --> ")
                            start_time = self.time_utils.parse_time(
                                start_str.strip(), "srt"
                            )
                            end_time = self.time_utils.parse_time(
                                end_str.strip(), "srt"
                            )

                            if start_time and end_time:
                                text = "\n".join(lines[2:])
                                timestamps.append(
                                    {
                                        "start_time": start_time,
                                        "end_time": end_time,
                                        "text": text.strip(),
                                        "start_str": self.time_utils.format_time(
                                            start_time, "ass"
                                        ),
                                        "end_str": self.time_utils.format_time(
                                            end_time, "ass"
                                        ),
                                    }
                                )
                    except Exception as e:
                        self.logger.warning(f" 跳过无效的 SRT 块: {str(e)}")
                        continue

            return timestamps

        except Exception as e:
            self.logger.error(f" 读取 SRT 时间戳失败: {str(e)}")
            return []

    def _update_ass_timestamps(
        self,
        ass_file_path: str,
        srt_timestamps: List[Dict],
        output_path: str,
        reference_style: str,
    ) -> bool:
        """
        更新 ASS 文件的时间戳

        Args:
            ass_file_path: ASS 文件路径
            srt_timestamps: SRT 时间戳列表
            output_path: 输出文件路径
            reference_style: 参考 Style 名称

        Returns:
            bool: 更新是否成功
        """
        try:
            # 检测编码并读取 ASS 文件
            encoding = self.detect_encoding(ass_file_path)
            if not encoding:
                return False

            with open(ass_file_path, "r", encoding=encoding) as f:
                content = f.read()

            lines = content.split("\n")
            updated_lines = []

            # 查找 [Events] 部分并处理对话行
            in_events_section = False
            format_line = None
            srt_index = 0
            current_group_start_time = None

            for line in lines:
                original_line = line
                line_stripped = line.strip()

                # 检查是否进入 Events 部分
                if line_stripped == "[Events]":
                    in_events_section = True
                    updated_lines.append(original_line)
                    continue

                # 检查是否离开 Events 部分
                if (
                    in_events_section
                    and line_stripped.startswith("[")
                    and line_stripped.endswith("]")
                ):
                    in_events_section = False
                    updated_lines.append(original_line)
                    continue

                if not in_events_section:
                    updated_lines.append(original_line)
                    continue

                # 在 Events 部分内
                if line_stripped.startswith("Format:"):
                    format_line = line_stripped
                    updated_lines.append(original_line)
                    continue

                # 处理对话行
                if line_stripped.startswith("Dialogue:"):
                    # 检查是否是新的时间组
                    current_start_time = self._extract_start_time(
                        line_stripped, format_line
                    )

                    # 如果是新的开始时间，移动到下一个 SRT 时间戳
                    if (
                        current_start_time
                        and current_start_time != current_group_start_time
                        and srt_index < len(srt_timestamps)
                    ):
                        current_group_start_time = current_start_time
                        # 只有当这是参考 Style 时才增加索引
                        if self._is_reference_style_line(
                            line_stripped, format_line, reference_style
                        ):
                            pass  # 保持当前索引
                        else:
                            # 如果不是参考 Style，检查是否需要增加索引
                            pass

                    # 更新时间戳
                    if srt_index < len(srt_timestamps):
                        updated_line = self._update_dialogue_timestamp(
                            original_line,
                            format_line,
                            reference_style,
                            srt_timestamps,
                            [],
                            srt_index,
                        )
                    else:
                        updated_line = original_line

                    # 如果这是参考 Style 的行，增加 SRT 索引
                    if self._is_reference_style_line(
                        line_stripped, format_line, reference_style
                    ):
                        srt_index += 1

                    updated_lines.append(updated_line)
                else:
                    updated_lines.append(original_line)

            # 写入更新后的内容
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8-sig") as f:
                f.write("\n".join(updated_lines))

            self.logger.debug(f" 同步了 {srt_index} 个时间戳组")
            return True

        except Exception as e:
            self.logger.error(f" 更新 ASS 时间戳失败: {str(e)}")
            return False

    def _update_dialogue_timestamp(
        self,
        dialogue_line: str,
        format_line: str,
        reference_style: str,
        srt_timestamps: List[Dict],
        reference_dialogues: List,
        current_srt_index: int,
    ) -> str:
        """
        更新单行对话的时间戳

        Args:
            dialogue_line: 对话行
            format_line: 格式行
            reference_style: 参考 Style
            srt_timestamps: SRT 时间戳列表
            reference_dialogues: 参考对话列表
            current_srt_index: 当前 SRT 索引

        Returns:
            str: 更新后的对话行
        """
        try:
            if not format_line or current_srt_index >= len(srt_timestamps):
                return dialogue_line

            # 解析格式行
            format_fields = [field.strip() for field in format_line[7:].split(",")]

            # 分割对话数据
            dialogue_data = dialogue_line.strip()[9:].strip()  # 移除 "Dialogue: "
            dialogue_parts = dialogue_data.split(",", len(format_fields) - 1)

            if len(dialogue_parts) != len(format_fields):
                return dialogue_line

            try:
                # 获取字段索引
                start_index = format_fields.index("Start")
                end_index = format_fields.index("End")
                style_index = format_fields.index("Style")

                # 获取当前行的 Style
                current_style = dialogue_parts[style_index].strip()

                # 获取当前 SRT 时间戳
                srt_timestamp = srt_timestamps[current_srt_index]

                # 简化逻辑：更新所有行的时间戳到当前 SRT 时间戳
                # 这样可以确保同一时间组的所有 Style 都有相同的时间戳

                # 更新时间戳
                dialogue_parts[start_index] = srt_timestamp["start_str"]
                dialogue_parts[end_index] = srt_timestamp["end_str"]

                # 重新构建对话行
                updated_line = "Dialogue: " + ",".join(dialogue_parts)

                self.logger.debug(
                    f"更新 [{current_style:9s}] 时间戳到: {srt_timestamp['start_str']} -> {srt_timestamp['end_str']}"
                )
                return updated_line

            except ValueError as e:
                # 字段不存在，返回原行
                self.logger.warning(f" 字段解析错误: {e}")
                return dialogue_line

        except Exception as e:
            self.logger.warning(f" 更新对话时间戳失败: {str(e)}")
            return dialogue_line

    def _extract_start_time(
        self, dialogue_line: str, format_line: str
    ) -> Optional[str]:
        """
        提取对话行的开始时间

        Args:
            dialogue_line: 对话行
            format_line: 格式行

        Returns:
            Optional[str]: 开始时间字符串
        """
        try:
            if not format_line:
                return None

            # 解析格式行
            format_fields = [field.strip() for field in format_line[7:].split(",")]

            # 分割对话数据
            dialogue_data = dialogue_line[9:].strip()  # 移除 "Dialogue: "
            dialogue_parts = dialogue_data.split(",", len(format_fields) - 1)

            if len(dialogue_parts) != len(format_fields):
                return None

            try:
                start_index = format_fields.index("Start")
                return dialogue_parts[start_index]
            except ValueError:
                return None

        except Exception:
            return None

    def _is_reference_style_line(
        self, dialogue_line: str, format_line: str, reference_style: str
    ) -> bool:
        """
        检查是否是参考 Style 的行

        Args:
            dialogue_line: 对话行
            format_line: 格式行
            reference_style: 参考 Style

        Returns:
            bool: 是否是参考 Style 的行
        """
        try:
            if not format_line:
                return False

            # 解析格式行
            format_fields = [field.strip() for field in format_line[7:].split(",")]

            # 分割对话数据
            dialogue_data = dialogue_line[9:].strip()  # 移除 "Dialogue: "
            dialogue_parts = dialogue_data.split(",", len(format_fields) - 1)

            if len(dialogue_parts) != len(format_fields):
                return False

            try:
                style_index = format_fields.index("Style")
                current_style = dialogue_parts[style_index].strip()
                return current_style == reference_style
            except ValueError:
                return False

        except Exception:
            return False

    def _timestamps_match_group(
        self, start_time: str, end_time: str, reference_dialogues: List, srt_index: int
    ) -> bool:
        """
        检查时间戳是否与当前组匹配

        这个方法用于判断非参考 Style 的行是否应该与当前 SRT 时间戳同步
        策略：检查时间戳是否在合理的范围内

        Args:
            start_time: 开始时间
            end_time: 结束时间
            reference_dialogues: 参考对话列表
            srt_index: SRT 索引

        Returns:
            bool: 是否匹配当前组
        """
        # 简化策略：如果开始时间相同或非常接近，则认为是同一组
        # 这里可以根据实际需求调整匹配逻辑

        try:
            # 解析当前时间戳
            current_start = self.time_utils.parse_time(start_time, "ass")
            if not current_start:
                return False

            # 如果有参考对话记录，检查是否在同一时间组
            if reference_dialogues and srt_index < len(reference_dialogues):
                ref_start = reference_dialogues[srt_index].get("start_time")
                if ref_start:
                    # 计算时间差（秒）
                    current_seconds = self.time_utils.time_to_seconds(current_start)
                    ref_seconds = self.time_utils.time_to_seconds(ref_start)
                    time_diff = abs(current_seconds - ref_seconds)

                    # 如果时间差小于 2 秒，认为是同一组
                    return time_diff < 2.0

            return False

        except Exception:
            return False

    # ==================== 字幕重新生成功能 ====================

    def regenerate_subtitles_from_audio(
        self,
        original_srt_path: str,
        audio_segments_dir: str,
        output_srt_path: str,
        audio_file_pattern: str = "{index}.wav",
        strategy: str = "adaptive",
    ) -> bool:
        """
        根据原字幕和TTS生成的音频片段重新生成字幕

        这是核心功能：根据原字幕文件和对应的TTS音频片段，
        重新计算时间轴并生成新的字幕文件

        Args:
            original_srt_path: 原始字幕文件路径
            audio_segments_dir: TTS生成的音频片段目录
            output_srt_path: 输出字幕文件路径
            audio_file_pattern: 音频文件命名模式，{index}会被替换为字幕序号
            strategy: 重新生成策略
                - "proportional": 等比例缩放
                - "cumulative": 累积时间调整
                - "gap_preserving": 间隙保持
                - "adaptive": 自适应混合策略（推荐）

        Returns:
            bool: 是否成功
        """
        try:
            self.logger.info(f" 开始字幕重新生成...")
            self.logger.debug(f"原始字幕: {original_srt_path}")
            self.logger.debug(f"音频目录: {audio_segments_dir}")
            self.logger.debug(f"输出文件: {output_srt_path}")
            self.logger.debug(f"使用策略: {strategy}")

            # 1. 读取原始字幕
            self.logger.debug("读取原始字幕...")
            original_subtitles = self.read_subtitle_file(original_srt_path)
            if not original_subtitles:
                self.logger.error(" 无法读取原始字幕文件")
                return False

            self.logger.debug(f"读取到 {len(original_subtitles)} 条字幕")

            # 2. 获取音频片段信息
            self.logger.debug("获取音频片段信息...")
            audio_info = self._collect_audio_segments_info(
                original_subtitles, audio_segments_dir, audio_file_pattern
            )

            if not audio_info:
                self.logger.error(" 无法获取音频片段信息")
                return False

            self.logger.debug(f"获取到 {len(audio_info)} 个音频片段信息")

            # 3. 重新生成字幕
            self.logger.debug("重新计算时间轴...")
            new_subtitles = self._regenerate_timeline(
                original_subtitles, audio_info, strategy
            )

            if not new_subtitles:
                self.logger.error(" 时间轴重新计算失败")
                return False

            # 4. 写入新字幕文件
            self.logger.debug("写入新字幕文件...")
            success = self.write_subtitle_file(new_subtitles, output_srt_path, ".srt")

            if success:
                self.logger.info("字幕重新生成成功!")
                self.logger.debug(f"生成了 {len(new_subtitles)} 条新字幕")

                # 显示统计信息
                self._print_regeneration_stats(
                    original_subtitles, new_subtitles, audio_info
                )

            return success

        except Exception as e:
            self.logger.error(f" 字幕重新生成失败: {str(e)}")
            return False

    def _collect_audio_segments_info(
        self,
        original_subtitles: List[SubtitleEntry],
        audio_segments_dir: str,
        audio_file_pattern: str,
    ) -> List[Dict]:
        """
        收集音频片段信息

        Args:
            original_subtitles: 原始字幕列表
            audio_segments_dir: 音频片段目录
            audio_file_pattern: 音频文件命名模式

        Returns:
            List[Dict]: 音频信息列表，每个元素包含 index, file_path, duration, exists
        """
        audio_info = []

        for i, subtitle in enumerate(original_subtitles, 1):
            # 构建音频文件路径
            audio_filename = audio_file_pattern.format(index=i)
            audio_file_path = os.path.join(audio_segments_dir, audio_filename)

            info = {
                "index": i,
                "file_path": audio_file_path,
                "exists": os.path.exists(audio_file_path),
                "duration": 0.0,
                "original_duration": subtitle.duration_seconds(),
            }

            if info["exists"]:
                try:
                    info["duration"] = self.audio_utils.get_audio_duration(
                        audio_file_path
                    )
                    self.logger.debug(
                        f"  📄 {i:3d}: {audio_filename} -> {info['duration']:.2f}s (原:{info['original_duration']:.2f}s)"
                    )
                except Exception as e:
                    self.logger.debug(
                        f"  ⚠️ {i:3d}: {audio_filename} -> 无法获取时长: {e}"
                    )
                    info["duration"] = info[
                        "original_duration"
                    ]  # 使用原始时长作为fallback
            else:
                self.logger.debug(
                    f"  ❌ {i:3d}: {audio_filename} -> 文件不存在，使用原始时长"
                )
                info["duration"] = info["original_duration"]  # 使用原始时长作为fallback

            audio_info.append(info)

        return audio_info

    def _regenerate_timeline(
        self,
        original_subtitles: List[SubtitleEntry],
        audio_info: List[Dict],
        strategy: str,
    ) -> List[SubtitleEntry]:
        """
        重新生成时间轴

        Args:
            original_subtitles: 原始字幕列表
            audio_info: 音频信息列表
            strategy: 重新生成策略

        Returns:
            List[SubtitleEntry]: 新的字幕列表
        """
        if strategy == "proportional":
            return self._proportional_scaling(original_subtitles, audio_info)
        elif strategy == "cumulative":
            return self._cumulative_adjustment(original_subtitles, audio_info)
        elif strategy == "gap_preserving":
            return self._gap_preserving_adjustment(original_subtitles, audio_info)
        elif strategy == "adaptive":
            return self._adaptive_strategy(original_subtitles, audio_info)
        else:
            self.logger.warning(f" 未知策略 '{strategy}'，使用默认策略 'adaptive'")
            return self._adaptive_strategy(original_subtitles, audio_info)

    def _proportional_scaling(
        self, original_subtitles: List[SubtitleEntry], audio_info: List[Dict]
    ) -> List[SubtitleEntry]:
        """
        策略1: 等比例缩放
        根据总时长变化等比例调整所有时间戳
        """
        self.logger.debug("📐 使用等比例缩放策略")

        # 计算原始总时长和新总时长
        original_total = sum(sub.duration_seconds() for sub in original_subtitles)
        new_total = sum(info["duration"] for info in audio_info)

        if original_total == 0:
            self.logger.error(" 原始总时长为0，无法进行等比例缩放")
            return []

        scale_factor = new_total / original_total
        self.logger.debug(
            f" 缩放因子: {scale_factor:.3f} (原始:{original_total:.1f}s -> 新:{new_total:.1f}s)"
        )

        new_subtitles = []
        current_time = 0.0

        for i, (subtitle, info) in enumerate(zip(original_subtitles, audio_info)):
            # 使用实际音频时长
            duration = info["duration"]

            start_time = self.time_utils.seconds_to_time(current_time)
            end_time = self.time_utils.seconds_to_time(current_time + duration)

            new_subtitles.append(
                SubtitleEntry(
                    start_time=start_time,
                    end_time=end_time,
                    text=subtitle.text,
                    style=subtitle.style,
                    actor=subtitle.actor,
                )
            )

            current_time += duration

            # 添加间隙（按比例缩放）
            if i < len(original_subtitles) - 1:
                next_original_start = original_subtitles[i + 1].start_time_seconds()
                current_original_end = subtitle.end_time_seconds()
                original_gap = next_original_start - current_original_end

                if original_gap > 0:
                    scaled_gap = original_gap * scale_factor
                    current_time += scaled_gap

        return new_subtitles

    def _cumulative_adjustment(
        self, original_subtitles: List[SubtitleEntry], audio_info: List[Dict]
    ) -> List[SubtitleEntry]:
        """
        策略2: 累积时间调整
        逐个调整字幕时间，保持原始间隙
        """
        self.logger.debug("⏰ 使用累积时间调整策略")

        # 计算原始间隙
        original_gaps = []
        for i in range(len(original_subtitles) - 1):
            current_end = original_subtitles[i].end_time_seconds()
            next_start = original_subtitles[i + 1].start_time_seconds()
            gap = max(0, next_start - current_end)
            original_gaps.append(gap)

        new_subtitles = []
        current_time = 0.0

        for i, (subtitle, info) in enumerate(zip(original_subtitles, audio_info)):
            duration = info["duration"]

            start_time = self.time_utils.seconds_to_time(current_time)
            end_time = self.time_utils.seconds_to_time(current_time + duration)

            new_subtitles.append(
                SubtitleEntry(
                    start_time=start_time,
                    end_time=end_time,
                    text=subtitle.text,
                    style=subtitle.style,
                    actor=subtitle.actor,
                )
            )

            current_time += duration

            # 添加原始间隙
            if i < len(original_gaps):
                current_time += original_gaps[i]

        return new_subtitles

    def _gap_preserving_adjustment(
        self, original_subtitles: List[SubtitleEntry], audio_info: List[Dict]
    ) -> List[SubtitleEntry]:
        """
        策略3: 间隙保持
        保持原始字幕间的时间间隙比例
        """
        self.logger.debug("🔄 使用间隙保持策略")

        # 计算间隙调整因子
        original_speech_total = sum(
            sub.duration_seconds() for sub in original_subtitles
        )
        new_speech_total = sum(info["duration"] for info in audio_info)

        if original_speech_total == 0:
            gap_scale_factor = 1.0
        else:
            gap_scale_factor = new_speech_total / original_speech_total

        self.logger.debug(f" 间隙缩放因子: {gap_scale_factor:.3f}")

        # 计算原始间隙
        original_gaps = []
        for i in range(len(original_subtitles) - 1):
            current_end = original_subtitles[i].end_time_seconds()
            next_start = original_subtitles[i + 1].start_time_seconds()
            gap = max(0, next_start - current_end)
            original_gaps.append(gap)

        new_subtitles = []
        current_time = 0.0

        for i, (subtitle, info) in enumerate(zip(original_subtitles, audio_info)):
            duration = info["duration"]

            start_time = self.time_utils.seconds_to_time(current_time)
            end_time = self.time_utils.seconds_to_time(current_time + duration)

            new_subtitles.append(
                SubtitleEntry(
                    start_time=start_time,
                    end_time=end_time,
                    text=subtitle.text,
                    style=subtitle.style,
                    actor=subtitle.actor,
                )
            )

            current_time += duration

            # 添加调整后的间隙
            if i < len(original_gaps):
                adjusted_gap = original_gaps[i] * gap_scale_factor
                current_time += adjusted_gap

        return new_subtitles

    def _adaptive_strategy(
        self, original_subtitles: List[SubtitleEntry], audio_info: List[Dict]
    ) -> List[SubtitleEntry]:
        """
        策略4: 自适应混合策略
        根据时长变化程度选择最佳策略
        """
        self.logger.debug("🤖 使用自适应混合策略")

        # 分析时长变化特征
        duration_changes = []
        for subtitle, info in zip(original_subtitles, audio_info):
            original_dur = subtitle.duration_seconds()
            new_dur = info["duration"]
            if original_dur > 0:
                change_ratio = new_dur / original_dur
                duration_changes.append(change_ratio)

        if not duration_changes:
            self.logger.warning(" 无法分析时长变化，使用累积调整策略")
            return self._cumulative_adjustment(original_subtitles, audio_info)

        avg_change = sum(duration_changes) / len(duration_changes)
        change_variance = sum((x - avg_change) ** 2 for x in duration_changes) / len(
            duration_changes
        )

        self.logger.debug(f" 平均变化比例: {avg_change:.3f}")
        self.logger.debug(f" 变化方差: {change_variance:.3f}")

        # 根据变化特征选择策略
        if change_variance < 0.1:  # 变化较均匀
            if 0.8 <= avg_change <= 1.2:  # 总体变化不大
                self.logger.debug("🎯 选择策略: 间隙保持 (变化均匀且幅度小)")
                return self._gap_preserving_adjustment(original_subtitles, audio_info)
            else:  # 总体变化较大但均匀
                self.logger.debug("🎯 选择策略: 等比例缩放 (变化均匀但幅度大)")
                return self._proportional_scaling(original_subtitles, audio_info)
        else:  # 变化不均匀
            self.logger.debug("🎯 选择策略: 累积调整 (变化不均匀)")
            return self._cumulative_adjustment(original_subtitles, audio_info)

    def _print_regeneration_stats(
        self,
        original_subtitles: List[SubtitleEntry],
        new_subtitles: List[SubtitleEntry],
        audio_info: List[Dict],
    ):
        """
        打印字幕重新生成的统计信息
        """
        self.logger.debug("\n📊 字幕重新生成统计信息:")
        self.logger.debug("=" * 50)

        # 总时长对比
        original_total = sum(sub.duration_seconds() for sub in original_subtitles)
        new_total = sum(sub.duration_seconds() for sub in new_subtitles)
        audio_total = sum(info["duration"] for info in audio_info)

        self.logger.debug(f"⏱️  原始总时长: {original_total:.1f}s")
        self.logger.debug(f"⏱️  音频总时长: {audio_total:.1f}s")
        self.logger.debug(f"⏱️  新字幕总时长: {new_total:.1f}s")
        self.logger.debug(
            f"📈 时长变化: {((new_total / original_total - 1) * 100):+.1f}%"
        )

        # 时长变化分析
        duration_changes = []
        for i, (orig, new, info) in enumerate(
            zip(original_subtitles, new_subtitles, audio_info)
        ):
            orig_dur = orig.duration_seconds()
            new_dur = new.duration_seconds()
            audio_dur = info["duration"]

            if orig_dur > 0:
                change_ratio = new_dur / orig_dur
                duration_changes.append(change_ratio)

        if duration_changes:
            avg_change = sum(duration_changes) / len(duration_changes)
            max_change = max(duration_changes)
            min_change = min(duration_changes)

            self.logger.info(f"平均变化比例: {avg_change:.3f}")
            self.logger.info(f"最大变化比例: {max_change:.3f}")
            self.logger.info(f"最小变化比例: {min_change:.3f}")

        # 音频文件统计
        existing_files = sum(1 for info in audio_info if info["exists"])
        missing_files = len(audio_info) - existing_files

        self.logger.debug(f"📁 音频文件: {existing_files}/{len(audio_info)} 存在")
        if missing_files > 0:
            self.logger.warning(f"  缺失文件: {missing_files} 个 (使用原始时长)")

        self.logger.debug("=" * 50)

    # ==================== 其他格式的简化处理 ====================

    def _read_lrc(self, file_path: str, encoding: str) -> List[SubtitleEntry]:
        """读取 LRC 格式文件"""
        subtitles = []
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line.startswith("[") and "]" in line:
                    # 提取时间和文本
                    time_match = re.match(r"\[(\d{2}):(\d{2})\.(\d{2})\](.*)", line)
                    if time_match:
                        minutes, seconds, centiseconds, text = time_match.groups()
                        start_time = self.time_utils.parse_time(
                            f"[{minutes}:{seconds}.{centiseconds}]", "lrc"
                        )
                        if start_time and text.strip():
                            # LRC 通常没有结束时间，估算 3 秒持续时间
                            end_seconds = (
                                self.time_utils.time_to_seconds(start_time) + 3.0
                            )
                            end_time = self.time_utils.seconds_to_time(end_seconds)

                            subtitles.append(
                                SubtitleEntry(
                                    start_time=start_time,
                                    end_time=end_time,
                                    text=text.strip(),
                                )
                            )
        except Exception as e:
            self.logger.error(f" 读取 LRC 文件失败: {str(e)}")

        return subtitles

    def _write_lrc(self, subtitles: List[SubtitleEntry], file_path: str) -> bool:
        """写入 LRC 格式文件"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for subtitle in subtitles:
                    time_str = self.time_utils.format_time(subtitle.start_time, "lrc")
                    f.write(f"{time_str}{subtitle.text}\n")
            return True
        except Exception as e:
            self.logger.error(f" 写入 LRC 文件失败: {str(e)}")
            return False

    def _read_sbv(self, file_path: str, encoding: str) -> List[SubtitleEntry]:
        """读取 SBV 格式文件"""
        subtitles = []
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            blocks = content.strip().split("\n\n")
            for block in blocks:
                lines = block.strip().split("\n")
                if len(lines) >= 2:
                    time_line = lines[0]
                    if "," in time_line:
                        start_str, end_str = time_line.split(",")
                        start_time = self.time_utils.parse_time(
                            start_str.strip(), "sbv"
                        )
                        end_time = self.time_utils.parse_time(end_str.strip(), "sbv")

                        if start_time and end_time:
                            text = "\n".join(lines[1:])
                            subtitles.append(
                                SubtitleEntry(
                                    start_time=start_time, end_time=end_time, text=text
                                )
                            )
        except Exception as e:
            self.logger.error(f" 读取 SBV 文件失败: {str(e)}")

        return subtitles

    def _write_sbv(self, subtitles: List[SubtitleEntry], file_path: str) -> bool:
        """写入 SBV 格式文件"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for subtitle in subtitles:
                    start_time = self.time_utils.format_time(subtitle.start_time, "sbv")
                    end_time = self.time_utils.format_time(subtitle.end_time, "sbv")
                    f.write(f"{start_time},{end_time}\n{subtitle.text}\n\n")
            return True
        except Exception as e:
            self.logger.error(f" 写入 SBV 文件失败: {str(e)}")
            return False

    # 其他格式的占位符方法
    def _read_sami(self, file_path: str, encoding: str) -> List[SubtitleEntry]:
        """读取 SAMI 格式文件（简化实现）"""
        self.logger.warning(" SAMI 格式支持有限，建议转换为其他格式")
        return []

    def _write_sami(self, subtitles: List[SubtitleEntry], file_path: str) -> bool:
        """写入 SAMI 格式文件（简化实现）"""
        self.logger.warning(" SAMI 格式支持有限，建议使用其他格式")
        return False

    def _read_ttml(self, file_path: str, encoding: str) -> List[SubtitleEntry]:
        """读取 TTML 格式文件（简化实现）"""
        self.logger.warning(" TTML 格式支持有限，建议转换为其他格式")
        return []

    def _write_ttml(self, subtitles: List[SubtitleEntry], file_path: str) -> bool:
        """写入 TTML 格式文件（简化实现）"""
        self.logger.warning(" TTML 格式支持有限，建议使用其他格式")
        return False

    def _read_txt(self, file_path: str, encoding: str) -> List[SubtitleEntry]:
        """读取纯文本格式文件"""
        subtitles = []
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.readlines()

            # 简单处理：每行作为一个字幕，时间间隔 3 秒
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    start_seconds = i * 3.0
                    end_seconds = start_seconds + 3.0
                    start_time = self.time_utils.seconds_to_time(start_seconds)
                    end_time = self.time_utils.seconds_to_time(end_seconds)

                    subtitles.append(
                        SubtitleEntry(
                            start_time=start_time, end_time=end_time, text=line
                        )
                    )
        except Exception as e:
            self.logger.error(f" 读取 TXT 文件失败: {str(e)}")

        return subtitles

    def _write_txt(self, subtitles: List[SubtitleEntry], file_path: str) -> bool:
        """写入纯文本格式文件"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for subtitle in subtitles:
                    f.write(f"{subtitle.text}\n")
            return True
        except Exception as e:
            self.logger.error(f" 写入 TXT 文件失败: {str(e)}")
            return False


# ==================== 便捷函数 ====================


def convert_subtitle(
    input_file: str, output_file: str, target_format: str = None
) -> bool:
    """
    便捷函数：字幕格式转换

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        target_format: 目标格式（可选）

    Returns:
        bool: 转换是否成功
    """
    processor = SubtitleProcessor()
    return processor.convert_format(input_file, output_file, target_format)


def extract_ass_to_srt(
    ass_file: str, output_srt: str, style_name: str = "Default"
) -> bool:
    """
    便捷函数：从 ASS 文件提取指定 Style 并转换为 SRT

    Args:
        ass_file: ASS 文件路径
        style_name: Style 名称
        output_srt: 输出 SRT 文件路径

    Returns:
        bool: 转换是否成功
    """
    processor = SubtitleProcessor()
    return processor.extract_ass_style_to_srt(ass_file, style_name, output_srt)


def get_available_styles(ass_file: str) -> List[str]:
    """
    便捷函数：获取 ASS 文件中的所有 Style 名称

    Args:
        ass_file: ASS 文件路径

    Returns:
        List[str]: Style 名称列表
    """
    processor = SubtitleProcessor()
    return processor.get_ass_styles(ass_file)


def get_supported_formats() -> Dict[str, str]:
    """
    便捷函数：获取支持的字幕格式

    Returns:
        Dict[str, str]: 格式扩展名到描述的映射
    """
    processor = SubtitleProcessor()
    return processor.get_supported_formats()


def format_ass_file(input_ass: str, output_ass: str = None) -> bool:
    """
    便捷函数：格式化 ASS 文件，处理相同 Style 连续出现的情况

    当检测到连续相同 Style 时，会将重复行的内容合并到前面对应的不同 Style 行中，
    并扩展该行的结束时间戳，然后删除重复行。

    Args:
        input_ass: 输入 ASS 文件路径
        output_ass: 输出 ASS 文件路径（如果为 None，则覆盖原文件）

    Returns:
        bool: 格式化是否成功
    """
    processor = SubtitleProcessor()
    return processor.format_ass_file(input_ass, output_ass)


def sync_srt_timestamps_to_ass(
    ass_file: str,
    srt_file: str,
    output_ass: str = None,
    reference_style: str = "Default",
) -> bool:
    """
    便捷函数：将 SRT 文件的时间戳同步到 ASS 文件中

    适用场景：ASS 文件是双语多 Style 的，SRT 是从 ASS 的某个 Style 提取的单行版本，
    经过时间戳处理后，需要将新的时间戳同步回 ASS 文件的所有对应行。

    Args:
        ass_file: 原始 ASS 文件路径
        srt_file: 包含新时间戳的 SRT 文件路径
        output_ass: 输出 ASS 文件路径（如果为 None，则覆盖原文件）
        reference_style: 参考的 Style 名称，用于匹配对应关系（默认 "Default"）

    Returns:
        bool: 同步是否成功
    """
    processor = SubtitleProcessor()
    return processor.sync_srt_timestamps_to_ass(
        ass_file, srt_file, output_ass, reference_style
    )


def regenerate_subtitles_from_audio(
    original_srt_path: str,
    audio_segments_dir: str,
    output_srt_path: str,
    audio_file_pattern: str = "{index}.wav",
    strategy: str = "adaptive",
) -> bool:
    """
    便捷函数：根据原字幕和TTS生成的音频片段重新生成字幕

    这是核心功能：根据原字幕文件和对应的TTS音频片段，
    重新计算时间轴并生成新的字幕文件

    Args:
        original_srt_path: 原始字幕文件路径
        audio_segments_dir: TTS生成的音频片段目录
        output_srt_path: 输出字幕文件路径
        audio_file_pattern: 音频文件命名模式，{index}会被替换为字幕序号
        strategy: 重新生成策略
            - "proportional": 等比例缩放
            - "cumulative": 累积时间调整
            - "gap_preserving": 间隙保持
            - "adaptive": 自适应混合策略（推荐）

    Returns:
        bool: 是否成功

    Example:
        success = regenerate_subtitles_from_audio(
            "original.srt",
            "audio_segments/",
            "new_subtitles.srt",
            "{index}.wav",
            "adaptive"
        )
    """
    processor = SubtitleProcessor()
    return processor.regenerate_subtitles_from_audio(
        original_srt_path,
        audio_segments_dir,
        output_srt_path,
        audio_file_pattern,
        strategy,
    )
