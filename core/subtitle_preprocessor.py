"""
字幕预处理器
专注于核心功能：根据字幕路径预处理字幕
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .subtitle.subtitle_entry import SubtitleEntry
from .subtitle.subtitle_processor import convert_subtitle, TextUtils, format_ass_file, extract_ass_to_srt
from .subtitle.text_processor import IntelligentTextProcessor, quick_clean_text, SplitStrategy


class SubtitlePreprocessor:
    """字幕预处理器，专注于字幕预处理和清理"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 初始化智能文本处理器
        self.text_processor = IntelligentTextProcessor.create_quality_processor()

    def preprocess_subtitle(self, subtitle_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        根据字幕路径预处理字幕
        
        Args:
            subtitle_path: 字幕文件路径
            output_dir: 输出目录，默认为字幕文件所在目录
            
        Returns:
            包含预处理结果的字典：
            {
                'success': bool,
                'processed_subtitle_path': str,  # 预处理后的字幕文件路径
                'subtitle_entries': List[SubtitleEntry],  # 字幕条目列表
                'total_entries': int,  # 字幕条目总数
                'total_duration': float,  # 字幕总时长（秒）
                'format': str,  # 字幕格式
                'error': str  # 错误信息（如果有）
            }
        """
        try:
            # 验证输入文件
            if not os.path.exists(subtitle_path):
                return {'success': False, 'error': f'字幕文件不存在: {subtitle_path}'}

            # 设置输出目录
            if output_dir is None:
                output_dir = Path(subtitle_path).parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

            subtitle_name = Path(subtitle_path).stem
            subtitle_format = Path(subtitle_path).suffix.lower()

            self.logger.info(f"开始预处理字幕: {subtitle_path}")

            # 1. 检测并转换字幕格式（统一转换为SRT格式）
            processed_subtitle_path = output_dir / f"{subtitle_name}_processed.srt"
            format_ass_subtitle_path = output_dir / f"{subtitle_name}_formatted.ass"

            if subtitle_format == '.ass':
                success = format_ass_file(subtitle_path, str(format_ass_subtitle_path))
                if success:
                    success = extract_ass_to_srt(str(format_ass_subtitle_path), str(processed_subtitle_path))
                if not success:
                    return {'success': False, 'error': 'ass字幕格式转换失败'}
            elif subtitle_format != '.srt':
                # 转换为SRT格式
                success = convert_subtitle(subtitle_path, str(processed_subtitle_path), 'srt')
                if not success:
                    return {'success': False, 'error': '字幕格式转换失败'}
                self.logger.debug(f"字幕格式转换完成: {subtitle_format} -> .srt")
            else:
                # 直接复制SRT文件并进行清理
                self._copy_and_clean_srt(subtitle_path, str(processed_subtitle_path))

            # 2. 解析字幕条目
            subtitle_entries = self._parse_srt_file(str(processed_subtitle_path))

            # 3. 清理和验证字幕条目（使用智能文本处理器）
            cleaned_entries = self._clean_subtitle_entries_with_ai(subtitle_entries)

            # 4. 重新写入清理后的字幕
            self._write_srt_file(str(processed_subtitle_path), cleaned_entries)

            # 5. 计算统计信息
            total_entries = len(cleaned_entries)
            total_duration = 0.0
            if cleaned_entries:
                total_duration = max(entry.end_time_seconds() for entry in cleaned_entries)

            self.logger.info(f"字幕预处理完成: {total_entries}条字幕, 总时长{total_duration:.1f}秒")

            return {
                'success': True,
                'processed_subtitle_path': str(processed_subtitle_path),
                'subtitle_entries': cleaned_entries,
                'total_entries': total_entries,
                'total_duration': total_duration,
                'format': 'srt',
                'text_processing_stats': self.text_processor.get_processing_statistics()  # 添加文本处理统计
            }

        except Exception as e:
            self.logger.error(f"字幕预处理失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _copy_and_clean_srt(self, source_path: str, target_path: str):
        """复制并清理SRT文件"""
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用智能文本处理器进行基本清理
        cleaned_content = quick_clean_text(content)

        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

    def _parse_srt_file(self, srt_path: str) -> List[SubtitleEntry]:
        """解析SRT文件为字幕条目列表"""
        entries = []

        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 分割字幕块
            subtitle_blocks = re.split(r'\n\s*\n', content.strip())

            for block in subtitle_blocks:
                if not block.strip():
                    continue

                lines = block.strip().split('\n')
                if len(lines) < 3:
                    continue

                # 解析时间戳
                time_line = lines[1]
                time_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})', time_line)

                if not time_match:
                    continue

                # 创建时间对象
                start_h, start_m, start_s, start_ms = map(int, time_match.groups()[:4])
                end_h, end_m, end_s, end_ms = map(int, time_match.groups()[4:])

                start_time = self._create_time(start_h, start_m, start_s, start_ms)
                end_time = self._create_time(end_h, end_m, end_s, end_ms)

                # 合并文本行并使用智能处理器清理
                text = '\n'.join(lines[2:])
                text = quick_clean_text(text)  # 使用智能文本处理器进行快速清理

                # 创建字幕条目（包括空文本条目，保持索引连续性）
                entry = SubtitleEntry(
                    start_time=start_time,
                    end_time=end_time,
                    text=text,  # 可能为空字符串
                    style="Default"
                )
                entries.append(entry)

        except Exception as e:
            self.logger.error(f"SRT文件解析失败: {str(e)}")

        return entries

    def _clean_subtitle_entries_with_ai(self, entries: List[SubtitleEntry]) -> List[SubtitleEntry]:
        """使用AI智能文本处理器清理和验证字幕条目"""
        cleaned_entries = []
        
        for entry in entries:
            # 验证时间戳
            if entry.start_time_seconds() >= entry.end_time_seconds():
                self.logger.warning(f"跳过无效时间戳的字幕: {entry.text[:20]}...")
                continue
                
            final_text = ""  # 默认为空文本
            
            # 只有当原文本不为空时，才进行AI处理
            if entry.text and entry.text.strip():
                # 调试信息：显示处理前的文本
                self.logger.debug(f"处理前文本: '{entry.text}'")
                
                # 使用智能文本处理器进行深度清理
                processing_result = self.text_processor.process(
                    entry.text,
                    split_strategy=SplitStrategy.SENTENCE  # 按句子分割，适合字幕
                )
                
                # 调试信息：显示处理结果
                self.logger.debug(f"AI处理结果 - is_valid: {processing_result.is_valid}")
                self.logger.debug(f"AI处理结果 - cleaned_text: '{processing_result.cleaned_text}'")
                self.logger.debug(f"AI处理结果 - segments: {processing_result.segments}")
                
                # 检查处理结果
                if processing_result.is_valid and processing_result.cleaned_text.strip():
                    # 如果文本被分割成多个片段，只取第一个片段（保持字幕完整性）
                    final_text = processing_result.segments[0] if processing_result.segments else processing_result.cleaned_text
                    final_text = final_text.strip()
                    self.logger.debug(f"AI处理后最终文本: '{final_text}'")
                else:
                    # AI处理无效或为空，手动检查是否为纯括号内容
                    if self._is_pure_bracket_content(entry.text):
                        final_text = ""
                        self.logger.debug(f"检测到纯括号内容，设为空: '{entry.text}' -> '{final_text}'")
                    else:
                        final_text = ""
                        self.logger.debug(f"AI处理后文本为空: '{entry.text}' -> '{final_text}'")
            else:
                self.logger.debug(f"保留空文本字幕条目，索引位置: {len(cleaned_entries) + 1}")
                
            # 创建清理后的条目（包括空文本条目，保持索引连续性）
            cleaned_entry = SubtitleEntry(
                start_time=entry.start_time,
                end_time=entry.end_time,
                text=final_text,  # 可能为空字符串
                style=entry.style,
                actor=entry.actor
            )
            cleaned_entries.append(cleaned_entry)
            
            # 调试信息：显示处理前后的文本变化
            if entry.text.strip() != final_text.strip():
                self.logger.debug(f"文本变化: '{entry.text}' -> '{final_text}'")
            
        # 按时间排序
        cleaned_entries.sort(key=lambda x: x.start_time_seconds())
        
        return cleaned_entries
        
    def _is_pure_bracket_content(self, text: str) -> bool:
        """检查文本是否为纯括号内容"""
        if not text or not text.strip():
            return False
            
        text = text.strip()
        
        # 检查是否完全被各种括号包围
        bracket_patterns = [
            (r'^\([^)]*\)$', '圆括号'),           # (内容)
            (r'^\[[^\]]*\]$', '方括号'),          # [内容]  
            (r'^\{[^}]*\}$', '花括号'),           # {内容}
            (r'^<[^>]*>$', '尖括号'),             # <内容>
            (r'^（[^）]*）$', '中文圆括号'),       # （内容）
            (r'^【[^】]*】$', '中文方括号'),       # 【内容】
        ]
        
        for pattern, bracket_type in bracket_patterns:
            if re.match(pattern, text):
                self.logger.debug(f"检测到{bracket_type}包围的纯括号内容: '{text}'")
                return True
                
        return False

    @staticmethod
    def _clean_text_content(text: str) -> str:
        """清理文本内容"""
        if not text:
            return ""

        # 移除HTML标签
        text = re.sub(r'<[^>]*>', '', text)

        # 移除ASS样式标签
        text = TextUtils.clean_ass_text(text)

        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # 移除特殊字符（保留中文、英文、数字、基本标点）
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef.,!?;:\'\"()-]', '', text)

        return text

    def _write_srt_file(self, srt_path: str, entries: List[SubtitleEntry]):
        """写入SRT文件"""
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(entries, 1):
                # 写入序号
                f.write(f"{i}\n")

                # 写入时间戳
                start_time_str = self._format_srt_time(entry.start_time)
                end_time_str = self._format_srt_time(entry.end_time)
                f.write(f"{start_time_str} --> {end_time_str}\n")

                # 写入文本
                f.write(f"{entry.text}\n\n")

    @staticmethod
    def _create_time(hour: int, minute: int, second: int, millisecond: int):
        """创建时间对象"""
        import datetime
        microsecond = millisecond * 1000
        return datetime.time(hour, minute, second, microsecond)

    @staticmethod
    def _format_srt_time(time_obj) -> str:
        """格式化SRT时间"""
        return f"{time_obj.hour:02d}:{time_obj.minute:02d}:{time_obj.second:02d},{time_obj.microsecond // 1000:03d}"

    def clear_cache(self):
        """清理文本处理器缓存"""
        self.text_processor.clear_cache()

    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.text_processor.get_processing_statistics()
