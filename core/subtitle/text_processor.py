"""
智能文本预处理器 - 专业的TTS文本预处理解决方案

提供完整的文本预处理功能，专为TTS（文本转语音）系统设计：

主要功能:
1. 基础清理 - 移除问题字符和边界情况处理
2. 深度标准化 - 术语替换和多语言规范化
3. 智能分割 - 多种分割策略自适应选择
4. 语言检测 - 智能多语言识别
5. 批量处理 - 支持大规模文本处理

特性:
- 高性能的正则表达式优化
- 灵活的分割策略选择
- 详细的处理统计信息
- 支持自定义配置和扩展
"""

import logging
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class SplitStrategy(Enum):
    """文本分割策略"""

    SENTENCE = "sentence"  # 按句子分割
    LENGTH = "length"  # 按长度分割
    PUNCTUATION = "punctuation"  # 按标点分割
    ADAPTIVE = "adaptive"  # 自适应混合策略


class LanguageType(Enum):
    """语言类型"""

    CHINESE = "zh"
    ENGLISH = "en"
    JAPANESE = "ja"
    KOREAN = "ko"
    FRENCH = "fr"
    SPANISH = "es"
    GERMAN = "de"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class TextProcessingResult:
    """文本处理结果"""

    original_text: str  # 原始文本
    cleaned_text: str  # 清理后文本
    segments: List[str]  # 分割后的片段
    language: LanguageType  # 检测到的语言
    text_hash: str  # 文本哈希
    metadata: Dict[str, Any]  # 元数据信息
    is_valid: bool  # 是否有效
    processing_time: float = 0.0  # 处理耗时（秒）
    error_message: Optional[str] = None  # 错误信息

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["language"] = self.language.value
        return result


class IntelligentTextProcessor:
    """
    智能文本预处理器

    专业的TTS文本预处理解决方案，提供高效、可靠的文本处理功能。
    """

    # 问题字符映射
    PROBLEMATIC_CHARS = {
        "&": "和",
        "®": "",
        "™": "",
        "©": "",
        "℃": "摄氏度",
        "℉": "华氏度",
        "€": "欧元",
        "$": "美元",
        "£": "英镑",
        "¥": "人民币",
        "@": "艾特",
        "#": "井号",
        # 注意：百分号(%)现在由_process_percentage()方法智能处理
    }

    # 术语替换映射
    TERM_REPLACEMENTS = {
        "AI": "人工智能",
        "API": "应用程序接口",
        "URL": "网址",
        "CPU": "中央处理器",
        "GPU": "图形处理器",
        "iPhone": "苹果手机",
        "iPad": "苹果平板",
        "Android": "安卓",
        "Windows": "视窗系统",
        "macOS": "苹果系统",
    }

    # 标点符号映射
    PUNCTUATION_MAP = {
        "...": "。",
        "…": "。",
        "!": "！",
        "?": "？",
        ":": "：",
        ";": "；",
        ",": "，",
        ".": "。",
        '"': '"',
        "'": "'",
        "(": "（",
        ")": "）",
    }

    def __init__(
        self,
        max_text_length: int = 200,
        default_split_strategy: SplitStrategy = SplitStrategy.ADAPTIVE,
    ):
        """初始化智能文本处理器"""
        # 基础配置
        self.max_text_length = max_text_length
        self.default_split_strategy = default_split_strategy

        # 编译正则表达式模式
        self._compile_regex_patterns()

        # 初始化日志记录器
        self.logger = logging.getLogger(__name__)

        # 性能统计
        self._processing_stats = defaultdict(int)

    def _compile_regex_patterns(self):
        """编译常用的正则表达式模式（性能优化）"""
        # 基础字符模式
        self.chinese_pattern = re.compile(r"[\u4e00-\u9fff]+")
        self.english_pattern = re.compile(r"[a-zA-Z]+")
        self.whitespace_pattern = re.compile(r"\s+")

        # 文本标准化模式
        self.uppercase_pattern = re.compile(r"(?<!^)([A-Z])")
        self.alphanumeric_pattern = re.compile(
            r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])"
        )

        # 语言检测模式
        self.japanese_pattern = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]+")
        self.korean_pattern = re.compile(r"[\uac00-\ud7af]+")

        # 清理模式
        self.punctuation_only_pattern = re.compile(r"^[^\w\s]*$")
        self.single_digit_pattern = re.compile(r"\b\d\b")

        # 括号内容删除模式
        self.parentheses_pattern = re.compile(r"\([^)]*\)")  # 圆括号
        self.square_brackets_pattern = re.compile(r"\[[^\]]*\]")  # 方括号
        self.curly_brackets_pattern = re.compile(r"\{[^}]*\}")  # 花括号
        self.angle_brackets_pattern = re.compile(r"<[^>]*>")  # 尖括号
        self.chinese_brackets_pattern = re.compile(r"（[^）]*）")  # 中文圆括号
        self.chinese_square_brackets_pattern = re.compile(r"【[^】]*】")  # 中文方括号

        # 英文缩写模式
        self.contractions_pattern = re.compile(
            r"\b(don't|won't|can't|n't|'re|'ve|'ll|'d|'m)\b", re.IGNORECASE
        )

    def clean_text_for_tts(self, text: str) -> str:
        """简化的文本清理接口（兼容现有代码）"""
        result = self.process(text)
        return result.cleaned_text if result.is_valid else ""

    def validate_text(self, text: str) -> Tuple[bool, Optional[str]]:
        """文本验证接口（兼容现有代码）"""
        if not text or not text.strip():
            return False, "文本为空"
        if len(text) > self.max_text_length * 10:
            return False, f"文本过长: {len(text)} 字符"
        return True, None

    def process(
        self,
        text: str,
        split_strategy: Optional[SplitStrategy] = None,
        target_language: Optional[LanguageType] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> TextProcessingResult:
        """
        智能文本处理主入口

        Args:
            text: 原始文本
            split_strategy: 分割策略
            target_language: 目标语言
            progress_callback: 进度回调函数

        Returns:
            TextProcessingResult: 处理结果
        """
        start_time = time.time()

        def report_progress(stage: str, progress: float):
            if progress_callback:
                progress_callback(stage, progress)

        try:
            # 1. 基础验证
            report_progress("验证输入", 0.1)
            if not text or not text.strip():
                return TextProcessingResult(
                    original_text=text,
                    cleaned_text="",
                    segments=[],
                    language=LanguageType.UNKNOWN,
                    text_hash="",
                    metadata={},
                    is_valid=False,
                    processing_time=time.time() - start_time,
                    error_message="文本为空",
                )

            # 2. 语言检测
            report_progress("检测语言", 0.2)
            detected_language = target_language or self._detect_language(text)

            # 3. 基础清理
            report_progress("基础清理", 0.3)
            cleaned_text = self._basic_cleaning(text)

            # 4. 深度标准化
            report_progress("深度标准化", 0.5)
            normalized_text = self._deep_normalization(cleaned_text, detected_language)

            # 5. 智能分割
            report_progress("智能分割", 0.7)
            split_strategy = split_strategy or self.default_split_strategy
            segments = self._intelligent_split(
                normalized_text, split_strategy, detected_language
            )

            # 6. 验证结果
            report_progress("验证结果", 0.8)
            is_valid, error_msg = self._validate_result(normalized_text, segments)

            # 7. 构建结果
            processing_time = time.time() - start_time
            result = TextProcessingResult(
                original_text=text,
                cleaned_text=normalized_text,
                segments=segments,
                language=detected_language,
                text_hash="",
                metadata={
                    "original_length": len(text),
                    "cleaned_length": len(normalized_text),
                    "segment_count": len(segments),
                    "split_strategy": split_strategy.value,
                    "processing_steps": [
                        "basic_cleaning",
                        "deep_normalization",
                        "intelligent_split",
                    ],
                },
                is_valid=is_valid,
                processing_time=processing_time,
                error_message=error_msg,
            )

            # 更新统计信息
            self._processing_stats["total_processed"] += 1
            self._processing_stats["total_processing_time"] += processing_time

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"文本处理异常: {str(e)}")
            self._processing_stats["total_errors"] += 1

            return TextProcessingResult(
                original_text=text,
                cleaned_text="",
                segments=[],
                language=LanguageType.UNKNOWN,
                text_hash="",
                metadata={"error_type": type(e).__name__},
                is_valid=False,
                processing_time=processing_time,
                error_message=f"处理异常: {str(e)}",
            )

    def _detect_language(self, text: str) -> LanguageType:
        """智能语言检测"""
        if not text or not text.strip():
            return LanguageType.UNKNOWN

        text_stripped = text.strip()
        total_chars = len(text_stripped)

        # 使用预编译的正则表达式进行高效检测
        chinese_matches = self.chinese_pattern.findall(text_stripped)
        english_matches = self.english_pattern.findall(text_stripped)
        japanese_matches = self.japanese_pattern.findall(text_stripped)
        korean_matches = self.korean_pattern.findall(text_stripped)

        # 计算字符数量
        chinese_chars = sum(len(match) for match in chinese_matches)
        english_chars = sum(len(match) for match in english_matches)
        japanese_chars = sum(len(match) for match in japanese_matches)
        korean_chars = sum(len(match) for match in korean_matches)

        # 计算比例
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        english_ratio = english_chars / total_chars if total_chars > 0 else 0

        # 语言判断逻辑
        if korean_chars > 0:
            return LanguageType.KOREAN
        elif japanese_chars > 0:
            return LanguageType.JAPANESE
        elif chinese_ratio > 0.3:
            if english_ratio > 0.2:
                return LanguageType.MIXED
            return LanguageType.CHINESE
        elif english_chars > 0:
            return LanguageType.ENGLISH
        else:
            return LanguageType.UNKNOWN

    def _basic_cleaning(self, text: str) -> str:
        """基础清理（包含删除括号内容）"""
        if not text:
            return ""

        # 🚨 修复编码问题：首先移除音乐相关emoji和可能导致GBK编码问题的字符
        original_text = text
        text = self._remove_problematic_unicode_chars(text)

        # 1. 删除各种括号内容（TTS不需要括号内的注释、说明等）
        text = self._remove_brackets_content(text)

        # 2. 批量替换问题字符
        for char, replacement in self.PROBLEMATIC_CHARS.items():
            if char in text:
                text = text.replace(char, replacement)

        # 3. 处理边界情况
        if self.punctuation_only_pattern.match(text.strip()):
            return ""

        stripped_text = text.strip()
        if len(stripped_text) <= 1:
            return ""

        # 4. 空白字符清理
        text = self.whitespace_pattern.sub(" ", stripped_text)

        # 5. 最终Unicode清理确保
        text = self._remove_problematic_unicode_chars(text)

        return text

    def _remove_brackets_content(self, text: str) -> str:
        """
        删除括号内容

        删除各种类型的括号及其内容，因为TTS通常不需要朗读括号内的注释、说明等内容

        Args:
            text: 原始文本

        Returns:
            删除括号内容后的文本
        """
        if not text:
            return ""

        # 按优先级删除各种括号内容
        # 1. 删除圆括号内容 (注释)
        text = self.parentheses_pattern.sub("", text)

        # 2. 删除方括号内容 [备注]
        text = self.square_brackets_pattern.sub("", text)

        # 3. 删除花括号内容 {说明}
        text = self.curly_brackets_pattern.sub("", text)

        # 4. 删除尖括号内容 <标签>
        text = self.angle_brackets_pattern.sub("", text)

        # 5. 删除中文圆括号内容 （注释）
        text = self.chinese_brackets_pattern.sub("", text)

        # 6. 删除中文方括号内容 【备注】
        text = self.chinese_square_brackets_pattern.sub("", text)

        # 清理多余的空白字符
        text = self.whitespace_pattern.sub(" ", text).strip()

        return text

    def _remove_problematic_unicode_chars(self, text: str) -> str:
        """
        移除可能导致GBK编码问题的Unicode字符

        主要移除音乐相关emoji和其他可能的问题字符
        """
        if not text:
            return ""

        import re

        # 🚨 关键修复：扩大Unicode字符清理范围，确保彻底移除问题字符
        # 移除所有可能导致GBK编码问题的Unicode字符
        problematic_unicode_pattern = re.compile(
            r"[\U0001F000-\U0001F9FF"  # 全部Emoji区域
            r"\U00002600-\U000026FF"  # 杂项符号
            r"\U00002700-\U000027BF"  # 装饰符号
            r"\U0000FE00-\U0000FE0F"  # 变体选择符
            r"\U0001F1E0-\U0001F1FF"  # 区域指示符
            r"\U0000200D"  # 零宽连接符
            r"\U0000FE0F"  # 变体选择符-16
            r"\U0000FE0E"  # 变体选择符-15
            r"\U0001F3FB-\U0001F3FF"  # 肤色修饰符
            r"\U0001FA70-\U0001FAFF"  # 扩展符号和象形文字
            r"\U00010000-\U0001FFFF]"  # 所有补充平面字符
        )

        # 移除匹配的字符
        cleaned_text = problematic_unicode_pattern.sub("", text)

        # 额外的安全清理：逐字符检查并移除所有非基本多文种平面的字符
        safe_chars = []
        for char in cleaned_text:
            char_code = ord(char)
            # 只保留基本多文种平面的字符 (U+0000 到 U+FFFF)
            # 排除代理对区域 (U+D800 到 U+DFFF)
            if char_code <= 0xFFFF and not (0xD800 <= char_code <= 0xDFFF):
                # 再次检查是否为GBK可编码字符
                try:
                    char.encode("gbk")
                    safe_chars.append(char)
                except UnicodeEncodeError:
                    # 如果GBK无法编码则跳过
                    continue
            # 跳过所有高位Unicode字符

        result = "".join(safe_chars)

        # 确保文本使用UTF-8编码但兼容GBK
        try:
            result = result.encode("utf-8", errors="ignore").decode("utf-8")
        except Exception:
            pass  # 如果编码转换失败，使用原文本

        return result.strip()

    def _gentle_bracket_removal(self, text: str) -> str:
        """
        温和的括号移除策略

        只移除明显的注释性括号内容，保留可能包含重要信息的内容
        """
        if not text:
            return ""

        import re

        original_text = text

        # 首先移除Unicode问题字符
        text = self._remove_problematic_unicode_chars(text)

        # 只移除明显的注释性内容
        # 移除翻译标注 (字幕翻译：xxx)
        text = re.sub(r"\(字幕翻译[：:][^)]*\)", "", text)
        # 移除说明性内容 (说明：xxx)、(注：xxx)、(备注：xxx)
        text = re.sub(r"\([说注备][明注：][：:][^)]*\)", "", text)
        # 移除单元预览标注
        text = re.sub(r"\(单元[^)]*预览[^)]*\)", "", text)

        # 清理空白
        text = self.whitespace_pattern.sub(" ", text).strip()

        # 如果清理后仍然为空，从原文本中提取有意义的内容
        if not text:
            # 从原文本中提取非括号内容
            cleaned_original = self._remove_problematic_unicode_chars(original_text)
            # 移除括号内容但保留外部内容
            no_brackets = re.sub(r"[\(（][^\)）]*[\)）]", " ", cleaned_original)
            text = self.whitespace_pattern.sub(" ", no_brackets).strip()

        # 最终Unicode清理
        text = self._remove_problematic_unicode_chars(text)

        return text

    def _deep_normalization(self, text: str, language: LanguageType) -> str:
        """深度标准化"""
        if not text:
            return ""

        # 术语替换
        for term, replacement in self.TERM_REPLACEMENTS.items():
            if term in text:
                text = text.replace(term, replacement)

        # 大写字母前插入空格
        text = self.uppercase_pattern.sub(r" \1", text)

        # 字母数字间插入空格
        text = self.alphanumeric_pattern.sub(" ", text)

        # 破折号智能处理（在语言特定处理之前，避免数字被转换）
        text = self._process_dash(text)

        # 百分比智能处理（在数字转换之前处理）
        text = self._process_percentage(text)

        # 语言特定处理
        if language in (LanguageType.CHINESE, LanguageType.MIXED):
            text = self._chinese_normalization(text)
        elif language == LanguageType.ENGLISH:
            text = self._english_normalization(text)

        # 标点符号标准化
        for punct, replacement in self.PUNCTUATION_MAP.items():
            if punct in text:
                text = text.replace(punct, replacement)

        # 最终清理
        text = self.whitespace_pattern.sub(" ", text).strip()
        return text

    def _process_dash(self, text: str) -> str:
        """
        智能处理破折号

        规则：
        1. 如果破折号（– 或 -）前后5个字符内都有数字，转换为"至"（表示范围）
        2. 否则转换为逗号（，）
        3. 如果前后5个字符中有标点符号，则从标点符号后面（或前面）的字符开始到破折号的范围内检查数字
        4. 保持破折号前后的空格数量

        例如：
        - "问题1 – 问题3" → "问题1 至 问题3"
        - "10-15点" → "10至15点"
        - "单元 1  –  预览" → "单元 1  ，  预览"
        - "第1章，导论 – 第2章" → "第1章，导论 至 第2章"
        """
        # 检查是否包含破折号（短破折号或长破折号）
        if "–" not in text and "-" not in text:
            return text

        # 查找所有破折号的位置（优先处理长破折号，然后处理短破折号）
        result = []
        last_pos = 0

        while True:
            # 同时查找长破折号和短破折号，选择最近的一个
            long_dash_pos = text.find("–", last_pos)
            short_dash_pos = text.find("-", last_pos)

            # 确定下一个要处理的破折号位置和类型
            if long_dash_pos == -1 and short_dash_pos == -1:
                # 没有更多破折号，添加剩余文本
                result.append(text[last_pos:])
                break
            elif long_dash_pos == -1:
                dash_pos = short_dash_pos
                dash_char = "-"
            elif short_dash_pos == -1:
                dash_pos = long_dash_pos
                dash_char = "–"
            else:
                # 两种破折号都存在，选择位置更靠前的
                if long_dash_pos < short_dash_pos:
                    dash_pos = long_dash_pos
                    dash_char = "–"
                else:
                    dash_pos = short_dash_pos
                    dash_char = "-"

            # 获取破折号前后的空格
            spaces_before = ""
            spaces_after = ""

            # 计算破折号前的空格
            i = dash_pos - 1
            while i >= 0 and text[i] == " ":
                spaces_before = " " + spaces_before
                i -= 1
            # i+1 现在是非空格字符的结束位置

            # 计算破折号后的空格
            j = dash_pos + 1
            while j < len(text) and text[j] == " ":
                spaces_after += " "
                j += 1
            # j 现在是非空格字符的开始位置

            # 添加破折号之前的文本（不包括前面的空格）
            result.append(text[last_pos : i + 1])

            # 检查破折号前后是否有数字
            has_digit_before = self._check_digit_in_range(
                text, dash_pos, direction="before"
            )
            has_digit_after = self._check_digit_in_range(
                text, dash_pos, direction="after"
            )

            # 根据规则选择替换内容，保持原有空格
            if has_digit_before and has_digit_after:
                result.append(f"{spaces_before}至{spaces_after}")
            else:
                result.append(f"{spaces_before}，{spaces_after}")

            # 更新位置到破折号后空格的结束位置
            last_pos = j

        return "".join(result)

    def _check_digit_in_range(self, text: str, dash_pos: int, direction: str) -> bool:
        """
        检查破折号前后5个字符范围内是否有数字
        如果有标点符号，则从标点符号后面（或前面）的字符开始计算

        Args:
            text: 完整文本
            dash_pos: 破折号位置
            direction: 'before' 或 'after'

        Returns:
            bool: 是否包含数字
        """
        # 定义常见标点符号
        punctuation_chars = set('，。！？；：,.:;!?()（）[]【】{}""' "<>")

        if direction == "before":
            # 检查破折号前面5个字符
            start_pos = max(0, dash_pos - 5)
            check_text = text[start_pos:dash_pos]

            # 从右向左查找最近的标点符号
            last_punct_pos = -1
            for i in range(len(check_text) - 1, -1, -1):
                if check_text[i] in punctuation_chars:
                    last_punct_pos = i
                    break

            # 如果找到标点符号，只检查标点符号之后的部分
            if last_punct_pos != -1:
                check_text = check_text[last_punct_pos + 1 :]

        else:  # direction == 'after'
            # 检查破折号后面5个字符
            end_pos = min(
                len(text), dash_pos + 6
            )  # +1 跳过破折号本身，+5 检查后面5个字符
            check_text = text[dash_pos + 1 : end_pos]

            # 从左向右查找最近的标点符号
            first_punct_pos = -1
            for i in range(len(check_text)):
                if check_text[i] in punctuation_chars:
                    first_punct_pos = i
                    break

            # 如果找到标点符号，只检查标点符号之前的部分
            if first_punct_pos != -1:
                check_text = check_text[:first_punct_pos]

        # 检查处理后的文本中是否包含数字
        return any(char.isdigit() for char in check_text)

    def _process_percentage(self, text: str) -> str:
        """
        智能处理百分比

        规则：
        1. 数字+%符号 → 百分之+中文数字
        2. 例如：20% → 百分之二十，5% → 百分之五，100% → 百分之一百
        3. 支持小数百分比：20.5% → 百分之二十点五
        4. 支持带空格的情况：20 % → 百分之二十

        例如：
        - "准确率达到95%" → "准确率达到百分之九十五"
        - "增长了20.5%" → "增长了百分之二十点五"
        - "只有 5 %" → "只有百分之五"
        """
        if "%" not in text:
            return text

        import re

        # 匹配数字+百分号的模式（支持小数和空格）
        # 匹配整数百分比：20%、20 %
        # 匹配小数百分比：20.5%、20.5 %
        percentage_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*%")

        def replace_percentage(match):
            number_str = match.group(1)

            # 处理小数情况
            if "." in number_str:
                integer_part, decimal_part = number_str.split(".")
                result = "百分之" + self._convert_number_to_chinese(integer_part)
                result += "点" + self._convert_decimal_to_chinese(decimal_part)
            else:
                # 处理整数情况
                result = "百分之" + self._convert_number_to_chinese(number_str)

            return result

        text = percentage_pattern.sub(replace_percentage, text)
        return text

    def _convert_number_to_chinese(self, number_str: str) -> str:
        """
        将阿拉伯数字转换为中文数字

        支持：
        - 个位数：0-9 → 零-九
        - 十位数：10-99 → 十-九十九
        - 百位数：100-999 → 一百-九百九十九
        - 更大数字：逐位转换
        """
        if not number_str.isdigit():
            return number_str

        # 基础数字映射
        digit_map = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        }

        num = int(number_str)

        # 特殊处理0
        if num == 0:
            return "零"

        # 处理1-99
        if num < 100:
            if num < 10:
                return digit_map[str(num)]
            elif num == 10:
                return "十"
            elif num < 20:
                return "十" + digit_map[str(num % 10)]
            else:
                tens = num // 10
                ones = num % 10
                result = digit_map[str(tens)] + "十"
                if ones > 0:
                    result += digit_map[str(ones)]
                return result

        # 处理100-999
        elif num < 1000:
            hundreds = num // 100
            remainder = num % 100
            result = digit_map[str(hundreds)] + "百"

            if remainder == 0:
                return result
            elif remainder < 10:
                result += "零" + digit_map[str(remainder)]
            elif remainder == 10:
                result += "一十"
            elif remainder < 20:
                result += "一十" + digit_map[str(remainder % 10)]
            else:
                tens = remainder // 10
                ones = remainder % 10
                result += digit_map[str(tens)] + "十"
                if ones > 0:
                    result += digit_map[str(ones)]
            return result

        # 对于更大的数字，简化处理：逐位转换
        else:
            return "".join(digit_map.get(digit, digit) for digit in number_str)

    def _convert_decimal_to_chinese(self, decimal_str: str) -> str:
        """
        将小数部分转换为中文

        例如：
        - "5" → "五"
        - "25" → "二五"
        """
        digit_map = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        }

        return "".join(digit_map.get(digit, digit) for digit in decimal_str)

    def _chinese_normalization(self, text: str) -> str:
        """中文文本特殊处理"""
        if not text:
            return ""

        # 数字转中文（简化版）
        number_map = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        }

        def replace_single_digit(match):
            digit = match.group()
            return number_map.get(digit, digit)

        text = self.single_digit_pattern.sub(replace_single_digit, text)
        return text

    def _english_normalization(self, text: str) -> str:
        """英文文本特殊处理"""
        if not text:
            return ""

        def expand_contraction(match):
            contraction = match.group().lower()
            expansions = {
                "don't": "do not",
                "won't": "will not",
                "can't": "cannot",
                "n't": " not",
                "'re": " are",
                "'ve": " have",
                "'ll": " will",
                "'d": " would",
                "'m": " am",
            }
            return expansions.get(contraction, contraction)

        text = self.contractions_pattern.sub(expand_contraction, text)
        return text

    def _intelligent_split(
        self, text: str, strategy: SplitStrategy, language: LanguageType
    ) -> List[str]:
        """智能分割策略"""
        if not text or len(text) <= self.max_text_length:
            return [text] if text else []

        if strategy == SplitStrategy.ADAPTIVE:
            return self._adaptive_split(text, language)
        elif strategy == SplitStrategy.SENTENCE:
            return self._sentence_split(text)
        elif strategy == SplitStrategy.LENGTH:
            return self._length_split(text)
        elif strategy == SplitStrategy.PUNCTUATION:
            return self._punctuation_split(text)
        else:
            return [text]

    def _adaptive_split(self, text: str, language: LanguageType) -> List[str]:
        """自适应分割策略"""
        # 首先尝试句子分割
        sentence_segments = self._sentence_split(text)

        # 检查句子分割效果
        if all(len(seg) <= self.max_text_length for seg in sentence_segments):
            return sentence_segments

        # 对长句子进行二次分割
        final_segments = []
        for segment in sentence_segments:
            if len(segment) <= self.max_text_length:
                final_segments.append(segment)
            else:
                # 使用长度分割
                final_segments.extend(self._length_split(segment))

        return final_segments

    def _sentence_split(self, text: str) -> List[str]:
        """按句子分割"""
        sentences = []
        current_sentence = ""

        for char in text:
            current_sentence += char
            if char in "。！？；.!?;":
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""

        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        return sentences

    def _punctuation_split(self, text: str) -> List[str]:
        """按标点符号分割"""
        segments = []
        current_segment = ""

        for char in text:
            current_segment += char
            if char in "，。！？；：,.:;!?":
                if current_segment.strip():
                    segments.append(current_segment.strip())
                current_segment = ""

        if current_segment.strip():
            segments.append(current_segment.strip())

        return segments

    def _length_split(self, text: str) -> List[str]:
        """按长度智能分割"""
        segments = []
        current_segment = ""
        words = text.split()

        for word in words:
            test_segment = current_segment + " " + word if current_segment else word

            if len(test_segment) <= self.max_text_length:
                current_segment = test_segment
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = word

        if current_segment:
            segments.append(current_segment.strip())

        return segments

    def _validate_result(
        self, text: str, segments: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """验证处理结果"""
        if not text.strip():
            return False, "处理后文本为空"

        if not segments:
            return False, "分割后无有效片段"

        for i, segment in enumerate(segments):
            if len(segment) > self.max_text_length:
                return (
                    False,
                    f"片段 {i+1} 长度超限: {len(segment)} > {self.max_text_length}",
                )

        return True, None

    def batch_process(
        self,
        texts: List[str],
        split_strategy: Optional[SplitStrategy] = None,
        target_language: Optional[LanguageType] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[TextProcessingResult]:
        """批量处理文本列表"""
        results = []
        total = len(texts)

        for i, text in enumerate(texts):
            try:
                result = self.process(text, split_strategy, target_language)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total)

            except Exception as e:
                self.logger.error(f"批量处理第{i+1}项失败: {str(e)}")
                error_result = TextProcessingResult(
                    original_text=text,
                    cleaned_text="",
                    segments=[],
                    language=LanguageType.UNKNOWN,
                    text_hash="",
                    metadata={},
                    is_valid=False,
                    error_message=str(e),
                )
                results.append(error_result)

        return results

    @classmethod
    def create_fast_processor(cls) -> "IntelligentTextProcessor":
        """创建快速处理器"""
        return cls(max_text_length=100, default_split_strategy=SplitStrategy.LENGTH)

    @classmethod
    def create_quality_processor(cls) -> "IntelligentTextProcessor":
        """创建高质量处理器"""
        return cls(max_text_length=300, default_split_strategy=SplitStrategy.ADAPTIVE)

    @classmethod
    def create_batch_processor(cls) -> "IntelligentTextProcessor":
        """创建批量处理器"""
        return cls(max_text_length=200, default_split_strategy=SplitStrategy.ADAPTIVE)

    @classmethod
    def create_minimal_processor(cls) -> "IntelligentTextProcessor":
        """创建最小化处理器"""
        return cls(max_text_length=150, default_split_strategy=SplitStrategy.LENGTH)

    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        total_processed = self._processing_stats["total_processed"]
        total_time = self._processing_stats["total_processing_time"]

        return {
            "total_processed": total_processed,
            "total_errors": self._processing_stats["total_errors"],
            "total_processing_time": total_time,
            "average_processing_time": (
                total_time / total_processed if total_processed > 0 else 0.0
            ),
            "success_rate": (
                (total_processed - self._processing_stats["total_errors"])
                / total_processed
                if total_processed > 0
                else 0.0
            ),
        }


# 全局处理器实例
_global_processor: Optional[IntelligentTextProcessor] = None


def get_global_processor() -> IntelligentTextProcessor:
    """获取全局处理器实例"""
    global _global_processor
    if _global_processor is None:
        _global_processor = IntelligentTextProcessor()
    return _global_processor


def quick_clean_text(text: str) -> str:
    """
    快速文本清理（最常用的基础功能）

    专门用于TTS前的文本清理，会自动删除：
    - 各种括号内容：(注释)、[备注]、{说明}、<标签>、（中文）、【中文】
    - 问题字符：&、®、™、©等特殊符号
    - 多余空白字符

    使用场景：
    - 快速清理用户输入的文本
    - 预处理TTS文本，去除不需要朗读的内容
    - 作为其他处理步骤的前置清理

    Args:
        text: 需要清理的原始文本

    Returns:
        清理后的文本，适合直接用于TTS

    Example:
        >>> quick_clean_text("这是AI技术(人工智能)的演示[备注]。")
        "这是AI技术的演示。"
    """
    processor = get_global_processor()
    return processor._basic_cleaning(text)


def process_text(
    text: str,
    split_strategy: Optional[SplitStrategy] = None,
    target_language: Optional[LanguageType] = None,
) -> TextProcessingResult:
    """
    完整文本处理（核心功能，推荐使用）

    提供完整的TTS文本预处理流程，包括：
    - 基础清理：删除括号内容、问题字符
    - 深度标准化：术语替换（AI→人工智能）、格式规范化
    - 智能分割：根据策略分割为适合TTS的片段
    - 语言检测：自动识别文本语言类型

    使用场景：
    - 单个文本的完整预处理
    - 需要详细处理信息和统计数据
    - 对处理质量要求较高的场景

    Args:
        text: 原始文本
        split_strategy: 分割策略（自适应/句子/长度/标点）
        target_language: 目标语言（自动检测如果为None）

    Returns:
        TextProcessingResult: 包含处理结果、统计信息、元数据的完整对象

    Example:
        >>> result = process_text("这是AI技术(注释)演示。很长的文本会被智能分割。")
        >>> print(result.cleaned_text)  # "这是人工智能演示。很长的文本会被智能分割。"
        >>> print(result.segments)      # ["这是人工智能演示。", "很长的文本会被智能分割。"]
        >>> print(result.language)      # LanguageType.CHINESE
    """
    return get_global_processor().process(text, split_strategy, target_language)


def batch_process_texts(
    texts: List[str],
    split_strategy: Optional[SplitStrategy] = None,
    target_language: Optional[LanguageType] = None,
) -> List[TextProcessingResult]:
    """
    批量文本处理（高效处理大量文本）

    对多个文本进行批量预处理，具有以下优势：
    - 复用全局处理器实例，减少初始化开销
    - 统一的错误处理，单个文本失败不影响其他文本
    - 适合大规模文本处理任务

    使用场景：
    - 批量处理字幕文件
    - 大量文本的预处理任务
    - 需要统一处理参数的多文本场景

    Args:
        texts: 文本列表
        split_strategy: 分割策略（应用于所有文本）
        target_language: 目标语言（应用于所有文本）

    Returns:
        List[TextProcessingResult]: 处理结果列表，与输入文本一一对应

    Example:
        >>> texts = ["文本1(注释)", "文本2[备注]", "文本3{说明}"]
        >>> results = batch_process_texts(texts)
        >>> for result in results:
        ...     print(result.cleaned_text)
        # "文本1"
        # "文本2"
        # "文本3"
    """
    processor = get_global_processor()
    results = []

    for text in texts:
        try:
            result = processor.process(text, split_strategy, target_language)
            results.append(result)
        except Exception as e:
            # 单个文本处理失败时，创建错误结果，不影响其他文本
            error_result = TextProcessingResult(
                original_text=text,
                cleaned_text="",
                segments=[],
                language=LanguageType.UNKNOWN,
                text_hash="",
                metadata={"error_type": type(e).__name__},
                is_valid=False,
                error_message=str(e),
            )
            results.append(error_result)

    return results


# ==================== 管理函数 ====================
