"""
字幕条目数据类

定义字幕条目的数据结构和基本操作
"""

import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SubtitleEntry:
    """字幕条目数据类"""

    start_time: datetime.time
    end_time: datetime.time
    text: str
    style: str = "Default"
    actor: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，支持JSON序列化"""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "text": self.text,
            "style": self.style,
            "actor": self.actor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtitleEntry":
        """从字典创建字幕条目"""
        # 处理时间字符串反序列化
        start_time = data["start_time"]
        end_time = data["end_time"]

        if isinstance(start_time, str):
            start_time = datetime.time.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.time.fromisoformat(end_time)

        return cls(
            start_time=start_time,
            end_time=end_time,
            text=data["text"],
            style=data.get("style", "Default"),
            actor=data.get("actor", ""),
        )

    def duration_seconds(self) -> float:
        """获取字幕持续时间（秒）"""
        return self.end_time_seconds() - self.start_time_seconds()

    def start_time_seconds(self) -> float:
        """获取开始时间（秒）"""
        return self._time_to_seconds(self.start_time)

    def end_time_seconds(self) -> float:
        """获取结束时间（秒）"""
        return self._time_to_seconds(self.end_time)

    @staticmethod
    def _time_to_seconds(time_obj: datetime.time) -> float:
        """时间对象转换为秒数"""
        return (
            time_obj.hour * 3600
            + time_obj.minute * 60
            + time_obj.second
            + time_obj.microsecond / 1000000
        )

    def shift_time(self, offset_seconds: float) -> "SubtitleEntry":
        """时间偏移，返回新的字幕条目"""
        start_seconds = self.start_time_seconds() + offset_seconds
        end_seconds = self.end_time_seconds() + offset_seconds

        return SubtitleEntry(
            start_time=self._seconds_to_time(start_seconds),
            end_time=self._seconds_to_time(end_seconds),
            text=self.text,
            style=self.style,
            actor=self.actor,
        )

    def scale_time(self, scale_factor: float) -> "SubtitleEntry":
        """时间缩放，返回新的字幕条目"""
        start_seconds = self.start_time_seconds() * scale_factor
        end_seconds = self.end_time_seconds() * scale_factor

        return SubtitleEntry(
            start_time=self._seconds_to_time(start_seconds),
            end_time=self._seconds_to_time(end_seconds),
            text=self.text,
            style=self.style,
            actor=self.actor,
        )

    @staticmethod
    def _seconds_to_time(seconds: float) -> datetime.time:
        """秒数转换为时间对象"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        microseconds = int((seconds % 1) * 1000000)

        return datetime.time(hours % 24, minutes, secs, microseconds)

    def is_valid(self) -> bool:
        """检查字幕条目是否有效"""
        return (
            self.start_time < self.end_time
            and self.text.strip() != ""
            and self.duration_seconds() > 0
        )

    def clean_text(self) -> str:
        """清理文本内容"""
        import re

        # 移除HTML标签
        text = re.sub(r"<[^>]+>", "", self.text)
        # 移除多余空格
        text = re.sub(r"\s+", " ", text)
        # 移除首尾空格
        return text.strip()

    def __str__(self) -> str:
        """字符串表示"""
        return f"SubtitleEntry({self.start_time}->{self.end_time}, {self.style}, {self.text[:20]}...)"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()
