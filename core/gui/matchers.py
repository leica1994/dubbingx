"""视频字幕文件匹配工具"""

from pathlib import Path
from typing import List, Optional, Set, Tuple


class VideoSubtitleMatcher:
    """视频字幕匹配器"""

    @staticmethod
    def find_video_subtitle_pairs(folder_path: str) -> List[Tuple[str, str]]:
        """在文件夹中查找视频和对应的字幕文件"""
        folder = Path(folder_path)
        if not folder.exists():
            return []

        video_files = VideoSubtitleMatcher._find_video_files(folder)
        return VideoSubtitleMatcher._match_subtitles(video_files)

    @staticmethod
    def _find_video_files(folder: Path) -> List[Path]:
        """查找所有视频文件"""
        video_extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
            ".m4v",
            ".webm",
            ".ts",
        }

        video_files_set = set()
        for ext in video_extensions:
            video_files_set.update(folder.rglob(f"*{ext}"))
            video_files_set.update(folder.rglob(f"*{ext.upper()}"))

        return sorted(list(video_files_set))

    @staticmethod
    def _match_subtitles(video_files: List[Path]) -> List[Tuple[str, str]]:
        """为视频文件匹配字幕"""
        subtitle_extensions = {".srt", ".ass", ".ssa", ".sub", ".vtt"}
        pairs = []
        matched_subtitles = set()

        for video_file in video_files:
            subtitle_file = VideoSubtitleMatcher._find_matching_subtitle(
                video_file, subtitle_extensions, matched_subtitles
            )
            if subtitle_file:
                pairs.append((str(video_file), str(subtitle_file)))
                matched_subtitles.add(str(subtitle_file))

        return pairs

    @staticmethod
    def _find_matching_subtitle(
        video_file: Path, subtitle_extensions: set, matched_subtitles: set
    ) -> Optional[Path]:
        """为单个视频文件查找匹配的字幕"""
        video_name = video_file.stem
        video_folder = video_file.parent

        for ext in subtitle_extensions:
            for case_ext in [ext, ext.upper()]:
                potential_subtitle = video_folder / f"{video_name}{case_ext}"
                if (
                    potential_subtitle.exists()
                    and str(potential_subtitle) not in matched_subtitles
                ):
                    return potential_subtitle
        return None