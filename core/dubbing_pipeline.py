import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.audio_align_processor import (
    align_audio_with_subtitles,
    generate_aligned_srt,
    process_video_speed_adjustment,
)
from core.media_processor import (
    generate_reference_audio,
    merge_audio_video,
    separate_media,
)
from core.subtitle.subtitle_processor import (
    convert_subtitle,
    sync_srt_timestamps_to_ass,
)
from core.subtitle_preprocessor import preprocess_subtitle
from core.tts_processor import generate_tts_from_reference


class DubbingPaths:
    """配音文件路径管理类"""

    def __init__(
            self,
            video_path: str,
            subtitle_path: Optional[str] = None,
            output_dir: Optional[Path] = None,
    ):
        """
        初始化配音路径管理器

        Args:
            video_path: 视频文件路径
            subtitle_path: 字幕文件路径，如果为None则自动匹配同名字幕文件
            output_dir: 输出目录路径，如果为None则使用视频父目录下的outputs
        """
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.video_ext = self.video_path.suffix  # 保存原始视频格式

        # 处理字幕文件路径
        if subtitle_path:
            self.subtitle_path = Path(subtitle_path)
        else:
            # 自动匹配同名字幕文件
            self.subtitle_path = self._find_matching_subtitle()

        if output_dir:
            self.output_dir = output_dir
        else:
            # 使用视频父目录下的outputs目录，并在其下创建文件名目录
            video_parent = self.video_path.parent
            outputs_dir = video_parent / "outputs"
            outputs_dir.mkdir(exist_ok=True)

            # 清理文件名并创建子目录
            clean_video_name = self._sanitize_filename(self.video_name)
            self.output_dir = outputs_dir / clean_video_name
            self.output_dir.mkdir(exist_ok=True)

        # 初始化所有路径
        self._initialize_paths()

    def _find_matching_subtitle(self) -> Path:
        """自动匹配同名字幕文件"""
        video_dir = self.video_path.parent
        video_name = self.video_name

        # 常见的字幕文件扩展名
        subtitle_extensions = [".srt", ".ass", ".ssa", ".sub", ".vtt"]

        for ext in subtitle_extensions:
            subtitle_file = video_dir / f"{video_name}{ext}"
            if subtitle_file.exists():
                return subtitle_file

        # 如果没有找到，尝试模糊匹配
        for file in video_dir.glob(f"{video_name}*"):
            if file.suffix.lower() in subtitle_extensions:
                return file

        # 如果仍然没有找到，抛出异常
        raise FileNotFoundError(f"无法找到与视频 {self.video_name} 匹配的字幕文件")

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，将不支持的字符转为下划线"""
        # 替换Windows和Linux不支持的特殊字符
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
        # 替换其他可能引起问题的字符
        sanitized = re.sub(r"[@#&%=+]", "_", sanitized)
        # 替换空格为下划线
        sanitized = sanitized.replace(" ", "_")
        # 替换连续的下划线为单个下划线
        sanitized = re.sub(r"_+", "_", sanitized)
        # 去除开头和结尾的下划线
        sanitized = sanitized.strip("_")
        # 确保文件名不为空
        if not sanitized:
            sanitized = "unnamed"
        return sanitized

    def _initialize_paths(self):
        """初始化所有文件路径"""
        # 为分离后的媒体文件创建子目录
        self._media_separation_dir = self.output_dir / "media_separation"

        # 初始化所有文件路径
        self._processed_subtitle = self.output_dir / f"{self.video_name}_processed.srt"
        self._vocal_audio = self._media_separation_dir / f"{self.video_name}_vocal.wav"
        self._background_audio = (
                self._media_separation_dir / f"{self.video_name}_background.wav"
        )
        self._silent_video = (
                self._media_separation_dir / f"{self.video_name}_silent{self.video_ext}"
        )
        self._reference_audio_dir = self.output_dir / "reference_audio"
        self._tts_output_dir = self.output_dir / "tts_output"
        self._aligned_audio_dir = self.output_dir / "aligned_audio"
        self._adjusted_video_dir = self.output_dir / "adjusted_video"
        self._reference_results = (
                self._reference_audio_dir
                / f"{self.video_name}_vocal_reference_audio_results.json"
        )
        self._tts_results = self._tts_output_dir / "tts_generation_results.json"
        self._aligned_results = (
                self._aligned_audio_dir / "aligned_tts_generation_results.json"
        )
        self._aligned_audio = (
                self._aligned_audio_dir / "aligned_tts_generation_results.wav"
        )
        self._aligned_srt = (
                self._aligned_audio_dir / "aligned_tts_generation_aligned.srt"
        )
        self._final_video = (
                self.output_dir / f"{self.video_name}_dubbed{self.video_ext}"
        )
        self._speed_adjusted_video = (
                self._adjusted_video_dir
                / f"final_speed_adjusted_{self.video_name}_silent{self.video_ext}"
        )
        self._pipeline_cache = (
                self.output_dir / f"{self.video_name}_pipeline_cache.json"
        )

        # 创建必要的子目录
        for dir_path in [
            self._media_separation_dir,
            self._reference_audio_dir,
            self._tts_output_dir,
            self._aligned_audio_dir,
            self._adjusted_video_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

    # 属性访问器
    @property
    def processed_subtitle(self) -> Path:
        return self._processed_subtitle

    @property
    def vocal_audio(self) -> Path:
        return self._vocal_audio

    @property
    def background_audio(self) -> Path:
        return self._background_audio

    @property
    def silent_video(self) -> Path:
        return self._silent_video

    @property
    def media_separation_dir(self) -> Path:
        return self._media_separation_dir

    @property
    def reference_audio_dir(self) -> Path:
        return self._reference_audio_dir

    @property
    def tts_output_dir(self) -> Path:
        return self._tts_output_dir

    @property
    def aligned_audio_dir(self) -> Path:
        return self._aligned_audio_dir

    @property
    def adjusted_video_dir(self) -> Path:
        return self._adjusted_video_dir

    @property
    def reference_results(self) -> Path:
        return self._reference_results

    @property
    def tts_results(self) -> Path:
        return self._tts_results

    @property
    def aligned_results(self) -> Path:
        return self._aligned_results

    @property
    def aligned_audio(self) -> Path:
        return self._aligned_audio

    @property
    def aligned_srt(self) -> Path:
        return self._aligned_srt

    @property
    def final_video(self) -> Path:
        return self._final_video

    @property
    def speed_adjusted_video(self) -> Path:
        return self._speed_adjusted_video

    @property
    def pipeline_cache(self) -> Path:
        return self._pipeline_cache

    # Get方法
    def get_processed_subtitle(self) -> Path:
        return self._processed_subtitle

    def get_vocal_audio(self) -> Path:
        return self._vocal_audio

    def get_background_audio(self) -> Path:
        return self._background_audio

    def get_silent_video(self) -> Path:
        return self._silent_video

    def get_media_separation_dir(self) -> Path:
        return self._media_separation_dir

    def get_reference_audio_dir(self) -> Path:
        return self._reference_audio_dir

    def get_tts_output_dir(self) -> Path:
        return self._tts_output_dir

    def get_aligned_audio_dir(self) -> Path:
        return self._aligned_audio_dir

    def get_adjusted_video_dir(self) -> Path:
        return self._adjusted_video_dir

    def get_reference_results(self) -> Path:
        return self._reference_results

    def get_tts_results(self) -> Path:
        return self._tts_results

    def get_aligned_results(self) -> Path:
        return self._aligned_results

    def get_aligned_audio(self) -> Path:
        return self._aligned_audio

    def get_aligned_srt(self) -> Path:
        return self._aligned_srt

    def get_final_video(self) -> Path:
        return self._final_video

    def get_speed_adjusted_video(self) -> Path:
        return self._speed_adjusted_video

    def get_pipeline_cache(self) -> Path:
        return self._pipeline_cache


class DubbingPipeline:
    """完整的视频配音处理流水线"""

    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化配音流水线

        Args:
            output_dir: 输出目录路径，如果为None则使用视频父目录下的outputs
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir) if output_dir else None

    def _get_file_paths(
            self, video_path: str, subtitle_path: Optional[str] = None
    ) -> DubbingPaths:
        """生成所有需要的文件路径"""
        return DubbingPaths(video_path, subtitle_path, self.output_dir)

    def cleanup_large_cache_file(self, video_path: str) -> Dict[str, Any]:
        """
        清理过大的缓存文件并重新创建优化后的缓存

        Args:
            video_path: 视频文件路径

        Returns:
            清理结果
        """
        try:
            paths = self._get_file_paths(video_path)
            cache_file = paths.pipeline_cache

            if not cache_file.exists():
                return {
                    "success": True,
                    "message": "缓存文件不存在",
                    "cache_file": str(cache_file),
                }

            # 检查文件大小
            file_size = cache_file.stat().st_size
            if file_size < 1024 * 1024:  # 小于1MB，不需要清理
                return {
                    "success": True,
                    "message": f"缓存文件大小正常 ({file_size} bytes)",
                    "cache_file": str(cache_file),
                    "file_size": file_size,
                }

            self.logger.info(f"检测到过大的缓存文件: {file_size} bytes，开始优化...")

            # 加载现有缓存
            old_cache = self._load_pipeline_cache(cache_file)
            if not old_cache:
                return {
                    "success": False,
                    "message": "无法加载现有缓存",
                    "cache_file": str(cache_file),
                }

            # 备份旧缓存
            backup_file = cache_file.with_suffix(".json.backup")
            cache_file.rename(backup_file)

            # 重新创建优化后的缓存
            new_cache = {
                "video_path": old_cache.get("video_path"),
                "subtitle_path": old_cache.get("subtitle_path"),
                "output_dir": old_cache.get("output_dir"),
                "created_at": old_cache.get("created_at"),
                "updated_at": datetime.now().isoformat(),
                "completed_steps": {},
                "file_paths": old_cache.get("file_paths", {}),
                "cache_optimized": True,
                "original_size": file_size,
            }

            # 重新处理已完成步骤的结果
            completed_steps = old_cache.get("completed_steps", {})
            for step_name, step_data in completed_steps.items():
                if step_data.get("completed", False):
                    # 优化步骤结果
                    optimized_result = self._optimize_result_for_cache(
                        step_name, step_data.get("result")
                    )

                    new_cache["completed_steps"][step_name] = {
                        "completed": True,
                        "completed_at": step_data.get("completed_at"),
                        "result": optimized_result,
                    }

            # 保存优化后的缓存
            self._save_pipeline_cache(cache_file, new_cache)

            # 检查新文件大小
            new_size = cache_file.stat().st_size
            saved_space = file_size - new_size

            self.logger.info(
                f"缓存优化完成: {file_size} -> {new_size} bytes (节省 {saved_space} bytes)"
            )

            return {
                "success": True,
                "message": f"缓存优化完成，节省 {saved_space} bytes",
                "cache_file": str(cache_file),
                "original_size": file_size,
                "new_size": new_size,
                "saved_space": saved_space,
                "backup_file": str(backup_file),
            }

        except Exception as e:
            self.logger.error(f"清理缓存文件失败: {str(e)}")
            return {
                "success": False,
                "message": f"清理缓存文件失败: {str(e)}",
                "error": str(e),
            }

    def _load_pipeline_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """加载流水线缓存"""
        try:
            if not cache_path.exists():
                self.logger.debug(f"缓存文件不存在: {cache_path}")
                return None

            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            self.logger.info(f"找到缓存文件: {cache_path}")

            # 验证缓存数据结构
            if not self._validate_cache_structure(cache_data):
                self.logger.warning("缓存数据结构无效，将删除损坏的缓存文件")
                cache_path.unlink()
                return None

            return cache_data
        except json.JSONDecodeError as e:
            self.logger.warning(f"缓存文件JSON格式错误: {str(e)}")
            # 删除损坏的缓存文件
            try:
                cache_path.unlink()
                self.logger.info("已删除损坏的缓存文件")
            except:
                pass
            return None
        except Exception as e:
            self.logger.warning(f"加载缓存失败: {str(e)}")
            return None

    def _validate_cache_structure(self, cache_data: Dict[str, Any]) -> bool:
        """验证缓存数据结构是否有效"""
        try:
            # 检查必需字段
            required_fields = [
                "video_path",
                "subtitle_path",
                "output_dir",
                "created_at",
            ]
            for field in required_fields:
                if field not in cache_data:
                    self.logger.warning(f"缓存缺少必需字段: {field}")
                    return False

            # 检查completed_steps结构
            completed_steps = cache_data.get("completed_steps", {})
            if not isinstance(completed_steps, dict):
                self.logger.warning("缓存中的completed_steps不是字典类型")
                return False

            # 检查每个步骤的数据结构
            for step_name, step_data in completed_steps.items():
                if not isinstance(step_data, dict):
                    self.logger.warning(f"步骤 {step_name} 的数据不是字典类型")
                    return False

                if not step_data.get("completed", False):
                    continue

                # 检查必需的时间字段
                if "completed_at" not in step_data:
                    self.logger.warning(f"步骤 {step_name} 缺少completed_at字段")
                    return False

            return True

        except Exception as e:
            self.logger.warning(f"验证缓存结构时发生错误: {str(e)}")
            return False

    def _save_pipeline_cache(
            self, cache_path: Path, cache_data: Dict[str, Any]
    ) -> bool:
        """保存流水线缓存"""
        try:
            cache_data["updated_at"] = datetime.now().isoformat()

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"缓存已保存: {cache_path}")
            return True
        except Exception as e:
            self.logger.warning(f"保存缓存失败: {str(e)}")
            return False

    def _check_step_completed(self, cache_data: Dict[str, Any], step_name: str) -> bool:
        """检查指定步骤是否已完成"""
        return (
            cache_data.get("completed_steps", {})
            .get(step_name, {})
            .get("completed", False)
        )

    def _mark_step_completed(
            self, cache_data: Dict[str, Any], step_name: str, result: Dict[str, Any] = None
    ) -> None:
        """标记指定步骤为已完成"""
        if "completed_steps" not in cache_data:
            cache_data["completed_steps"] = {}

        # 优化缓存数据，只存储必要信息
        optimized_result = self._optimize_result_for_cache(step_name, result)

        cache_data["completed_steps"][step_name] = {
            "completed": True,
            "completed_at": datetime.now().isoformat(),
            "result": optimized_result,
        }

    def _optimize_result_for_cache(
            self, step_name: str, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """优化结果数据以减少缓存大小"""
        if not result:
            return {}

        optimized = {"success": result.get("success", False)}

        if step_name == "preprocess_subtitle":
            # 只存储处理后的字幕文件路径，不存储完整的字幕条目
            if result.get("result") and result["result"].get("processed_subtitle_path"):
                optimized["processed_subtitle_path"] = result["result"][
                    "processed_subtitle_path"
                ]
                optimized["entries_count"] = len(
                    result["result"].get("subtitle_entries", [])
                )

        elif step_name == "separate_media":
            # 只存储分离后的文件路径
            if result.get("result"):
                optimized.update(
                    {
                        "silent_video_path": result["result"].get("silent_video_path"),
                        "vocal_audio_path": result["result"].get("vocal_audio_path"),
                        "background_audio_path": result["result"].get(
                            "background_audio_path"
                        ),
                    }
                )

        elif step_name == "generate_reference_audio":
            # 只存储关键信息，不存储完整的音频片段信息
            if result.get("result"):
                optimized.update(
                    {
                        "output_dir": result["result"].get("output_dir"),
                        "total_segments": result["result"].get("total_segments", 0),
                        "success": result["result"].get("success", False),
                    }
                )

        elif step_name == "generate_tts":
            # 只存储TTS生成结果的关键信息
            if result.get("result"):
                optimized.update(
                    {
                        "success": result["result"].get("success", False),
                        "total_segments": result["result"].get("total_segments", 0),
                        "failed_segments": result["result"].get("failed_segments", 0),
                    }
                )

        elif step_name == "align_audio":
            # 只存储对齐结果的关键信息
            if result.get("result"):
                optimized.update(
                    {
                        "success": result["result"].get("success", False),
                        "total_duration": result["result"].get("total_duration", 0),
                        "segments_count": result["result"].get("segments_count", 0),
                    }
                )

        else:
            # 对于其他步骤，只存储成功状态
            optimized["success"] = result.get("success", False)

        return optimized

    def _validate_step_dependencies(
            self, cache_data: Dict[str, Any], step_name: str
    ) -> bool:
        """验证步骤依赖是否满足"""
        step_dependencies = {
            "preprocess_subtitle": [],
            "separate_media": ["preprocess_subtitle"],
            "generate_reference_audio": ["preprocess_subtitle", "separate_media"],
            "generate_tts": [
                "preprocess_subtitle",
                "separate_media",
                "generate_reference_audio",
            ],
            "align_audio": [
                "preprocess_subtitle",
                "separate_media",
                "generate_reference_audio",
                "generate_tts",
            ],
            "generate_aligned_srt": [
                "preprocess_subtitle",
                "separate_media",
                "generate_reference_audio",
                "generate_tts",
                "align_audio",
            ],
            "process_video_speed": [
                "preprocess_subtitle",
                "separate_media",
                "generate_reference_audio",
                "generate_tts",
                "align_audio",
                "generate_aligned_srt",
            ],
            "merge_audio_video": [
                "preprocess_subtitle",
                "separate_media",
                "generate_reference_audio",
                "generate_tts",
                "align_audio",
                "generate_aligned_srt",
                "process_video_speed",
            ],
        }

        dependencies = step_dependencies.get(step_name, [])
        for dep in dependencies:
            if not self._check_step_completed(cache_data, dep):
                self.logger.warning(f"步骤 {step_name} 的依赖 {dep} 未完成")
                return False

        return True

    
    def check_and_repair_cache(self, video_path: str) -> Dict[str, Any]:
        """
        检查并修复缓存文件

        Args:
            video_path: 视频文件路径

        Returns:
            检查结果
        """
        try:
            paths = self._get_file_paths(video_path)
            cache_file = paths.pipeline_cache

            if not cache_file.exists():
                return {
                    "success": True,
                    "message": "缓存文件不存在",
                    "cache_file": str(cache_file),
                    "cache_exists": False,
                    "repaired": False,
                }

            # 尝试加载缓存
            cache_data = self._load_pipeline_cache(cache_file)

            if cache_data is None:
                # 缓存文件已损坏，已被删除
                return {
                    "success": True,
                    "message": "检测到损坏的缓存文件，已删除",
                    "cache_file": str(cache_file),
                    "cache_exists": False,
                    "repaired": True,
                    "action": "deleted_corrupted_cache",
                }

            # 检查缓存文件大小
            file_size = cache_file.stat().st_size
            if file_size > 1024 * 1024:  # 大于1MB
                self.logger.info(f"检测到过大的缓存文件: {file_size} bytes")

                # 优化缓存文件
                cleanup_result = self.cleanup_large_cache_file(video_path)
                return {
                    "success": True,
                    "message": "缓存文件已优化",
                    "cache_file": str(cache_file),
                    "cache_exists": True,
                    "repaired": True,
                    "action": "optimized_cache",
                    "cleanup_result": cleanup_result,
                }

            return {
                "success": True,
                "message": "缓存文件正常",
                "cache_file": str(cache_file),
                "cache_exists": True,
                "repaired": False,
                "file_size": file_size,
            }

        except Exception as e:
            self.logger.error(f"检查缓存文件失败: {str(e)}")
            return {
                "success": False,
                "message": f"检查缓存文件失败: {str(e)}",
                "error": str(e),
            }

    def clear_pipeline_cache(self, video_path: str) -> Dict[str, Any]:
        """
        清理流水线缓存

        Args:
            video_path: 视频文件路径

        Returns:
            清理结果
        """
        try:
            paths = self._get_file_paths(video_path)
            cache_file = paths.pipeline_cache

            if cache_file.exists():
                cache_file.unlink()
                self.logger.info(f"缓存文件已删除: {cache_file}")
                return {
                    "success": True,
                    "message": "缓存文件已删除",
                    "cache_file": str(cache_file),
                }
            else:
                self.logger.info(f"缓存文件不存在: {cache_file}")
                return {
                    "success": True,
                    "message": "缓存文件不存在",
                    "cache_file": str(cache_file),
                }

        except Exception as e:
            self.logger.error(f"清理缓存失败: {str(e)}")
            return {
                "success": False,
                "message": f"清理缓存失败: {str(e)}",
                "error": str(e),
            }

    def get_pipeline_cache_info(self, video_path: str) -> Dict[str, Any]:
        """
        获取流水线缓存信息

        Args:
            video_path: 视频文件路径

        Returns:
            缓存信息
        """
        try:
            paths = self._get_file_paths(video_path)
            cache_file = paths.pipeline_cache

            if not cache_file.exists():
                return {
                    "success": True,
                    "message": "缓存文件不存在",
                    "cache_file": str(cache_file),
                    "cache_exists": False,
                }

            cache_data = self._load_pipeline_cache(cache_file)
            if not cache_data:
                return {
                    "success": False,
                    "message": "无法读取缓存文件",
                    "cache_file": str(cache_file),
                    "cache_exists": True,
                }

            completed_steps = cache_data.get("completed_steps", {})
            completed_count = sum(
                1 for step in completed_steps.values() if step.get("completed", False)
            )

            return {
                "success": True,
                "message": "缓存信息获取成功",
                "cache_file": str(cache_file),
                "cache_exists": True,
                "created_at": cache_data.get("created_at"),
                "updated_at": cache_data.get("updated_at"),
                "total_steps": 8,
                "completed_steps": completed_count,
                "completed_step_names": [
                    name
                    for name, step in completed_steps.items()
                    if step.get("completed", False)
                ],
                "remaining_steps": 8 - completed_count,
                "remaining_step_names": [
                    name
                    for name, step in completed_steps.items()
                    if not step.get("completed", False)
                ],
            }

        except Exception as e:
            self.logger.error(f"获取缓存信息失败: {str(e)}")
            return {
                "success": False,
                "message": f"获取缓存信息失败: {str(e)}",
                "error": str(e),
            }

    def clean_temp_files(self, video_path: str) -> Dict[str, Any]:
        """
        清理临时文件

        Args:
            video_path: 视频文件路径

        Returns:
            清理结果
        """
        try:
            output_dir = self._get_output_dir(video_path)
            temp_dirs = [
                output_dir / "reference_audio",
                output_dir / "tts_output",
                output_dir / "aligned_audio",
                output_dir / "adjusted_video",
            ]

            cleaned_files = []
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    for file in temp_dir.rglob("*"):
                        if file.is_file():
                            file.unlink()
                            cleaned_files.append(str(file))
                    temp_dir.rmdir()

            return {
                "success": True,
                "message": f"清理完成，共清理 {len(cleaned_files)} 个文件",
                "cleaned_files": cleaned_files,
            }

        except Exception as e:
            return {"success": False, "message": f"清理失败: {str(e)}", "error": str(e)}


class StepType(Enum):
    """处理步骤类型，用于资源管理"""

    GPU_INTENSIVE = "gpu_intensive"  # GPU密集型：媒体分离、TTS生成
    CPU_INTENSIVE = "cpu_intensive"  # CPU密集型：字幕处理、音频对齐
    IO_INTENSIVE = "io_intensive"  # I/O密集型：文件操作、FFmpeg处理


@dataclass
class VideoTask:
    """视频处理任务"""

    video_path: str
    subtitle_path: Optional[str]
    task_id: str
    paths: DubbingPaths = None
    current_step: int = 0
    completed_steps: Dict[str, bool] = field(default_factory=dict)
    step_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cache_data: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None


@dataclass
class ResourcePool:
    """资源池管理"""

    gpu_semaphore: threading.Semaphore = field(
        default_factory=lambda: threading.Semaphore(2)
    )
    cpu_semaphore: threading.Semaphore = field(
        default_factory=lambda: threading.Semaphore(4)
    )
    io_semaphore: threading.Semaphore = field(
        default_factory=lambda: threading.Semaphore(8)
    )

    def get_config_summary(self) -> Dict[StepType, Dict[str, int]]:
        """获取资源池配置摘要"""
        return {
            StepType.GPU_INTENSIVE: {
                "max_workers": self.gpu_semaphore._value,
                "available": self.gpu_semaphore._value,
            },
            StepType.CPU_INTENSIVE: {
                "max_workers": self.cpu_semaphore._value,
                "available": self.cpu_semaphore._value,
            },
            StepType.IO_INTENSIVE: {
                "max_workers": self.io_semaphore._value,
                "available": self.io_semaphore._value,
            },
        }


class ParallelDubbingPipeline(DubbingPipeline):
    """支持并行处理的配音流水线"""

    PROCESSING_STEPS = [
        ("preprocess_subtitle", StepType.CPU_INTENSIVE),
        ("separate_media", StepType.GPU_INTENSIVE),
        ("generate_reference_audio", StepType.CPU_INTENSIVE),
        ("generate_tts", StepType.GPU_INTENSIVE),
        ("align_audio", StepType.CPU_INTENSIVE),
        ("generate_aligned_srt", StepType.IO_INTENSIVE),
        ("process_video_speed", StepType.IO_INTENSIVE),
        ("merge_audio_video", StepType.IO_INTENSIVE),
    ]

    def __init__(self, output_dir: Optional[str] = None, max_workers: int = None):
        """
        初始化并行配音流水线

        Args:
            output_dir: 输出目录路径
            max_workers: 最大工作线程数，默认为CPU核心数
        """
        super().__init__(output_dir)
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.resource_pool = ResourcePool()
        self.active_tasks: Dict[str, VideoTask] = {}
        self.task_lock = threading.Lock()
        self.progress_callback = None

    def process_batch_parallel(
            self,
            video_subtitle_pairs: List[Tuple[str, Optional[str]]],
            resume_from_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        并行批量处理多个视频

        Args:
            video_subtitle_pairs: 包含(video_path, subtitle_path)元组的列表
            resume_from_cache: 是否从缓存恢复

        Returns:
            批量处理结果
        """
        start_time = time.time()
        self.logger.info(f"开始并行批量处理 {len(video_subtitle_pairs)} 个视频")

        # 创建视频任务
        tasks = []
        for i, (video_path, subtitle_path) in enumerate(video_subtitle_pairs):
            task_id = f"task_{i:03d}_{Path(video_path).stem}"
            try:
                paths = self._get_file_paths(video_path, subtitle_path)
                task = VideoTask(
                    video_path=video_path,
                    subtitle_path=subtitle_path,
                    task_id=task_id,
                    paths=paths,
                )

                # 加载缓存数据
                if resume_from_cache:
                    cache_data = self._load_pipeline_cache(paths.pipeline_cache)
                    if cache_data:
                        task.cache_data = cache_data
                        task.completed_steps = {
                            step_name: step_data.get("completed", False)
                            for step_name, step_data in cache_data.get(
                                "completed_steps", {}
                            ).items()
                        }

                tasks.append(task)

            except Exception as e:
                self.logger.error(f"创建任务失败 {task_id}: {e}")
                tasks.append(
                    VideoTask(
                        video_path=video_path,
                        subtitle_path=subtitle_path,
                        task_id=task_id,
                        status="failed",
                        error_message=str(e),
                    )
                )

        # 执行并行处理
        results = self._execute_parallel_processing(tasks)

        # 统计结果
        total_time = time.time() - start_time
        success_count = sum(1 for result in results if result["success"])
        failed_count = len(results) - success_count

        self.logger.info(f"并行批量处理完成，耗时 {total_time:.2f}s")
        self.logger.info(f"成功: {success_count}, 失败: {failed_count}")

        return {
            "success": failed_count == 0,
            "message": f"并行批量处理完成: {success_count} 成功, {failed_count} 失败",
            "total_count": len(video_subtitle_pairs),
            "success_count": success_count,
            "failed_count": failed_count,
            "total_time": total_time,
            "results": results,
        }

    def _execute_parallel_processing(
            self, tasks: List[VideoTask]
    ) -> List[Dict[str, Any]]:
        """
        执行并行处理

        Args:
            tasks: 视频任务列表

        Returns:
            处理结果列表
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 将任务添加到活动任务字典
            with self.task_lock:
                for task in tasks:
                    self.active_tasks[task.task_id] = task

            try:
                # 提交所有任务
                future_to_task = {
                    executor.submit(self._process_single_video_parallel, task): task
                    for task in tasks
                }

                # 等待所有任务完成
                results = []
                completed_count = 0
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed_count += 1

                        # 调用进度回调
                        if self.progress_callback:
                            self.progress_callback(
                                completed_count, len(tasks), task.video_path
                            )

                    except Exception as e:
                        self.logger.error(f"任务 {task.task_id} 执行失败: {e}")
                        results.append(
                            {
                                "success": False,
                                "task_id": task.task_id,
                                "video_path": task.video_path,
                                "message": f"任务执行失败: {str(e)}",
                                "error": str(e),
                            }
                        )
                        completed_count += 1

                        # 调用进度回调
                        if self.progress_callback:
                            self.progress_callback(
                                completed_count, len(tasks), task.video_path
                            )

                return results

            finally:
                # 清理活动任务
                with self.task_lock:
                    self.active_tasks.clear()

    def _process_single_video_parallel(self, task: VideoTask) -> Dict[str, Any]:
        """
        并行处理单个视频任务

        Args:
            task: 视频任务

        Returns:
            处理结果
        """
        try:
            task.status = "processing"
            self.logger.info(f"开始处理任务 {task.task_id}: {task.video_path}")

            if task.error_message:
                return {
                    "success": False,
                    "task_id": task.task_id,
                    "video_path": task.video_path,
                    "message": task.error_message,
                    "error": task.error_message,
                }

            # 初始化缓存数据
            if not task.cache_data:
                task.cache_data = {
                    "video_path": task.video_path,
                    "subtitle_path": task.subtitle_path,
                    "output_dir": str(task.paths.output_dir),
                    "created_at": datetime.now().isoformat(),
                    "completed_steps": {},
                    "file_paths": self._generate_file_paths_dict(task.paths),
                }

            # 按步骤执行处理
            for step_index, (step_name, step_type) in enumerate(self.PROCESSING_STEPS):
                if task.completed_steps.get(step_name, False):
                    self.logger.info(
                        f"任务 {task.task_id} - 步骤 {step_name} (已完成，跳过)"
                    )
                    continue

                # 检查依赖
                if not self._check_step_dependencies_for_task(task, step_name):
                    self.logger.error(
                        f"任务 {task.task_id} - 步骤 {step_name} 依赖未满足"
                    )
                    raise Exception(f"步骤 {step_name} 依赖未满足")

                # 获取对应的信号量
                semaphore = self._get_semaphore_for_step_type(step_type)

                # 执行步骤
                with semaphore:
                    self.logger.info(f"任务 {task.task_id} - 开始步骤 {step_name}")
                    step_result = self._execute_single_step(task, step_name)

                    if not step_result.get("success", False):
                        raise Exception(
                            f"步骤 {step_name} 执行失败: {step_result.get('error', '未知错误')}"
                        )

                    # 更新任务状态
                    task.completed_steps[step_name] = True
                    task.step_results[step_name] = step_result
                    task.current_step = step_index + 1

                    # 标记步骤完成并保存缓存
                    self._mark_step_completed(task.cache_data, step_name, step_result)
                    self._save_pipeline_cache(
                        task.paths.pipeline_cache, task.cache_data
                    )

                    self.logger.info(f"任务 {task.task_id} - 完成步骤 {step_name}")

            # 任务完成
            task.status = "completed"
            total_time = time.time() - task.start_time

            self.logger.info(f"任务 {task.task_id} 处理完成，耗时 {total_time:.2f}s")

            return {
                "success": True,
                "task_id": task.task_id,
                "video_path": task.video_path,
                "message": "视频配音处理完成",
                "output_file": str(task.paths.final_video),
                "output_dir": str(task.paths.output_dir),
                "steps_completed": len(self.PROCESSING_STEPS),
                "processing_time": total_time,
            }

        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            self.logger.error(f"任务 {task.task_id} 处理失败: {e}")

            return {
                "success": False,
                "task_id": task.task_id,
                "video_path": task.video_path,
                "message": f"处理失败: {str(e)}",
                "error": str(e),
            }

    def _get_semaphore_for_step_type(self, step_type: StepType) -> threading.Semaphore:
        """
        根据步骤类型获取对应的信号量

        Args:
            step_type: 步骤类型

        Returns:
            对应的信号量
        """
        if step_type == StepType.GPU_INTENSIVE:
            return self.resource_pool.gpu_semaphore
        elif step_type == StepType.CPU_INTENSIVE:
            return self.resource_pool.cpu_semaphore
        else:  # IO_INTENSIVE
            return self.resource_pool.io_semaphore

    def _generate_file_paths_dict(self, paths: DubbingPaths) -> Dict[str, str]:
        """
        生成文件路径字典

        Args:
            paths: 路径管理对象

        Returns:
            文件路径字典
        """
        return {
            "video_path": str(paths.video_path),
            "subtitle_path": str(paths.subtitle_path),
            "processed_subtitle": str(paths.processed_subtitle),
            "vocal_audio": str(paths.vocal_audio),
            "background_audio": str(paths.background_audio),
            "silent_video": str(paths.silent_video),
            "media_separation_dir": str(paths.media_separation_dir),
            "reference_audio_dir": str(paths.reference_audio_dir),
            "tts_output_dir": str(paths.tts_output_dir),
            "aligned_audio_dir": str(paths.aligned_audio_dir),
            "adjusted_video_dir": str(paths.adjusted_video_dir),
            "reference_results": str(paths.reference_results),
            "tts_results": str(paths.tts_results),
            "aligned_results": str(paths.aligned_results),
            "aligned_audio": str(paths.aligned_audio),
            "aligned_srt": str(paths.aligned_srt),
            "final_video": str(paths.final_video),
            "speed_adjusted_video": str(paths.speed_adjusted_video),
            "pipeline_cache": str(paths.pipeline_cache),
        }

    def _check_step_dependencies_for_task(
            self, task: VideoTask, step_name: str
    ) -> bool:
        """
        检查任务的步骤依赖是否满足

        Args:
            task: 视频任务
            step_name: 步骤名称

        Returns:
            依赖是否满足
        """
        step_dependencies = {
            "preprocess_subtitle": [],
            "separate_media": ["preprocess_subtitle"],
            "generate_reference_audio": ["preprocess_subtitle", "separate_media"],
            "generate_tts": [
                "preprocess_subtitle",
                "separate_media",
                "generate_reference_audio",
            ],
            "align_audio": [
                "preprocess_subtitle",
                "separate_media",
                "generate_reference_audio",
                "generate_tts",
            ],
            "generate_aligned_srt": [
                "preprocess_subtitle",
                "separate_media",
                "generate_reference_audio",
                "generate_tts",
                "align_audio",
            ],
            "process_video_speed": [
                "preprocess_subtitle",
                "separate_media",
                "generate_reference_audio",
                "generate_tts",
                "align_audio",
                "generate_aligned_srt",
            ],
            "merge_audio_video": [
                "preprocess_subtitle",
                "separate_media",
                "generate_reference_audio",
                "generate_tts",
                "align_audio",
                "generate_aligned_srt",
                "process_video_speed",
            ],
        }

        dependencies = step_dependencies.get(step_name, [])
        for dep in dependencies:
            if not task.completed_steps.get(dep, False):
                self.logger.warning(
                    f"任务 {task.task_id} - 步骤 {step_name} 的依赖 {dep} 未完成"
                )
                return False

        return True

    def _execute_single_step(self, task: VideoTask, step_name: str) -> Dict[str, Any]:
        """
        执行单个处理步骤

        Args:
            task: 视频任务
            step_name: 步骤名称

        Returns:
            步骤执行结果
        """
        try:
            paths = task.paths

            if step_name == "preprocess_subtitle":
                result = preprocess_subtitle(
                    str(paths.subtitle_path), str(paths.output_dir)
                )
                return {"success": True, "result": result}

            elif step_name == "separate_media":
                result = separate_media(
                    str(paths.video_path), str(paths.media_separation_dir)
                )
                return {"success": True, "result": result}

            elif step_name == "generate_reference_audio":
                result = generate_reference_audio(
                    str(paths.vocal_audio),
                    str(paths.processed_subtitle),
                    str(paths.reference_audio_dir),
                )
                return {"success": True, "result": result}

            elif step_name == "generate_tts":
                # 检查参考结果文件是否存在
                if not paths.reference_results.exists():
                    return {
                        "success": False,
                        "error": f"参考结果文件不存在: {paths.reference_results}",
                    }

                result = generate_tts_from_reference(
                    str(paths.reference_results), str(paths.tts_output_dir),
                )
                return {
                    "success": result.get("success", False),
                    "result": result,
                    "error": result.get("error"),
                }

            elif step_name == "align_audio":
                from core.audio_align_processor import align_audio_with_subtitles

                result = align_audio_with_subtitles(
                    tts_results_path=str(paths.tts_results),
                    srt_path=str(paths.processed_subtitle),
                    output_path=str(paths.aligned_audio),
                )

                # 保存对齐结果到JSON文件
                align_result_copy = result.copy()
                align_result_copy["saved_at"] = datetime.now().isoformat()
                with open(paths.aligned_results, "w", encoding="utf-8") as f:
                    json.dump(align_result_copy, f, ensure_ascii=False, indent=2)

                return {"success": True, "result": result}

            elif step_name == "generate_aligned_srt":
                from core.audio_align_processor import generate_aligned_srt
                from core.subtitle.subtitle_processor import (
                    convert_subtitle,
                    sync_srt_timestamps_to_ass,
                )

                # 生成对齐后的SRT字幕
                generate_aligned_srt(
                    str(paths.aligned_results),
                    str(paths.processed_subtitle),
                    str(paths.aligned_srt),
                )

                # 检查原始字幕格式，如果不是SRT则需要转换为相应格式
                original_subtitle_path = str(paths.subtitle_path)
                original_subtitle_ext = Path(original_subtitle_path).suffix.lower()

                if original_subtitle_ext != ".srt":
                    self.logger.info(
                        f"检测到原始字幕格式为 {original_subtitle_ext}，正在转换..."
                    )

                    if original_subtitle_ext == ".ass":
                        # 对于ASS格式，使用sync_srt_timestamps_to_ass方法同步时间戳
                        aligned_ass_path = (
                                paths.output_dir
                                / f"{Path(task.video_path).stem}_aligned.ass"
                        )
                        sync_success = sync_srt_timestamps_to_ass(
                            original_subtitle_path,
                            str(paths.aligned_srt),
                            str(aligned_ass_path),
                        )
                        if sync_success:
                            self.logger.info(
                                f"ASS字幕时间戳同步完成: {aligned_ass_path}"
                            )
                        else:
                            self.logger.warning("ASS字幕时间戳同步失败")
                    else:
                        # 对于其他格式，使用convert_subtitle转换
                        aligned_subtitle_path = (
                                paths.output_dir
                                / f"{Path(task.video_path).stem}_aligned{original_subtitle_ext}"
                        )
                        convert_success = convert_subtitle(
                            str(paths.aligned_srt), str(aligned_subtitle_path)
                        )
                        if convert_success:
                            self.logger.info(
                                f"字幕格式转换完成: {aligned_subtitle_path}"
                            )
                        else:
                            self.logger.warning("字幕格式转换失败")

                return {"success": True}

            elif step_name == "process_video_speed":
                from core.audio_align_processor import process_video_speed_adjustment

                process_video_speed_adjustment(
                    str(paths.silent_video),
                    str(paths.processed_subtitle),
                    str(paths.aligned_srt),
                )
                return {"success": True}

            elif step_name == "merge_audio_video":
                merge_audio_video(
                    str(paths.speed_adjusted_video),
                    str(paths.aligned_audio),
                    str(paths.final_video),
                )
                return {"success": True}

            else:
                return {"success": False, "error": f"未知步骤: {step_name}"}

        except Exception as e:
            self.logger.error(f"步骤 {step_name} 执行失败: {e}")
            return {"success": False, "error": str(e)}

    def get_processing_status(self) -> Dict[str, Any]:
        """
        获取当前处理状态

        Returns:
            处理状态信息
        """
        with self.task_lock:
            active_count = len(self.active_tasks)
            status_counts = {}
            step_counts = {}

            for task in self.active_tasks.values():
                status = task.status
                status_counts[status] = status_counts.get(status, 0) + 1

                current_step = task.current_step
                step_counts[current_step] = step_counts.get(current_step, 0) + 1

            return {
                "active_tasks": active_count,
                "status_distribution": status_counts,
                "step_distribution": step_counts,
                "resource_usage": {
                    "gpu_available": self.resource_pool.gpu_semaphore._value,
                    "cpu_available": self.resource_pool.cpu_semaphore._value,
                    "io_available": self.resource_pool.io_semaphore._value,
                },
            }

    def optimize_cache_for_parallel(self, video_path: str) -> Dict[str, Any]:
        """
        为并行处理优化缓存

        Args:
            video_path: 视频文件路径

        Returns:
            优化结果
        """
        try:
            paths = self._get_file_paths(video_path)
            cache_file = paths.pipeline_cache

            if not cache_file.exists():
                return {
                    "success": True,
                    "message": "缓存文件不存在，无需优化",
                    "cache_file": str(cache_file),
                    "optimized": False,
                }

            # 检查缓存大小
            file_size = cache_file.stat().st_size
            if file_size > 512 * 1024:  # 大于512KB
                self.logger.info(f"检测到大型缓存文件 {file_size} bytes，开始优化...")

                # 使用父类的缓存清理方法
                cleanup_result = self.cleanup_large_cache_file(video_path)

                return {
                    "success": True,
                    "message": "缓存已为并行处理优化",
                    "cache_file": str(cache_file),
                    "original_size": file_size,
                    "optimized": True,
                    "cleanup_result": cleanup_result,
                }
            else:
                return {
                    "success": True,
                    "message": "缓存文件大小正常，无需优化",
                    "cache_file": str(cache_file),
                    "original_size": file_size,
                    "optimized": False,
                }

        except Exception as e:
            self.logger.error(f"优化缓存失败: {e}")
            return {
                "success": False,
                "message": f"优化缓存失败: {str(e)}",
                "error": str(e),
            }

    def preheat_cache_for_batch(
            self, video_subtitle_pairs: List[Tuple[str, Optional[str]]]
    ) -> Dict[str, Any]:
        """
        预热批量处理的缓存

        Args:
            video_subtitle_pairs: 包含(video_path, subtitle_path)元组的列表

        Returns:
            预热结果
        """
        cache_stats = {
            "total_videos": len(video_subtitle_pairs),
            "cache_found": 0,
            "cache_missing": 0,
            "cache_optimized": 0,
            "cache_errors": 0,
        }

        for video_path, subtitle_path in video_subtitle_pairs:
            try:
                paths = self._get_file_paths(video_path, subtitle_path)
                cache_file = paths.pipeline_cache

                if cache_file.exists():
                    cache_stats["cache_found"] += 1

                    # 检查并优化缓存
                    optimize_result = self.optimize_cache_for_parallel(video_path)
                    if optimize_result.get("optimized", False):
                        cache_stats["cache_optimized"] += 1
                else:
                    cache_stats["cache_missing"] += 1

            except Exception as e:
                self.logger.warning(f"预热缓存失败 {video_path}: {e}")
                cache_stats["cache_errors"] += 1

        return {
            "success": True,
            "message": f'缓存预热完成: {cache_stats["cache_found"]} 个已缓存, {cache_stats["cache_missing"]} 个新任务',
            "cache_stats": cache_stats,
        }

    def get_detailed_progress(self) -> Dict[str, Any]:
        """
        获取详细的处理进度信息

        Returns:
            详细进度信息
        """
        with self.task_lock:
            if not self.active_tasks:
                return {
                    "active": False,
                    "message": "当前没有活动的处理任务",
                    "tasks": [],
                    "summary": {
                        "total": 0,
                        "completed": 0,
                        "processing": 0,
                        "failed": 0,
                        "pending": 0,
                    },
                }

            tasks_detail = []
            summary = {
                "total": 0,
                "completed": 0,
                "processing": 0,
                "failed": 0,
                "pending": 0,
            }

            for task in self.active_tasks.values():
                task_info = {
                    "task_id": task.task_id,
                    "video_path": task.video_path,
                    "status": task.status,
                    "current_step": task.current_step,
                    "total_steps": len(self.PROCESSING_STEPS),
                    "progress_percent": round(
                        (task.current_step / len(self.PROCESSING_STEPS)) * 100, 1
                    ),
                    "processing_time": round(time.time() - task.start_time, 2),
                    "completed_steps": list(task.completed_steps.keys()),
                    "error_message": task.error_message,
                }

                tasks_detail.append(task_info)
                summary["total"] += 1
                summary[task.status] += 1

            # 计算总体进度
            total_steps_possible = summary["total"] * len(self.PROCESSING_STEPS)
            completed_steps_total = sum(
                task.current_step for task in self.active_tasks.values()
            )
            overall_progress = (
                round((completed_steps_total / total_steps_possible) * 100, 1)
                if total_steps_possible > 0
                else 0
            )

            return {
                "active": True,
                "message": f'正在处理 {summary["total"]} 个任务，总体进度 {overall_progress}%',
                "overall_progress": overall_progress,
                "summary": summary,
                "tasks": tasks_detail,
                "resource_usage": {
                    "gpu_available": self.resource_pool.gpu_semaphore._value,
                    "cpu_available": self.resource_pool.cpu_semaphore._value,
                    "io_available": self.resource_pool.io_semaphore._value,
                    "gpu_queue_size": 2 - self.resource_pool.gpu_semaphore._value,
                    "cpu_queue_size": 4 - self.resource_pool.cpu_semaphore._value,
                    "io_queue_size": 8 - self.resource_pool.io_semaphore._value,
                },
                "step_distribution": self._get_step_distribution(),
            }

    def _get_step_distribution(self) -> Dict[str, int]:
        """
        获取当前步骤分布统计

        Returns:
            步骤分布统计
        """
        step_counts = {}

        for task in self.active_tasks.values():
            if task.status == "processing" and task.current_step > 0:
                step_name = self.PROCESSING_STEPS[task.current_step - 1][0]
                step_counts[step_name] = step_counts.get(step_name, 0) + 1

        return step_counts

    def log_processing_summary(self, results: List[Dict[str, Any]]) -> None:
        """
        记录处理总结

        Args:
            results: 处理结果列表
        """
        if not results:
            return

        total = len(results)
        success = sum(1 for r in results if r.get("success", False))
        failed = total - success

        # 统计处理时间
        processing_times = [
            r.get("processing_time", 0) for r in results if r.get("processing_time")
        ]
        avg_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0
        )
        max_time = max(processing_times) if processing_times else 0
        min_time = min(processing_times) if processing_times else 0

        self.logger.info("=== 批量处理总结 ===")
        self.logger.info(f"总任务数: {total}")
        self.logger.info(f"成功: {success}, 失败: {failed}")
        self.logger.info(f"成功率: {success / total * 100:.1f}%")

        if processing_times:
            self.logger.info(f"平均处理时间: {avg_time:.2f}s")
            self.logger.info(f"最短处理时间: {min_time:.2f}s")
            self.logger.info(f"最长处理时间: {max_time:.2f}s")

        # 记录失败任务
        failed_tasks = [r for r in results if not r.get("success", False)]
        if failed_tasks:
            self.logger.warning(f"失败任务 ({len(failed_tasks)} 个):")
            for task in failed_tasks:
                video_name = Path(task.get("video_path", "unknown")).stem
                error = task.get("error", "未知错误")
                self.logger.warning(f"  - {video_name}: {error}")

    def export_processing_report(
            self, results: List[Dict[str, Any]], output_path: str
    ) -> Dict[str, Any]:
        """
        导出处理报告

        Args:
            results: 处理结果列表
            output_path: 输出文件路径

        Returns:
            导出结果
        """
        try:
            report = {
                "export_time": datetime.now().isoformat(),
                "total_tasks": len(results),
                "success_count": sum(1 for r in results if r.get("success", False)),
                "failed_count": sum(1 for r in results if not r.get("success", False)),
                "tasks": results,
                "summary": self._generate_processing_summary(results),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            self.logger.info(f"处理报告已导出到: {output_path}")
            return {
                "success": True,
                "message": f"报告已导出到: {output_path}",
                "report_file": output_path,
            }

        except Exception as e:
            self.logger.error(f"导出报告失败: {e}")
            return {
                "success": False,
                "message": f"导出报告失败: {str(e)}",
                "error": str(e),
            }

    def _generate_processing_summary(
            self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        生成处理摘要

        Args:
            results: 处理结果列表

        Returns:
            处理摘要
        """
        if not results:
            return {}

        total = len(results)
        success = sum(1 for r in results if r.get("success", False))

        processing_times = [
            r.get("processing_time", 0) for r in results if r.get("processing_time")
        ]

        return {
            "total_tasks": total,
            "success_rate": f"{success / total * 100:.1f}%",
            "average_processing_time": (
                f"{sum(processing_times) / len(processing_times):.2f}s"
                if processing_times
                else "N/A"
            ),
            "total_processing_time": f"{sum(processing_times):.2f}s",
            "failed_tasks": [
                {"video_path": r.get("video_path"), "error": r.get("error", "未知错误")}
                for r in results
                if not r.get("success", False)
            ],
        }
