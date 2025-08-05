"""
任务缓存管理器

提供统一的任务缓存保存、加载和转换功能
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .cache_types import CacheVersion, CacheMetadata


class TaskCacheManager:
    """任务缓存管理器"""
    
    def __init__(self):
        """初始化缓存管理器"""
        self.logger = logging.getLogger(__name__)
    
    def save_task_cache(self, task, cache_file_path: Path, metadata: Optional[CacheMetadata] = None) -> bool:
        """
        保存任务缓存
        
        Args:
            task: 任务对象（必须有to_dict方法）
            cache_file_path: 缓存文件路径
            metadata: 缓存元数据
            
        Returns:
            是否保存成功
        """
        try:
            # 创建或更新元数据
            if metadata is None:
                metadata = CacheMetadata(
                    task_id=task.task_id,
                    video_path=task.video_path
                )
            else:
                metadata.update_save_time()
            
            # 构建缓存数据
            cache_data = {
                "metadata": metadata.to_dict(),
                "task": task.to_dict(),
            }
            
            # 确保缓存目录存在
            cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存到文件
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"任务缓存已保存: {cache_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存任务缓存失败: {e}")
            return False
    
    def load_task_cache(self, cache_file_path: Path, video_path: str, 
                       subtitle_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        加载任务缓存
        
        Args:
            cache_file_path: 缓存文件路径
            video_path: 视频文件路径
            subtitle_path: 字幕文件路径
            
        Returns:
            缓存数据字典或None
        """
        if not cache_file_path.exists():
            self.logger.debug(f"缓存文件不存在: {cache_file_path}")
            return None
            
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 检查缓存版本
            metadata = cache_data.get("metadata", {})
            version = metadata.get("version", CacheVersion.V1_0.value)
            
            if version == CacheVersion.V2_0.value:
                # 新版本缓存格式
                self.logger.info(f"加载新版本缓存: {cache_file_path}")
                return cache_data
            else:
                # 旧版本缓存格式，需要转换
                self.logger.info(f"转换旧版本缓存: {cache_file_path}")
                return self._convert_from_old_cache(cache_data, video_path, subtitle_path)
                
        except Exception as e:
            self.logger.error(f"加载任务缓存失败: {e}")
            return None
    
    def _convert_from_old_cache(self, old_cache_data: Dict[str, Any], 
                               video_path: str, subtitle_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        将旧版本缓存格式转换为新版本格式
        
        Args:
            old_cache_data: 旧版本缓存数据
            video_path: 视频文件路径
            subtitle_path: 字幕文件路径
            
        Returns:
            转换后的缓存数据
        """
        try:
            # 步骤名称到步骤ID的映射
            step_name_to_id = {
                "preprocess_subtitle": 0,
                "separate_media": 1, 
                "generate_reference_audio": 2,
                "generate_tts": 3,
                "align_audio": 4,
                "generate_aligned_srt": 5,
                "process_video_speed": 6,
                "merge_audio_video": 7,
            }
            
            # 创建新的任务数据结构
            task_id = f"converted_{Path(video_path).stem}_{int(time.time())}"
            current_step = 0
            
            # 转换完成的步骤
            step_results = {}
            step_details = {}
            completed_steps = old_cache_data.get("completed_steps", {})
            
            for step_name, step_info in completed_steps.items():
                if step_info.get("completed", False):
                    step_id = step_name_to_id.get(step_name)
                    if step_id is not None:
                        # 创建步骤结果
                        step_results[str(step_id)] = {
                            "success": True,
                            "message": f"从缓存恢复: {step_name}",
                            "data": step_info.get("result", {}),
                            "error": None,
                            "processing_time": 0.0,
                            "partial_success": False,
                            "step_detail": None,
                        }
                        
                        # 创建步骤详情
                        step_details[str(step_id)] = {
                            "step_id": step_id,
                            "step_name": step_name,
                            "status": "completed",
                            "current_item": 1,
                            "total_items": 1,
                            "progress_percent": 100.0,
                            "completed_items": [{"item_id": 0, "completed_at": step_info.get("completed_at", "")}],
                            "pending_items": [],
                            "failed_items": [],
                            "metadata": {},
                        }
                        
                        current_step = max(current_step, step_id + 1)
            
            # 构建新的任务数据
            task_data = {
                "task_id": task_id,
                "video_path": video_path,
                "subtitle_path": subtitle_path,
                "current_step": current_step,
                "status": "pending",
                "progress": (current_step / 8.0 * 100.0),
                "step_results": step_results,
                "step_details": step_details,
                "error_message": None,
                "retry_count": 0,
                "max_retries": 3,
                "created_at": old_cache_data.get("created_at", ""),
                "updated_at": old_cache_data.get("updated_at", ""),
                "started_at": None,
                "completed_at": None,
                "paths": old_cache_data.get("file_paths", {}),
            }
            
            # 创建元数据
            metadata = CacheMetadata(
                version=CacheVersion.V2_0.value,
                created_at=old_cache_data.get("created_at", ""),
                task_id=task_id,
                video_path=video_path,
                completed_steps=current_step,
            )
            
            converted_data = {
                "metadata": metadata.to_dict(),
                "task": task_data,
            }
            
            self.logger.info(f"成功转换旧缓存格式，已完成步骤: {current_step}/8")
            return converted_data
            
        except Exception as e:
            self.logger.error(f"转换旧缓存格式失败: {e}")
            return None
    
    def clear_cache(self, cache_file_path: Path) -> bool:
        """
        清理缓存文件
        
        Args:
            cache_file_path: 缓存文件路径
            
        Returns:
            是否清理成功
        """
        try:
            if cache_file_path.exists():
                cache_file_path.unlink()
                self.logger.info(f"缓存文件已清理: {cache_file_path}")
            return True
        except Exception as e:
            self.logger.error(f"清理缓存文件失败: {e}")
            return False
    
    def get_cache_info(self, cache_file_path: Path) -> Optional[CacheMetadata]:
        """
        获取缓存信息
        
        Args:
            cache_file_path: 缓存文件路径
            
        Returns:
            缓存元数据或None
        """
        if not cache_file_path.exists():
            return None
            
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            metadata_data = cache_data.get("metadata", {})
            return CacheMetadata.from_dict(metadata_data)
            
        except Exception as e:
            self.logger.error(f"获取缓存信息失败: {e}")
            return None