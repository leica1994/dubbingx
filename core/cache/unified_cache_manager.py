"""
统一缓存管理器

提供统一的缓存目录结构和实时状态保存，支持断点继续
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .cache_types import CacheMetadata, CacheVersion


class UnifiedCacheManager:
    """统一缓存管理器 - 将所有步骤缓存文件放在统一目录下"""

    def __init__(self, base_output_dir: Path):
        """
        初始化统一缓存管理器

        Args:
            base_output_dir: 视频输出基础目录（如 outputs/video_name/）
        """
        self.logger = logging.getLogger(__name__)
        self.base_output_dir = base_output_dir
        
        # 创建统一缓存目录
        self.cache_dir = base_output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径定义
        self.task_state_file = self.cache_dir / "task_state.json"
        self.step_status_file = self.cache_dir / "step_status.json"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 内存中的状态缓存（减少文件I/O）
        self._memory_cache = {
            "task_state": {},
            "step_status": {},
            "metadata": {},
        }
        self._cache_loaded = False

    def initialize_cache(self, task_id: str, video_path: str, subtitle_path: Optional[str] = None) -> bool:
        """
        初始化缓存结构

        Args:
            task_id: 任务ID
            video_path: 视频文件路径
            subtitle_path: 字幕文件路径

        Returns:
            是否初始化成功
        """
        with self._lock:
            try:
                # 加载现有缓存
                self._load_cache()
                
                # 初始化元数据
                if not self._memory_cache["metadata"]:
                    metadata = CacheMetadata(
                        task_id=task_id,
                        video_path=video_path,
                        total_steps=8,
                        completed_steps=0,
                    )
                    self._memory_cache["metadata"] = metadata.to_dict()
                    self._save_metadata()

                # 初始化任务状态
                if not self._memory_cache["task_state"]:
                    task_state = {
                        "task_id": task_id,
                        "video_path": video_path,
                        "subtitle_path": subtitle_path,
                        "current_step": 0,
                        "status": "pending",
                        "progress": 0.0,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "started_at": None,
                        "completed_at": None,
                        "error_message": None,
                        "retry_count": 0,
                        "max_retries": 3,
                    }
                    self._memory_cache["task_state"] = task_state
                    self._save_task_state()

                # 初始化步骤状态（8个步骤）
                if not self._memory_cache["step_status"]:
                    step_status = {}
                    for step_id in range(8):
                        step_status[str(step_id)] = {
                            "step_id": step_id,
                            "status": "pending",
                            "started_at": None,
                            "completed_at": None,
                            "progress_percent": 0.0,
                            "current_item": 0,
                            "total_items": 1,
                            "error_message": None,
                            "result_files": [],  # 该步骤生成的文件列表
                            "metadata": {},
                        }
                    self._memory_cache["step_status"] = step_status
                    self._save_step_status()

                self.logger.info(f"缓存结构初始化完成: {self.cache_dir}")
                return True

            except Exception as e:
                self.logger.error(f"初始化缓存结构失败: {e}")
                return False

    def update_step_status(
        self,
        step_id: int,
        status: str,
        progress_percent: Optional[float] = None,
        current_item: Optional[int] = None,
        total_items: Optional[int] = None,
        error_message: Optional[str] = None,
        result_files: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        实时更新步骤状态

        Args:
            step_id: 步骤ID (0-7)
            status: 状态 (pending/processing/completed/failed)
            progress_percent: 进度百分比
            current_item: 当前处理项目
            total_items: 总项目数
            error_message: 错误消息
            result_files: 生成的结果文件列表
            metadata: 额外元数据

        Returns:
            是否更新成功
        """
        with self._lock:
            try:
                self._load_cache()
                
                step_key = str(step_id)
                if step_key not in self._memory_cache["step_status"]:
                    self.logger.warning(f"步骤 {step_id} 不存在，跳过状态更新")
                    return False

                step_info = self._memory_cache["step_status"][step_key]
                
                # 更新状态字段
                step_info["status"] = status
                if progress_percent is not None:
                    step_info["progress_percent"] = progress_percent
                if current_item is not None:
                    step_info["current_item"] = current_item
                if total_items is not None:
                    step_info["total_items"] = total_items
                if error_message is not None:
                    step_info["error_message"] = error_message
                if result_files is not None:
                    step_info["result_files"] = result_files
                if metadata is not None:
                    step_info["metadata"].update(metadata)

                # 更新时间戳
                now = datetime.now().isoformat()
                if status == "processing" and not step_info["started_at"]:
                    step_info["started_at"] = now
                elif status in ["completed", "failed"]:
                    step_info["completed_at"] = now

                # 实时保存到文件
                self._save_step_status()
                
                # 更新任务整体进度
                self._update_task_progress()

                self.logger.debug(f"步骤 {step_id} 状态已更新: {status} ({progress_percent}%)")
                return True

            except Exception as e:
                self.logger.error(f"更新步骤状态失败: {e}")
                return False

    def save_step_result(
        self,
        step_id: int,
        result_data: Dict[str, Any],
        result_files: Optional[List[str]] = None,
    ) -> bool:
        """
        保存步骤处理结果

        Args:
            step_id: 步骤ID
            result_data: 结果数据
            result_files: 生成的文件列表

        Returns:
            是否保存成功
        """
        with self._lock:
            try:
                # 保存结果数据到单独文件
                result_file = self.cache_dir / f"step_{step_id}_result.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)

                # 更新步骤状态
                self.update_step_status(
                    step_id=step_id,
                    status="completed",
                    progress_percent=100.0,
                    result_files=result_files or [],
                )

                self.logger.info(f"步骤 {step_id} 结果已保存: {result_file}")
                return True

            except Exception as e:
                self.logger.error(f"保存步骤结果失败: {e}")
                return False

    def load_step_result(self, step_id: int) -> Optional[Dict[str, Any]]:
        """
        加载步骤处理结果

        Args:
            step_id: 步骤ID

        Returns:
            结果数据或None
        """
        try:
            result_file = self.cache_dir / f"step_{step_id}_result.json"
            if not result_file.exists():
                return None

            with open(result_file, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception as e:
            self.logger.error(f"加载步骤结果失败: {e}")
            return None

    def get_task_resume_point(self) -> int:
        """
        获取任务恢复点（下一个要执行的步骤）

        Returns:
            下一步骤ID (0-8, 8表示全部完成)
        """
        with self._lock:
            try:
                self._load_cache()
                
                step_status = self._memory_cache["step_status"]
                for step_id in range(8):
                    step_key = str(step_id)
                    if step_key in step_status:
                        status = step_status[step_key]["status"]
                        if status not in ["completed"]:
                            return step_id
                
                # 所有步骤都已完成
                return 8

            except Exception as e:
                self.logger.error(f"获取任务恢复点失败: {e}")
                return 0

    def get_completed_steps(self) -> Set[int]:
        """
        获取已完成的步骤集合

        Returns:
            已完成步骤的ID集合
        """
        with self._lock:
            try:
                self._load_cache()
                
                completed_steps = set()
                step_status = self._memory_cache["step_status"]
                
                for step_id in range(8):
                    step_key = str(step_id)
                    if (step_key in step_status and 
                        step_status[step_key]["status"] == "completed"):
                        completed_steps.add(step_id)
                
                return completed_steps

            except Exception as e:
                self.logger.error(f"获取已完成步骤失败: {e}")
                return set()

    def get_step_status_summary(self) -> Dict[str, Any]:
        """
        获取步骤状态摘要（用于GUI显示）

        Returns:
            状态摘要字典
        """
        with self._lock:
            try:
                self._load_cache()
                
                summary = {
                    "total_steps": 8,
                    "completed_steps": 0,
                    "failed_steps": 0,
                    "processing_steps": 0,
                    "pending_steps": 0,
                    "overall_progress": 0.0,
                    "current_step": 0,
                    "step_details": {},
                }
                
                step_status = self._memory_cache["step_status"]
                
                for step_id in range(8):
                    step_key = str(step_id)
                    if step_key in step_status:
                        step_info = step_status[step_key]
                        status = step_info["status"]
                        
                        if status == "completed":
                            summary["completed_steps"] += 1
                        elif status == "failed":
                            summary["failed_steps"] += 1
                        elif status == "processing":
                            summary["processing_steps"] += 1
                            summary["current_step"] = step_id
                        else:  # pending
                            summary["pending_steps"] += 1
                            if summary["current_step"] == 0 and step_id == 0:
                                summary["current_step"] = step_id
                        
                        summary["step_details"][step_key] = {
                            "step_id": step_id,
                            "status": status,
                            "progress": step_info["progress_percent"],
                            "current_item": step_info["current_item"],
                            "total_items": step_info["total_items"],
                            "error": step_info.get("error_message"),
                        }
                
                # 计算整体进度
                summary["overall_progress"] = (summary["completed_steps"] / 8) * 100
                
                return summary

            except Exception as e:
                self.logger.error(f"获取状态摘要失败: {e}")
                return {"error": str(e)}

    def clear_cache(self) -> bool:
        """
        清理所有缓存文件

        Returns:
            是否清理成功
        """
        with self._lock:
            try:
                # 清理内存缓存
                self._memory_cache = {
                    "task_state": {},
                    "step_status": {},
                    "metadata": {},
                }
                self._cache_loaded = False
                
                # 删除缓存文件
                if self.cache_dir.exists():
                    for cache_file in self.cache_dir.glob("*.json"):
                        cache_file.unlink()
                    
                    # 如果目录为空则删除
                    if not any(self.cache_dir.iterdir()):
                        self.cache_dir.rmdir()
                
                self.logger.info(f"缓存已清理: {self.cache_dir}")
                return True

            except Exception as e:
                self.logger.error(f"清理缓存失败: {e}")
                return False

    # 私有方法

    def _load_cache(self) -> None:
        """从文件加载缓存到内存（如果还未加载）"""
        if self._cache_loaded:
            return
        
        try:
            # 加载元数据
            if self.metadata_file.exists():
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self._memory_cache["metadata"] = json.load(f)
            
            # 加载任务状态
            if self.task_state_file.exists():
                with open(self.task_state_file, "r", encoding="utf-8") as f:
                    self._memory_cache["task_state"] = json.load(f)
            
            # 加载步骤状态
            if self.step_status_file.exists():
                with open(self.step_status_file, "r", encoding="utf-8") as f:
                    self._memory_cache["step_status"] = json.load(f)
            
            self._cache_loaded = True
            self.logger.debug("缓存已加载到内存")

        except Exception as e:
            self.logger.warning(f"加载缓存失败，使用空缓存: {e}")
            self._memory_cache = {
                "task_state": {},
                "step_status": {},
                "metadata": {},
            }
            self._cache_loaded = True

    def _save_metadata(self) -> None:
        """保存元数据到文件"""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self._memory_cache["metadata"], f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存元数据失败: {e}")

    def _save_task_state(self) -> None:
        """保存任务状态到文件"""
        try:
            # 更新时间戳
            if "updated_at" in self._memory_cache["task_state"]:
                self._memory_cache["task_state"]["updated_at"] = datetime.now().isoformat()
            
            with open(self.task_state_file, "w", encoding="utf-8") as f:
                json.dump(self._memory_cache["task_state"], f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存任务状态失败: {e}")

    def _save_step_status(self) -> None:
        """保存步骤状态到文件"""
        try:
            with open(self.step_status_file, "w", encoding="utf-8") as f:
                json.dump(self._memory_cache["step_status"], f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存步骤状态失败: {e}")

    def _update_task_progress(self) -> None:
        """更新任务整体进度"""
        try:
            completed_steps = len(self.get_completed_steps())
            progress = (completed_steps / 8) * 100
            
            task_state = self._memory_cache["task_state"]
            task_state["progress"] = progress
            task_state["current_step"] = self.get_task_resume_point()
            
            # 更新元数据中的完成步骤数
            if self._memory_cache["metadata"]:
                self._memory_cache["metadata"]["completed_steps"] = completed_steps
                self._memory_cache["metadata"]["saved_at"] = datetime.now().isoformat()
                self._save_metadata()
            
            self._save_task_state()

        except Exception as e:
            self.logger.error(f"更新任务进度失败: {e}")