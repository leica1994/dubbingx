"""
步骤缓存管理器 - 统一缓存系统的处理器适配器

为现有处理器提供兼容的缓存接口，内部使用UnifiedCacheManager
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .unified_cache_manager import UnifiedCacheManager


class StepCacheManager:
    """步骤缓存管理器 - 处理器专用的缓存接口"""

    def __init__(self, unified_cache_manager: UnifiedCacheManager, step_id: int, step_name: str):
        """
        初始化步骤缓存管理器

        Args:
            unified_cache_manager: 统一缓存管理器实例
            step_id: 步骤ID (0-7)
            step_name: 步骤名称
        """
        self.logger = logging.getLogger(f"{__name__}.Step{step_id}")
        self.unified_cache = unified_cache_manager
        self.step_id = step_id
        self.step_name = step_name

    def is_step_completed(self) -> bool:
        """
        检查步骤是否已完成

        Returns:
            是否已完成
        """
        completed_steps = self.unified_cache.get_completed_steps()
        return self.step_id in completed_steps

    def save_step_result(
        self,
        result_data: Dict[str, Any],
        result_files: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        保存步骤处理结果

        Args:
            result_data: 结果数据
            result_files: 生成的文件列表
            metadata: 额外元数据

        Returns:
            是否保存成功
        """
        # 保存结果数据
        success = self.unified_cache.save_step_result(
            step_id=self.step_id,
            result_data=result_data,
            result_files=result_files,
        )
        
        if success and metadata:
            # 更新元数据
            self.unified_cache.update_step_status(
                step_id=self.step_id,
                status="completed",
                metadata=metadata,
            )
        
        self.logger.info(f"步骤 {self.step_name} 结果保存: {'成功' if success else '失败'}")
        return success

    def load_step_result(self) -> Optional[Dict[str, Any]]:
        """
        加载步骤处理结果

        Returns:
            结果数据或None
        """
        result = self.unified_cache.load_step_result(self.step_id)
        if result:
            self.logger.debug(f"步骤 {self.step_name} 结果已加载")
        else:
            self.logger.debug(f"步骤 {self.step_name} 无缓存结果")
        return result

    def update_progress(
        self,
        current_item: int,
        total_items: int,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        更新步骤处理进度

        Args:
            current_item: 当前处理项目编号
            total_items: 总项目数
            message: 进度消息
            metadata: 额外元数据

        Returns:
            是否更新成功
        """
        # 计算进度百分比
        progress_percent = 0.0
        if total_items > 0:
            progress_percent = (current_item / total_items) * 100.0
        
        # 确定状态
        status = "processing"
        if current_item >= total_items:
            status = "completed"
        elif current_item == 0:
            status = "pending"

        success = self.unified_cache.update_step_status(
            step_id=self.step_id,
            status=status,
            progress_percent=progress_percent,
            current_item=current_item,
            total_items=total_items,
            metadata=metadata,
        )
        
        if success:
            self.logger.debug(
                f"步骤 {self.step_name} 进度更新: {current_item}/{total_items} ({progress_percent:.1f}%)"
            )
        
        return success

    def mark_step_started(self, total_items: int = 1, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        标记步骤开始处理

        Args:
            total_items: 预期处理的总项目数
            metadata: 元数据

        Returns:
            是否标记成功
        """
        success = self.unified_cache.update_step_status(
            step_id=self.step_id,
            status="processing",
            progress_percent=0.0,
            current_item=0,
            total_items=total_items,
            metadata=metadata or {},
        )
        
        if success:
            self.logger.info(f"步骤 {self.step_name} 开始处理 (预期 {total_items} 项)")
        
        return success

    def mark_step_failed(self, error_message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        标记步骤处理失败

        Args:
            error_message: 错误消息
            metadata: 元数据

        Returns:
            是否标记成功
        """
        success = self.unified_cache.update_step_status(
            step_id=self.step_id,
            status="failed",
            error_message=error_message,
            metadata=metadata or {},
        )
        
        if success:
            self.logger.error(f"步骤 {self.step_name} 处理失败: {error_message}")
        
        return success

    def mark_step_completed(
        self,
        result_files: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        标记步骤完成

        Args:
            result_files: 生成的文件列表
            metadata: 元数据

        Returns:
            是否标记成功
        """
        success = self.unified_cache.update_step_status(
            step_id=self.step_id,
            status="completed",
            progress_percent=100.0,
            result_files=result_files or [],
            metadata=metadata or {},
        )
        
        if success:
            self.logger.info(f"步骤 {self.step_name} 完成处理")
        
        return success

    def get_step_files(self) -> List[str]:
        """
        获取步骤生成的文件列表

        Returns:
            文件路径列表
        """
        try:
            summary = self.unified_cache.get_step_status_summary()
            step_details = summary.get("step_details", {})
            step_key = str(self.step_id)
            
            if step_key in step_details:
                # 从步骤状态获取文件列表
                step_status = self.unified_cache._memory_cache.get("step_status", {})
                if step_key in step_status:
                    return step_status[step_key].get("result_files", [])
            
            return []

        except Exception as e:
            self.logger.error(f"获取步骤文件列表失败: {e}")
            return []

    def has_valid_cache(self, input_files: Optional[List[str]] = None) -> bool:
        """
        检查是否有有效缓存

        Args:
            input_files: 输入文件列表（用于检查文件时间戳）

        Returns:
            是否有有效缓存
        """
        # 简化版本：只检查步骤是否完成
        # 后续可以扩展为检查文件时间戳等
        return self.is_step_completed()

    def clear_step_cache(self) -> bool:
        """
        清理步骤缓存

        Returns:
            是否清理成功
        """
        try:
            # 清理步骤结果文件
            result_file = self.unified_cache.cache_dir / f"step_{self.step_id}_result.json"
            if result_file.exists():
                result_file.unlink()
            
            # 重置步骤状态
            success = self.unified_cache.update_step_status(
                step_id=self.step_id,
                status="pending",
                progress_percent=0.0,
                current_item=0,
                total_items=1,
                error_message=None,
                result_files=[],
                metadata={},
            )
            
            if success:
                self.logger.info(f"步骤 {self.step_name} 缓存已清理")
            
            return success

        except Exception as e:
            self.logger.error(f"清理步骤缓存失败: {e}")
            return False

    # 兼容性方法（保持与旧接口兼容）

    def get_cache_file_path(self) -> Path:
        """
        获取缓存文件路径（兼容性方法）

        Returns:
            缓存文件路径
        """
        return self.unified_cache.cache_dir / f"step_{self.step_id}_result.json"

    def load_cached_data(self) -> Optional[Dict[str, Any]]:
        """加载缓存数据（兼容性方法）"""
        return self.load_step_result()

    def save_cached_data(self, data: Dict[str, Any]) -> bool:
        """保存缓存数据（兼容性方法）"""
        return self.save_step_result(data)

    def __repr__(self) -> str:
        return f"StepCacheManager(step_id={self.step_id}, step_name='{self.step_name}')"