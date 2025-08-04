#!/usr/bin/env python3
"""
验证 process_batch 方法删除后的测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_process_batch_removal():
    """测试 process_batch 方法已被删除"""
    try:
        from core.dubbing_pipeline import DubbingPipeline, ParallelDubbingPipeline
        
        # 创建基础流水线实例
        basic_pipeline = DubbingPipeline()
        
        # 检查基础流水线是否有 process_batch 方法
        if hasattr(basic_pipeline, 'process_batch'):
            print("ERROR: 基础 DubbingPipeline 仍然有 process_batch 方法")
            return False
        else:
            print("OK: 基础 DubbingPipeline 的 process_batch 方法已删除")
        
        # 检查并行流水线是否有 process_batch_parallel 方法
        parallel_pipeline = ParallelDubbingPipeline()
        if hasattr(parallel_pipeline, 'process_batch_parallel'):
            print("OK: ParallelDubbingPipeline 有 process_batch_parallel 方法")
        else:
            print("ERROR: ParallelDubbingPipeline 缺少 process_batch_parallel 方法")
            return False
        
        # 检查其他关键方法
        key_methods = ['process_video', 'get_processing_status', 'get_detailed_progress']
        for method in key_methods:
            if hasattr(parallel_pipeline, method):
                print(f"OK: {method} 方法存在")
            else:
                print(f"ERROR: {method} 方法不存在")
                return False
        
        print("SUCCESS: 所有检查通过，process_batch 方法已成功删除")
        return True
        
    except Exception as e:
        print(f"ERROR: 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_process_batch_removal()
    sys.exit(0 if success else 1)