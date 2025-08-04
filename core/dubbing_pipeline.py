import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from core.media_processor import MediaProcessor, merge_audio_video, generate_reference_audio, separate_media
from core.subtitle_preprocessor import SubtitlePreprocessor, preprocess_subtitle
from core.tts_processor import TTSProcessor, generate_tts_from_reference
from core.audio_align_processor import (
    align_audio_with_subtitles,
    generate_aligned_srt,
    process_video_speed_adjustment
)
from core.subtitle.subtitle_processor import convert_subtitle, sync_srt_timestamps_to_ass


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

    def _get_output_dir(self, video_path: str) -> Path:
        """获取输出目录"""
        if self.output_dir:
            return self.output_dir

        # 使用视频父目录下的outputs目录
        video_parent = Path(video_path).parent
        output_dir = video_parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def _get_file_paths(self, video_path: str, subtitle_path: str) -> Dict[str, Path]:
        """生成所有需要的文件路径"""
        output_dir = self._get_output_dir(video_path)
        video_name = Path(video_path).stem

        paths = {
            'output_dir': output_dir,
            'video_path': Path(video_path),
            'subtitle_path': Path(subtitle_path),
            'processed_subtitle': output_dir / f"{video_name}_processed.srt",
            'vocal_audio': output_dir / f"{video_name}_vocal.wav",
            'background_audio': output_dir / f"{video_name}_background.wav",
            'silent_video': output_dir / f"{video_name}_silent.mp4",
            'reference_audio_dir': output_dir / "reference_audio",
            'tts_output_dir': output_dir / "tts_output",
            'aligned_audio_dir': output_dir / "aligned_audio",
            'adjusted_video_dir': output_dir / "adjusted_video",
            'reference_results': output_dir / "reference_audio" / f"{video_name}_vocal_reference_audio_results.json",
            'tts_results': output_dir / "tts_output" / "tts_generation_results.json",
            'aligned_results': output_dir / "aligned_audio" / "aligned_tts_generation_results.json",
            'aligned_audio': output_dir / "aligned_audio" / "aligned_tts_generation_results.wav",
            'aligned_srt': output_dir / "aligned_audio" / "aligned_tts_generation_aligned.srt",
            'final_video': output_dir / f"{video_name}_dubbed.mp4",
            'speed_adjusted_video': output_dir / "adjusted_video" / f"final_speed_adjusted_{video_name}_silent.mp4"
        }

        # 创建必要的子目录
        for dir_path in ['reference_audio_dir', 'tts_output_dir', 'aligned_audio_dir', 'adjusted_video_dir']:
            paths[dir_path].mkdir(exist_ok=True)

        # 添加缓存文件路径
        paths['pipeline_cache'] = output_dir / f"{video_name}_pipeline_cache.json"

        return paths

    def cleanup_large_cache_file(self, video_path: str) -> Dict[str, Any]:
        """
        清理过大的缓存文件并重新创建优化后的缓存
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            清理结果
        """
        try:
            paths = self._get_file_paths(video_path, "")
            cache_file = paths['pipeline_cache']
            
            if not cache_file.exists():
                return {
                    'success': True,
                    'message': '缓存文件不存在',
                    'cache_file': str(cache_file)
                }
            
            # 检查文件大小
            file_size = cache_file.stat().st_size
            if file_size < 1024 * 1024:  # 小于1MB，不需要清理
                return {
                    'success': True,
                    'message': f'缓存文件大小正常 ({file_size} bytes)',
                    'cache_file': str(cache_file),
                    'file_size': file_size
                }
            
            self.logger.info(f"检测到过大的缓存文件: {file_size} bytes，开始优化...")
            
            # 加载现有缓存
            old_cache = self._load_pipeline_cache(cache_file)
            if not old_cache:
                return {
                    'success': False,
                    'message': '无法加载现有缓存',
                    'cache_file': str(cache_file)
                }
            
            # 备份旧缓存
            backup_file = cache_file.with_suffix('.json.backup')
            cache_file.rename(backup_file)
            
            # 重新创建优化后的缓存
            new_cache = {
                'video_path': old_cache.get('video_path'),
                'subtitle_path': old_cache.get('subtitle_path'),
                'output_dir': old_cache.get('output_dir'),
                'created_at': old_cache.get('created_at'),
                'updated_at': datetime.now().isoformat(),
                'completed_steps': {},
                'file_paths': old_cache.get('file_paths', {}),
                'cache_optimized': True,
                'original_size': file_size
            }
            
            # 重新处理已完成步骤的结果
            completed_steps = old_cache.get('completed_steps', {})
            for step_name, step_data in completed_steps.items():
                if step_data.get('completed', False):
                    # 优化步骤结果
                    optimized_result = self._optimize_result_for_cache(step_name, step_data.get('result'))
                    
                    new_cache['completed_steps'][step_name] = {
                        'completed': True,
                        'completed_at': step_data.get('completed_at'),
                        'result': optimized_result
                    }
            
            # 保存优化后的缓存
            self._save_pipeline_cache(cache_file, new_cache)
            
            # 检查新文件大小
            new_size = cache_file.stat().st_size
            saved_space = file_size - new_size
            
            self.logger.info(f"缓存优化完成: {file_size} -> {new_size} bytes (节省 {saved_space} bytes)")
            
            return {
                'success': True,
                'message': f'缓存优化完成，节省 {saved_space} bytes',
                'cache_file': str(cache_file),
                'original_size': file_size,
                'new_size': new_size,
                'saved_space': saved_space,
                'backup_file': str(backup_file)
            }
            
        except Exception as e:
            self.logger.error(f"清理缓存文件失败: {str(e)}")
            return {
                'success': False,
                'message': f'清理缓存文件失败: {str(e)}',
                'error': str(e)
            }

    def _load_pipeline_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """加载流水线缓存"""
        try:
            if not cache_path.exists():
                self.logger.debug(f"缓存文件不存在: {cache_path}")
                return None

            with open(cache_path, 'r', encoding='utf-8') as f:
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
            required_fields = ['video_path', 'subtitle_path', 'output_dir', 'created_at']
            for field in required_fields:
                if field not in cache_data:
                    self.logger.warning(f"缓存缺少必需字段: {field}")
                    return False
            
            # 检查completed_steps结构
            completed_steps = cache_data.get('completed_steps', {})
            if not isinstance(completed_steps, dict):
                self.logger.warning("缓存中的completed_steps不是字典类型")
                return False
            
            # 检查每个步骤的数据结构
            for step_name, step_data in completed_steps.items():
                if not isinstance(step_data, dict):
                    self.logger.warning(f"步骤 {step_name} 的数据不是字典类型")
                    return False
                
                if not step_data.get('completed', False):
                    continue
                
                # 检查必需的时间字段
                if 'completed_at' not in step_data:
                    self.logger.warning(f"步骤 {step_name} 缺少completed_at字段")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"验证缓存结构时发生错误: {str(e)}")
            return False

    def _save_pipeline_cache(self, cache_path: Path, cache_data: Dict[str, Any]) -> bool:
        """保存流水线缓存"""
        try:
            cache_data['updated_at'] = datetime.now().isoformat()
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"缓存已保存: {cache_path}")
            return True
        except Exception as e:
            self.logger.warning(f"保存缓存失败: {str(e)}")
            return False

    def _check_step_completed(self, cache_data: Dict[str, Any], step_name: str) -> bool:
        """检查指定步骤是否已完成"""
        return cache_data.get('completed_steps', {}).get(step_name, {}).get('completed', False)

    def _mark_step_completed(self, cache_data: Dict[str, Any], step_name: str, result: Dict[str, Any] = None) -> None:
        """标记指定步骤为已完成"""
        if 'completed_steps' not in cache_data:
            cache_data['completed_steps'] = {}
        
        # 优化缓存数据，只存储必要信息
        optimized_result = self._optimize_result_for_cache(step_name, result)
        
        cache_data['completed_steps'][step_name] = {
            'completed': True,
            'completed_at': datetime.now().isoformat(),
            'result': optimized_result
        }

    def _optimize_result_for_cache(self, step_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """优化结果数据以减少缓存大小"""
        if not result:
            return {}
        
        optimized = {'success': result.get('success', False)}
        
        if step_name == 'preprocess_subtitle':
            # 只存储处理后的字幕文件路径，不存储完整的字幕条目
            if result.get('result') and result['result'].get('processed_subtitle_path'):
                optimized['processed_subtitle_path'] = result['result']['processed_subtitle_path']
                optimized['entries_count'] = len(result['result'].get('subtitle_entries', []))
        
        elif step_name == 'separate_media':
            # 只存储分离后的文件路径
            if result.get('result'):
                optimized.update({
                    'silent_video_path': result['result'].get('silent_video_path'),
                    'vocal_audio_path': result['result'].get('vocal_audio_path'),
                    'background_audio_path': result['result'].get('background_audio_path')
                })
        
        elif step_name == 'generate_reference_audio':
            # 只存储关键信息，不存储完整的音频片段信息
            if result.get('result'):
                optimized.update({
                    'output_dir': result['result'].get('output_dir'),
                    'total_segments': result['result'].get('total_segments', 0),
                    'success': result['result'].get('success', False)
                })
        
        elif step_name == 'generate_tts':
            # 只存储TTS生成结果的关键信息
            if result.get('result'):
                optimized.update({
                    'success': result['result'].get('success', False),
                    'total_segments': result['result'].get('total_segments', 0),
                    'failed_segments': result['result'].get('failed_segments', 0)
                })
        
        elif step_name == 'align_audio':
            # 只存储对齐结果的关键信息
            if result.get('result'):
                optimized.update({
                    'success': result['result'].get('success', False),
                    'total_duration': result['result'].get('total_duration', 0),
                    'segments_count': result['result'].get('segments_count', 0)
                })
        
        else:
            # 对于其他步骤，只存储成功状态
            optimized['success'] = result.get('success', False)
        
        return optimized

    def _validate_step_dependencies(self, cache_data: Dict[str, Any], step_name: str) -> bool:
        """验证步骤依赖是否满足"""
        step_dependencies = {
            'preprocess_subtitle': [],
            'separate_media': ['preprocess_subtitle'],
            'generate_reference_audio': ['preprocess_subtitle', 'separate_media'],
            'generate_tts': ['preprocess_subtitle', 'separate_media', 'generate_reference_audio'],
            'align_audio': ['preprocess_subtitle', 'separate_media', 'generate_reference_audio', 'generate_tts'],
            'generate_aligned_srt': ['preprocess_subtitle', 'separate_media', 'generate_reference_audio', 'generate_tts', 'align_audio'],
            'process_video_speed': ['preprocess_subtitle', 'separate_media', 'generate_reference_audio', 'generate_tts', 'align_audio', 'generate_aligned_srt'],
            'merge_audio_video': ['preprocess_subtitle', 'separate_media', 'generate_reference_audio', 'generate_tts', 'align_audio', 'generate_aligned_srt', 'process_video_speed']
        }

        dependencies = step_dependencies.get(step_name, [])
        for dep in dependencies:
            if not self._check_step_completed(cache_data, dep):
                self.logger.warning(f"步骤 {step_name} 的依赖 {dep} 未完成")
                return False
        
        return True

    def process_video(self, video_path: str, subtitle_path: str, resume_from_cache: bool = True) -> Dict[str, Any]:
        """
        处理视频配音的完整流程
        
        Args:
            video_path: 视频文件路径
            subtitle_path: 字幕文件路径
            resume_from_cache: 是否从缓存恢复，默认为True
            
        Returns:
            处理结果字典
        """
        try:
            self.logger.info(f"开始处理视频: {video_path}")
            self.logger.info(f"字幕文件: {subtitle_path}")

            # 获取文件路径
            paths = self._get_file_paths(video_path, subtitle_path)

            # 初始化缓存
            cache_data = {
                'video_path': video_path,
                'subtitle_path': subtitle_path,
                'output_dir': str(paths['output_dir']),
                'created_at': datetime.now().isoformat(),
                'completed_steps': {},
                'file_paths': {k: str(v) for k, v in paths.items() if k != 'output_dir'}
            }

            # 检查和修复缓存
            existing_cache = None
            if resume_from_cache:
                # 自动检查和修复缓存
                repair_result = self.check_and_repair_cache(video_path)
                self.logger.info(f"缓存检查结果: {repair_result['message']}")
                
                if repair_result.get('cache_exists'):
                    existing_cache = self._load_pipeline_cache(paths['pipeline_cache'])
                    if existing_cache:
                        cache_data = existing_cache
                        self.logger.info("使用缓存继续处理")
                    else:
                        self.logger.info("缓存加载失败，开始全新处理")
                else:
                    self.logger.info("未找到缓存，开始全新处理")
            else:
                self.logger.info("禁用缓存，开始全新处理")

            # 1. 预处理字幕
            if not self._check_step_completed(cache_data, 'preprocess_subtitle'):
                self.logger.info("步骤1: 预处理字幕")
                preprocess_result = preprocess_subtitle(str(paths['subtitle_path']), str(paths['output_dir']))
                self._mark_step_completed(cache_data, 'preprocess_subtitle', {'result': preprocess_result})
                self._save_pipeline_cache(paths['pipeline_cache'], cache_data)
            else:
                self.logger.info("步骤1: 预处理字幕 (已完成，跳过)")

            # 2. 分离媒体文件
            if not self._check_step_completed(cache_data, 'separate_media'):
                self.logger.info("步骤2: 分离音视频")
                separate_result = separate_media(str(paths['video_path']), str(paths['output_dir']))
                self._mark_step_completed(cache_data, 'separate_media', {'result': separate_result})
                self._save_pipeline_cache(paths['pipeline_cache'], cache_data)
            else:
                self.logger.info("步骤2: 分离音视频 (已完成，跳过)")

            # 3. 生成参考音频
            if not self._check_step_completed(cache_data, 'generate_reference_audio'):
                self.logger.info("步骤3: 生成参考音频")
                ref_result = generate_reference_audio(
                    str(paths['vocal_audio']),
                    str(paths['processed_subtitle']),
                    str(paths['reference_audio_dir'])
                )
                self._mark_step_completed(cache_data, 'generate_reference_audio', {'result': ref_result})
                self._save_pipeline_cache(paths['pipeline_cache'], cache_data)
            else:
                self.logger.info("步骤3: 生成参考音频 (已完成，跳过)")

            # 4. 生成TTS音频
            if not self._check_step_completed(cache_data, 'generate_tts'):
                self.logger.info("步骤4: 生成TTS音频")
                self.logger.info(f"参考结果文件: {paths['reference_results']}")
                self.logger.info(f"输出目录: {paths['tts_output_dir']}")

                # 检查参考结果文件是否存在
                if not paths['reference_results'].exists():
                    self.logger.error(f"参考结果文件不存在: {paths['reference_results']}")
                    return {
                        'success': False,
                        'message': '参考结果文件不存在',
                        'error': f'文件不存在: {paths["reference_results"]}'
                    }

                tts_result = generate_tts_from_reference(str(paths['reference_results']), str(paths['tts_output_dir']))
                self.logger.info(f"TTS生成结果: {tts_result}")

                if not tts_result.get('success', False):
                    self.logger.error(f"TTS生成失败: {tts_result.get('error', '未知错误')}")
                    return {
                        'success': False,
                        'message': 'TTS生成失败',
                        'error': tts_result.get('error', '未知错误')
                    }

                self._mark_step_completed(cache_data, 'generate_tts', {'result': tts_result})
                self._save_pipeline_cache(paths['pipeline_cache'], cache_data)
            else:
                self.logger.info("步骤4: 生成TTS音频 (已完成，跳过)")

            # 5. 音频对齐
            if not self._check_step_completed(cache_data, 'align_audio'):
                self.logger.info("步骤5: 音频对齐")
                align_result = align_audio_with_subtitles(
                    tts_results_path=str(paths['tts_results']),
                    srt_path=str(paths['processed_subtitle']),
                    output_path=str(paths['aligned_audio'])
                )

                # 保存对齐结果到JSON文件
                align_result_copy = align_result.copy()
                align_result_copy['saved_at'] = datetime.now().isoformat()
                with open(paths['aligned_results'], 'w', encoding='utf-8') as f:
                    json.dump(align_result_copy, f, ensure_ascii=False, indent=2)

                self._mark_step_completed(cache_data, 'align_audio', {'result': align_result})
                self._save_pipeline_cache(paths['pipeline_cache'], cache_data)
            else:
                self.logger.info("步骤5: 音频对齐 (已完成，跳过)")

            # 6. 生成对齐字幕
            if not self._check_step_completed(cache_data, 'generate_aligned_srt'):
                self.logger.info("步骤6: 生成对齐字幕")
                
                # 生成对齐后的SRT字幕
                generate_aligned_srt(
                    str(paths['aligned_results']),
                    str(paths['processed_subtitle']),
                    str(paths['aligned_srt'])
                )
                
                # 检查原始字幕格式，如果不是SRT则需要转换为相应格式
                original_subtitle_path = str(paths['subtitle_path'])
                original_subtitle_ext = Path(original_subtitle_path).suffix.lower()
                
                if original_subtitle_ext != '.srt':
                    self.logger.info(f"检测到原始字幕格式为 {original_subtitle_ext}，正在转换...")
                    
                    if original_subtitle_ext == '.ass':
                        # 对于ASS格式，使用sync_srt_timestamps_to_ass方法同步时间戳
                        aligned_ass_path = paths['output_dir'] / f"{Path(video_path).stem}_aligned.ass"
                        sync_success = sync_srt_timestamps_to_ass(
                            original_subtitle_path,
                            str(paths['aligned_srt']),
                            str(aligned_ass_path)
                        )
                        if sync_success:
                            self.logger.info(f"ASS字幕时间戳同步完成: {aligned_ass_path}")
                        else:
                            self.logger.error("ASS字幕时间戳同步失败")
                    else:
                        # 对于其他格式，使用convert_subtitle转换
                        aligned_subtitle_path = paths['output_dir'] / f"{Path(video_path).stem}_aligned{original_subtitle_ext}"
                        convert_success = convert_subtitle(
                            str(paths['aligned_srt']),
                            str(aligned_subtitle_path)
                        )
                        if convert_success:
                            self.logger.info(f"字幕格式转换完成: {aligned_subtitle_path}")
                        else:
                            self.logger.error("字幕格式转换失败")
                
                self._mark_step_completed(cache_data, 'generate_aligned_srt')
                self._save_pipeline_cache(paths['pipeline_cache'], cache_data)
            else:
                self.logger.info("步骤6: 生成对齐字幕 (已完成，跳过)")

            # 7. 处理视频速度调整
            if not self._check_step_completed(cache_data, 'process_video_speed'):
                self.logger.info("步骤7: 处理视频速度调整")
                process_video_speed_adjustment(
                    str(paths['silent_video']),
                    str(paths['processed_subtitle']),
                    str(paths['aligned_srt'])
                )
                self._mark_step_completed(cache_data, 'process_video_speed')
                self._save_pipeline_cache(paths['pipeline_cache'], cache_data)
            else:
                self.logger.info("步骤7: 处理视频速度调整 (已完成，跳过)")

            # 8. 合并音视频
            if not self._check_step_completed(cache_data, 'merge_audio_video'):
                self.logger.info("步骤8: 合并音视频")
                merge_audio_video(
                    str(paths['speed_adjusted_video']),
                    str(paths['aligned_audio']),
                    str(paths['final_video'])
                )
                self._mark_step_completed(cache_data, 'merge_audio_video')
                self._save_pipeline_cache(paths['pipeline_cache'], cache_data)
            else:
                self.logger.info("步骤8: 合并音视频 (已完成，跳过)")

            self.logger.info(f"处理完成！输出文件: {paths['final_video']}")

            # 计算完成的步骤数
            completed_steps = sum(1 for step in cache_data['completed_steps'].values() if step.get('completed', False))

            return {
                'success': True,
                'message': '视频配音处理完成',
                'output_file': str(paths['final_video']),
                'output_dir': str(paths['output_dir']),
                'steps_completed': completed_steps,
                'cache_file': str(paths['pipeline_cache']),
                'resumed_from_cache': resume_from_cache and existing_cache is not None
            }

        except Exception as e:
            self.logger.error(f"处理过程中发生错误: {str(e)}")
            return {
                'success': False,
                'message': f'处理失败: {str(e)}',
                'error': str(e)
            }

    def process_batch(self, video_subtitle_pairs: list) -> Dict[str, Any]:
        """
        批量处理多个视频

        Args:
            video_subtitle_pairs: 包含(video_path, subtitle_path)元组的列表

        Returns:
            批量处理结果
        """
        results = []
        success_count = 0
        failed_count = 0

        for i, (video_path, subtitle_path) in enumerate(video_subtitle_pairs):
            self.logger.info(f"处理第 {i + 1}/{len(video_subtitle_pairs)} 个视频")

            result = self.process_video(video_path, subtitle_path)
            results.append(result)

            if result['success']:
                success_count += 1
            else:
                failed_count += 1

        return {
            'success': failed_count == 0,
            'message': f'批量处理完成: {success_count} 成功, {failed_count} 失败',
            'total_count': len(video_subtitle_pairs),
            'success_count': success_count,
            'failed_count': failed_count,
            'results': results
        }

    def check_and_repair_cache(self, video_path: str) -> Dict[str, Any]:
        """
        检查并修复缓存文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            检查结果
        """
        try:
            paths = self._get_file_paths(video_path, "")
            cache_file = paths['pipeline_cache']
            
            if not cache_file.exists():
                return {
                    'success': True,
                    'message': '缓存文件不存在',
                    'cache_file': str(cache_file),
                    'cache_exists': False,
                    'repaired': False
                }
            
            # 尝试加载缓存
            cache_data = self._load_pipeline_cache(cache_file)
            
            if cache_data is None:
                # 缓存文件已损坏，已被删除
                return {
                    'success': True,
                    'message': '检测到损坏的缓存文件，已删除',
                    'cache_file': str(cache_file),
                    'cache_exists': False,
                    'repaired': True,
                    'action': 'deleted_corrupted_cache'
                }
            
            # 检查缓存文件大小
            file_size = cache_file.stat().st_size
            if file_size > 1024 * 1024:  # 大于1MB
                self.logger.info(f"检测到过大的缓存文件: {file_size} bytes")
                
                # 优化缓存文件
                cleanup_result = self.cleanup_large_cache_file(video_path)
                return {
                    'success': True,
                    'message': '缓存文件已优化',
                    'cache_file': str(cache_file),
                    'cache_exists': True,
                    'repaired': True,
                    'action': 'optimized_cache',
                    'cleanup_result': cleanup_result
                }
            
            return {
                'success': True,
                'message': '缓存文件正常',
                'cache_file': str(cache_file),
                'cache_exists': True,
                'repaired': False,
                'file_size': file_size
            }
            
        except Exception as e:
            self.logger.error(f"检查缓存文件失败: {str(e)}")
            return {
                'success': False,
                'message': f'检查缓存文件失败: {str(e)}',
                'error': str(e)
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
            paths = self._get_file_paths(video_path, "")
            cache_file = paths['pipeline_cache']
            
            if cache_file.exists():
                cache_file.unlink()
                self.logger.info(f"缓存文件已删除: {cache_file}")
                return {
                    'success': True,
                    'message': '缓存文件已删除',
                    'cache_file': str(cache_file)
                }
            else:
                self.logger.info(f"缓存文件不存在: {cache_file}")
                return {
                    'success': True,
                    'message': '缓存文件不存在',
                    'cache_file': str(cache_file)
                }
                
        except Exception as e:
            self.logger.error(f"清理缓存失败: {str(e)}")
            return {
                'success': False,
                'message': f'清理缓存失败: {str(e)}',
                'error': str(e)
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
            paths = self._get_file_paths(video_path, "")
            cache_file = paths['pipeline_cache']
            
            if not cache_file.exists():
                return {
                    'success': True,
                    'message': '缓存文件不存在',
                    'cache_file': str(cache_file),
                    'cache_exists': False
                }
            
            cache_data = self._load_pipeline_cache(cache_file)
            if not cache_data:
                return {
                    'success': False,
                    'message': '无法读取缓存文件',
                    'cache_file': str(cache_file),
                    'cache_exists': True
                }
            
            completed_steps = cache_data.get('completed_steps', {})
            completed_count = sum(1 for step in completed_steps.values() if step.get('completed', False))
            
            return {
                'success': True,
                'message': '缓存信息获取成功',
                'cache_file': str(cache_file),
                'cache_exists': True,
                'created_at': cache_data.get('created_at'),
                'updated_at': cache_data.get('updated_at'),
                'total_steps': 8,
                'completed_steps': completed_count,
                'completed_step_names': [name for name, step in completed_steps.items() if step.get('completed', False)],
                'remaining_steps': 8 - completed_count,
                'remaining_step_names': [name for name, step in completed_steps.items() if not step.get('completed', False)]
            }
            
        except Exception as e:
            self.logger.error(f"获取缓存信息失败: {str(e)}")
            return {
                'success': False,
                'message': f'获取缓存信息失败: {str(e)}',
                'error': str(e)
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
                output_dir / "adjusted_video"
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
                'success': True,
                'message': f'清理完成，共清理 {len(cleaned_files)} 个文件',
                'cleaned_files': cleaned_files
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'清理失败: {str(e)}',
                'error': str(e)
            }