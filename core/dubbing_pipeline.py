import logging
import os
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
            'aligned_results': output_dir / "aligned_audio" / "aligned_tts_generation_results_results.json",
            'aligned_audio': output_dir / "aligned_audio" / "aligned_tts_generation_results.wav",
            'aligned_srt': output_dir / "aligned_audio" / "aligned_tts_generation_aligned.srt",
            'final_video': output_dir / f"{video_name}_dubbed.mp4",
            'speed_adjusted_video': output_dir / "adjusted_video" / f"final_speed_adjusted_{video_name}_silent.mp4"
        }
        
        # 创建必要的子目录
        for dir_path in ['reference_audio_dir', 'tts_output_dir', 'aligned_audio_dir', 'adjusted_video_dir']:
            paths[dir_path].mkdir(exist_ok=True)
            
        return paths
    
    def process_video(self, video_path: str, subtitle_path: str) -> Dict[str, Any]:
        """
        处理视频配音的完整流程
        
        Args:
            video_path: 视频文件路径
            subtitle_path: 字幕文件路径
            
        Returns:
            处理结果字典
        """
        try:
            self.logger.info(f"开始处理视频: {video_path}")
            self.logger.info(f"字幕文件: {subtitle_path}")
            
            # 获取文件路径
            paths = self._get_file_paths(video_path, subtitle_path)
            
            # 1. 预处理字幕
            self.logger.info("步骤1: 预处理字幕")
            preprocess_subtitle(str(paths['subtitle_path']))
            
            # 2. 分离媒体文件
            self.logger.info("步骤2: 分离音视频")
            separate_media(str(paths['video_path']))
            
            # 3. 生成参考音频
            self.logger.info("步骤3: 生成参考音频")
            generate_reference_audio(
                str(paths['vocal_audio']),
                str(paths['processed_subtitle'])
            )
            
            # 4. 生成TTS音频
            self.logger.info("步骤4: 生成TTS音频")
            generate_tts_from_reference(str(paths['reference_results']))
            
            # 5. 音频对齐
            self.logger.info("步骤5: 音频对齐")
            align_result = align_audio_with_subtitles(
                tts_results_path=str(paths['tts_results']),
                srt_path=str(paths['processed_subtitle'])
            )
            
            # 6. 生成对齐字幕
            self.logger.info("步骤6: 生成对齐字幕")
            generate_aligned_srt(
                str(paths['aligned_results']),
                str(paths['processed_subtitle'])
            )
            
            # 7. 处理视频速度调整
            self.logger.info("步骤7: 处理视频速度调整")
            process_video_speed_adjustment(
                str(paths['silent_video']),
                str(paths['processed_subtitle']),
                str(paths['aligned_srt'])
            )
            
            # 8. 合并音视频
            self.logger.info("步骤8: 合并音视频")
            merge_audio_video(
                str(paths['speed_adjusted_video']),
                str(paths['aligned_audio'])
            )
            
            self.logger.info(f"处理完成！输出文件: {paths['final_video']}")
            
            return {
                'success': True,
                'message': '视频配音处理完成',
                'output_file': str(paths['final_video']),
                'output_dir': str(paths['output_dir']),
                'steps_completed': 8
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
            self.logger.info(f"处理第 {i+1}/{len(video_subtitle_pairs)} 个视频")
            
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