from core.audio_align_processor import *
from core.media_processor import MediaProcessor, merge_audio_video
from core.subtitle_preprocessor import SubtitlePreprocessor
from core.tts_processor import TTSProcessor
import logging
import json
import os

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def print_detailed_segments(result, title=""):
    """打印详细的片段信息"""
    if not result.get('success', False):
        print(f"❌ {title} 处理失败: {result.get('error', 'Unknown error')}")
        return

    print(f"\n🎯 {title}")
    print(f"✅ 输出文件: {result['output_path']}")
    print(f"⏱️  总时长: {result['total_duration']:.2f}秒")
    print(
        f"📊 统计: {result['audio_segments']} 总片段, {result['tts_segments']} TTS片段, {result['silence_segments']} 静音片段")

    # 显示详细片段信息
    aligned_segments = result.get('aligned_audio_segments', [])
    if aligned_segments:
        print(f"\n📋 详细片段信息 ({len(aligned_segments)} 个):")
        print("=" * 100)

        for i, segment in enumerate(aligned_segments[:8]):  # 显示前8个片段
            segment_type = "🔇 静音" if segment['is_silence'] else "🗣️  TTS"
            print(
                f"{segment['index']:2d}. {segment_type} | {segment['start_time']:6.3f}s - {segment['end_time']:6.3f}s | "
                f"时长:{segment['duration']:5.3f}s | 文件:{os.path.basename(segment['file_path']):30}")
            if segment['text']:
                print(f"     文本: {segment['text'][:40]}{'...' if len(segment['text']) > 40 else ''}")
            if segment.get('silence_reason'):
                print(f"     原因: {segment['silence_reason']}")
            print("-" * 100)

        if len(aligned_segments) > 8:
            print(f"... 还有 {len(aligned_segments) - 8} 个片段")

        # 统计信息
        tts_files = [s for s in aligned_segments if not s['is_silence']]
        silence_files = [s for s in aligned_segments if s['is_silence']]
        total_size = sum(s['file_size'] for s in aligned_segments)

        print(f"\n📈 详细统计:")
        print(f"  TTS音频文件: {len(tts_files)} 个")
        print(f"  静音音频文件: {len(silence_files)} 个")
        print(f"  总文件大小: {total_size:,} 字节 ({total_size / 1024 / 1024:.2f} MB)")

    # JSON文件信息
    json_file = result.get('results_json_file')
    if json_file:
        print(f"\n💾 JSON结果文件: {json_file}")
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                segments_in_json = len(json_data.get('aligned_audio_segments', []))
                print(f"  JSON包含片段数: {segments_in_json}")


if __name__ == '__main__':
    print("=== 音频对齐处理测试 - 详细片段信息版本 ===")

    media_processor = MediaProcessor()
    # media_processor.separate_media(r'C:\Users\leica\Desktop\1\1. Welcome and Course Overview.mp4')

    subtitle_preprocessor = SubtitlePreprocessor()
    subtitle_preprocessor.preprocess_subtitle(r'C:\Users\leica\Desktop\1\1. Welcome and Course Overview.vtt')

    media_processor.generate_reference_audio(
        r'C:\Users\leica\Desktop\1\output\1. Welcome and Course Overview_vocal.wav',
        r'C:\Users\leica\Desktop\1\output\1. Welcome and Course Overview_processed.srt')

    tts_processor = TTSProcessor()
    tts_processor.generate_tts_from_reference(
        r'C:\Users\leica\Desktop\1\output\reference_audio\1. Welcome and Course Overview_vocal_reference_audio_results.json')

    print("\n🔄 处理并生成详细片段信息:")
    result = align_audio_with_subtitles(
        tts_results_path=r'C:\Users\leica\Desktop\1\output\tts_output\tts_generation_results.json',
        srt_path=r'C:\Users\leica\Desktop\1\output\1. Welcome and Course Overview_processed.srt'
    )
    print_detailed_segments(result)

    generate_aligned_srt(r'C:\Users\leica\Desktop\1\output\aligned_audio\aligned_tts_generation_results_results.json',
                         r'C:\Users\leica\Desktop\1\output\1. Welcome and Course Overview_processed.srt')

    # process_video_speed_adjustment(r'C:\Users\leica\Desktop\1\output\1. Welcome and Course Overview_silent.mp4',
    #                                r'C:\Users\leica\Desktop\1\output\1. Welcome and Course Overview_processed.srt',
    #                                r'C:\Users\leica\Desktop\1\output\aligned_subtitles\aligned_tts_generation_aligned.srt')

    merge_audio_video(
        r'C:\Users\leica\Desktop\1\output\adjusted_video\final_speed_adjusted_1. Welcome and Course Overview_silent.mp4',
        r'C:\Users\leica\Desktop\1\output\aligned_audio\aligned_tts_generation_results.wav')
    print("\n✅ 测试完成!")
