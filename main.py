"""
使用 DubbingPipeline 的示例
"""

from core.dubbing_pipeline import DubbingPipeline
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    # 创建配音流水线实例
    pipeline = DubbingPipeline()

    # 处理单个视频
    video_path = r'C:\Users\leica\Desktop\1\13. 日线图、周线图和月线图.mp4'
    subtitle_path = r'C:\Users\leica\Desktop\1\13. 日线图、周线图和月线图.ass'

    print(f"开始处理视频: {video_path}")
    print(f"字幕文件: {subtitle_path}")

    result = pipeline.process_video(video_path, subtitle_path)

    if result['success']:
        print(f"处理成功！")
        print(f"输出文件: {result['output_file']}")
        print(f"输出目录: {result['output_dir']}")
        print(f"完成步骤数: {result['steps_completed']}")
    else:
        print(f"处理失败: {result['message']}")
        if 'error' in result:
            print(f"错误详情: {result['error']}")


if __name__ == '__main__':
    main()