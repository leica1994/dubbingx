"""文件名处理工具函数"""

import re


def sanitize_filename(filename: str) -> str:
    """清理文件名，将不支持的字符转为下划线"""
    # 替换Windows和Linux不支持的特殊字符
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # 替换其他可能引起问题的字符
    sanitized = re.sub(r"[@#&%=+]", "_", sanitized)
    # 统一处理分隔符：将点号、连字符、空格都转为下划线
    sanitized = re.sub(r"[.\-\s]", "_", sanitized)
    # 替换连续的下划线为单个下划线
    sanitized = re.sub(r"_+", "_", sanitized)
    # 去除开头和结尾的下划线
    sanitized = sanitized.strip("_")
    # 确保文件名不为空
    if not sanitized:
        sanitized = "unnamed"
    return sanitized
