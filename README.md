# DubbingX - 智能视频配音系统

基于AI的视频配音系统，支持GPU加速和声音克隆。

## 快速开始

### 环境要求
- Python 3.9-3.12
- NVIDIA GPU (推荐，支持CUDA 12.8)
- FFmpeg

### 安装依赖

使用uv一键安装所有依赖（包括PyTorch CUDA版本）：

```bash
# 安装uv（如果还没有）
pip install uv

# 克隆项目
git clone <your-repo-url>
cd dubbingx

# 一键安装所有依赖（包括PyTorch CUDA 12.8）
uv sync

# 安装开发依赖
uv sync --extra dev
```

### 开发工具

```bash
# 代码格式化
uv run black core/
uv run isort core/

# 代码检查
uv run flake8 core/

# 运行测试
uv run pytest

# 跳过慢速测试
uv run pytest -m "not slow"
```

## 核心功能

- **音频分离**: 使用Demucs AI模型分离人声和背景音乐
- **声音克隆**: 基于Index-TTS的参考音频声音克隆
- **字幕处理**: 智能字幕预处理和格式转换
- **GPU加速**: 自动检测CUDA并启用GPU加速

## 架构概览

```
视频输入 → 音频分离 → 字幕预处理 → TTS生成 → 最终合成
```

详细架构和开发指南请参考 [CLAUDE.md](CLAUDE.md)。