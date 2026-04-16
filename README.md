# 🎬 Qwen Image Edit 3D 资产生产工具 (增强版)

**Windows环境・低显存GPU优化版** - 基于 Qwen-Image-Edit-2511 的本地 3D 相机控制图像生成工具

本仓库是 [qwen-image-multiple-angles-3d-camera_lowspec](https://github.com/tomosud/qwen-image-multiple-angles-3d-camera_lowspec) 的增强版，进行了多处优化，并新增多项实用功能。

<img width="1286" height="739" alt="image" src="https://github.com/user-attachments/assets/dda9a816-535b-4c85-9105-72a1368df98a" />

## ✨ 主要特性

### 核心优化
- **GGUF Q2_K 量子化**: 40GB → 7.47GB（约 80% 削减）
- **CPU Offloading**: 支持 12GB 显存运行
- **Windows 完全兼容**: 一键启动脚本，自动安装依赖
- **Hugging Face 镜像**: 国内 hf-mirror.com 加速下载

### 🆕 新增功能

| 功能 | 说明 |
|------|------|
| **一键启动脚本** | `start.bat` / `start.sh` 交互式启动，可选 CPU 优化模式 |
| **八方向序列帧生成** | 一键生成 360° 全方位 8 张精灵图（0°/45°/90°/135°/180°/225°/270°/315°） |
| **附加编辑提示词** | 支持叠加自定义 Prompt（如换装、换风格） |
| **一键抠图 (Rembg)** | 生成后自动移除背景，输出透明 PNG |
| **图像画廊** | 历史记录画廊，保存所有生成图片，可预览/下载/清空 |
| **界面汉化** | 全中文 UI，降低使用门槛 |

## 💻 系统需求

| 项目 | 要求 |
|------|------|
| **操作系统** | Windows 10/11 或 Linux |
| **GPU** | NVIDIA 12GB 显存以上（RTX 3060 12GB, RTX 4060 Ti 等） |
| **大显存模式** | 24GB+ 显存可关闭 CPU 优化，全速运行 |
| **内存** | 16GB 以上推荐 |
| **磁盘** | 约 15GB（模型 + 依赖） |
| **前置条件** | Python 3.10+ 已安装 |

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/tomosud/qwen-image-multiple-angles-3d-camera_lowspec.git
cd qwen-image-multiple-angles-3d-camera_lowspec
```

### 2. 安装依赖

**Windows:**
```bash
setup.bat
```

**Linux:**
```bash
bash setup.sh
```

### 3. 启动程序

**Windows:**
```bash
start.bat
```

**Linux:**
```bash
bash start.sh
```

启动时会提示选择 CPU 优化模式：
- **[1] 开启（默认）** - 适合 8G~16G 显存，防止爆显存
- **[2] 关闭** - 适合 24G+ 大显存（RTX 3090/4090/A100），全速运行

启动后在浏览器打开显示的地址（默认 `http://127.0.0.1:6006`）

> **注意**: 初次启动需要下载约 13GB 的量化模型文件。

## 🎮 使用指南

### 基础操作

1. **上传图像** - 在左侧上传原始图片
2. **调整视角** - 在 3D 控制器中拖动：
   - 🟢 **绿色球** - 水平旋转（Azimuth 方位角）
   - 💗 **粉色球** - 垂直角度（Elevation 俯仰角）
   - 🟠 **橙色球** - 距离控制
3. **生成** - 点击「单张生成」或「一键生成 360° 序列」

### 高级功能

#### ✍️ 附加编辑提示词
在「附加编辑控制」区输入英文描述，可叠加编辑效果：
```
wearing a red jacket, cyberpunk style
```
> AI 会将视角描述 + 你的提示词组合执行

#### 🖼️ 一键抠图
勾选「生成后一键移除背景」，生成的图片自动去除背景，输出透明 PNG。

#### 🎬 八方向序列帧
点击「一键生成 360° 序列」，自动生成 8 个方位视角的精灵图，适合制作：
- 游戏 2D 角色 Sprite Sheet
- 产品展示多角度图
- 旋转 GIF 动画素材

#### 🖼️ 历史画廊
右侧画廊保存所有生成历史，可：
- 点击查看大图
- 右键下载保存
- 「清空历史记录」一键重置

## ⚙️ 高级参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Seed | 随机 | 固定种子可复现结果 |
| Guidance Scale | 1.0 | 提示词引导强度 |
| Inference Steps | 4 | 推理步数（Lightning LoRA 加速） |
| Height/Width | 1024 | 输出分辨率（自动适配输入图） |

## 📝 许可证

Apache 2.0 - 原始模型: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

## 🙏 致谢

- **原始项目**: [multimodalart/qwen-image-multiple-angles-3d-camera](https://huggingface.co/spaces/multimodalart/qwen-image-multiple-angles-3d-camera)
- **GGUF 量化**: [unsloth](https://huggingface.co/unsloth)
- **模型**: [Qwen Team](https://huggingface.co/Qwen)
- **抠图**: [Rembg](https://github.com/danielgatis/rembg)