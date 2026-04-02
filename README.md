# MultiModalPestDetectionWithQwen

基于 `RGB + 高光谱(HSI) + 文本描述 + Qwen2.5-7B` 的多模态农作物病虫害识别项目。该仓库实现了从数据准备、训练、评估到推理的完整流程，目标是在植物叶片场景下融合可见光图像、光谱信息与文本先验，提升病害和虫害分类的鲁棒性。

当前代码中的核心实现位于 [`models/main_model_qwen.py`]，训练入口位于 [`scripts/train_qwen_fixed.py`]

## 1. 模型能做什么

这个模型的核心任务是多类别病虫害分类。按代码当前实现，它具备以下能力：

- 接收一张 RGB 叶片图像，提取空间纹理、颜色和形态特征。
- 可选接收一份高光谱图像 `HSI`，补充 RGB 无法直接表达的细粒度光谱差异。
- 接收一段文本描述，提供症状语义先验，例如“叶片黄化”“出现褐色病斑”等。
- 在编码后执行跨模态对齐，让图像与文本、RGB 与 HSI 在统一特征空间中互相校正。
- 将融合后的多模态表示适配到 Qwen2.5-7B 的隐藏空间，用大模型的表达能力进一步建模。
- 输出每个类别的分类 logits、预测概率、Top-K 结果，以及中间特征用于分析或可视化。

从仓库内置的 [`data/class_mapping.json`] 看，当前示例类别映射包含 15 类，覆盖辣椒、马铃薯、番茄等常见病害/健康状态。

## 2. 模型设计目标

这份代码并不是简单把三种输入直接拼起来做分类，而是围绕下面几个目标设计的：

- 让模型不只“背文本描述”，而是真的学习图像特征。
- 让 RGB 与 HSI 在同一决策中互补，而不是各自独立预测。
- 通过参数高效微调，把 Qwen 当作高层语义建模器使用，降低全量微调成本。
- 在缺失 HSI 时仍可运行，便于在真实部署场景中做降级推理。

其中一个很关键的实现是训练集文本遮蔽：[`data/dataset.py`] 在训练阶段按 `20%` 概率把原始描述替换为通用句子 `"一张需要分析的植物叶片图片。"`，目的是减少模型对文本标签泄露的依赖，强制其从视觉模态中学习。

## 3. 总体架构

整体流程可以概括为：

```text
RGB 图像 ──> RGBEncoder(ViT)
                          \
HSI 图像 ──> HSIEncoder ----> Cross-Modal Alignment
                          /                |
文本描述 ──> TextEncoder(BERT)          
                                  Multi-Modal Fusion
                                           |
                                  Embedding Adapter
                                           |
                                   Qwen2.5-7B + LoRA
                                           |
                                      Classification Head
                                           |
                                     病虫害类别预测
```

## 4. 各模块功能说明

### 4.1 RGB 编码器

文件：[`models/encoders/rgb_encoder.py`]

作用：

- 基于 Vision Transformer 思路，把 `224x224` RGB 图像切成 patch。
- 通过多层自注意力建模叶片纹理、病斑分布、边缘形态和整体结构。
- 输出：
  - `cls_token`: 全局图像表征。
  - `patch_tokens`: patch 级细粒度视觉特征。

默认配置：

- 输入尺寸：`224 x 224`
- patch 大小：`16`
- 深度：`12`
- 嵌入维度：`768`

### 4.2 HSI 编码器

文件：[`models/encoders/hsi_encoder.py`]

作用：

- 针对高光谱输入做光谱降维和空间建模。
- 先用 `EfficientSpectralReduction` 压缩通道，再用空间下采样与分组空间注意力提取联合特征。
- 输出：
  - `cls_token`: 全局 HSI 表征。
  - `patch_tokens`: 光谱-空间 patch 特征。

默认输入规格：

- 通道数：`224`
- 空间尺寸：`64 x 64`

代码特点：

- 自动调整注意力头数，避免通道数与 head 数不整除导致报错。
- 提供 `HSIEncoderCheckpoint`，可以在显存紧张时启用梯度检查点。

### 4.3 文本编码器

文件：[`models/encoders/text_encoder.py`]
作用：

- 使用 `bert-base-chinese` 编码病虫害症状描述。
- 把文本 token 序列映射到与视觉模态相同的 `embed_dim=768`。
- 提供：
  - `sequence_features`: token 级特征，用于跨模态对齐。
  - `pooled_features`: 句级特征。
  - `enhanced_features`: 额外线性层增强后的句向量。

### 4.4 跨模态对齐模块

文件：[`models/fusion/cross_modal_alignment.py`]

作用：

- 建立 `RGB ↔ Text`、`HSI ↔ Text`、`RGB ↔ HSI` 之间的注意力交互。
- 让文本去关注视觉关键区域，也让视觉特征吸收症状语义信息。
- 对池化后的模态特征计算对比学习损失，拉近同一样本的不同模态表示。

输出包括：

- `rgb_aligned`
- `hsi_aligned`
- `text_aligned`
- `alignment_loss`

### 4.5 多模态融合模块

文件：[`models/fusion/multi_modal_fusion.py`]

作用：

- 在对齐后进一步深度融合三种模态。
- 先为每个模态加入可学习 token，再拼接后送入多层 Transformer。
- 支持三种融合策略：
  - `concat`
  - `gated`
  - `hierarchical`

当前训练脚本默认使用 `hierarchical`，即先做两两融合，再做最终融合，适合表达 RGB、HSI、文本之间的不同互补关系。

### 4.6 Embedding 适配器

文件：[`models/adapters/embedding_adapter.py`]

作用：

- 将融合后的多模态特征转换为 Qwen 能直接处理的 token embeddings。
- 使用 Q-Former 风格的可学习 query tokens，通过 cross-attention 从多模态特征中提炼信息。
- 最终把特征从 `768` 维投影到 Qwen 的隐藏维度。

默认设置：

- query token 数：`32`
- cross-attention 层数：`4`

### 4.7 Qwen2.5-7B 与 LoRA

文件：

- [`models/main_model_qwen.py`]
- [`models/adapters/lora.py`]

作用：

- 使用本地 Qwen2.5-7B 作为高层语义建模器。
- 默认冻结大部分 Qwen 参数，只保留指定层和 LoRA 参数参与训练。
- LoRA 默认注入到注意力投影层：
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`

这样做的收益：

- 明显减少可训练参数量。
- 降低显存压力。
- 保留大模型表征能力，同时适配领域任务。

### 4.8 分类头

位置：[`models/main_model_qwen.py`]

作用：

- 对 Qwen 最后一层隐藏状态做均值池化。
- 通过两层 MLP 输出病虫害分类 logits。

最终损失：

```text
total_loss = cls_loss + 0.1 * alignment_loss
```

其中 `cls_loss` 为交叉熵损失。

## 5. 输入与输出定义

### 5.1 输入

训练和推理时模型接收以下张量：

- `rgb_images`: `[B, 3, 224, 224]`
- `hsi_images`: `[B, 224, 64, 64]`
- `text_input_ids`: `[B, L]`
- `text_attention_mask`: `[B, L]`
- `labels`: `[B]`，仅训练/验证需要

如果关闭 `--use_hsi`，数据集会用零张量占位，保持接口一致。

### 5.2 输出

模型前向返回一个字典，至少包含：

- `logits`: 分类分数
- `pooled_features`: Qwen 池化特征
- `alignment_loss`: 跨模态对齐损失
- `cls_loss`: 分类损失
- `total_loss`: 总损失

开启 `return_features=True` 时还会返回：

- `rgb_features`
- `hsi_features`
- `text_features`
- `fused_features`
- `llm_embeddings`

这些结果可用于做特征可视化、错误分析或模态消融实验。

## 6. 数据组织方式

### 6.1 预处理后的目录结构

项目约定的数据目录结构如下：

```text
processed_data/
├── class_mapping.json
├── train.json
├── val.json
├── test.json
├── rgb/
│   └── <class_name>/
│       └── <sample_id>.jpg
└── hsi/
    └── <class_name>/
        └── <sample_id>.npy
```

### 6.2 单条样本索引格式

`train.json / val.json / test.json` 中每条样本大致如下：

```json
{
  "id": "Tomato_healthy_xxx",
  "rgb_path": "rgb/Tomato_healthy/Tomato_healthy_xxx.jpg",
  "hsi_path": "hsi/Tomato_healthy/Tomato_healthy_xxx.npy",
  "class": "Tomato_healthy",
  "description": "叶片呈鲜亮的绿色，形态为典型的羽状复叶。"
}
```

### 6.3 数据准备脚本做了什么

文件：[`data/prepare_data.py`]

该脚本会：

- 收集 RGB 目录下的所有类别和图像。
- 按 `train/val/test` 比例分层划分数据。
- 复制 RGB 文件到统一输出目录。
- 复制已有 HSI，或在缺失时生成 dummy HSI / 零张量 HSI。
- 自动生成文本描述，且描述带有随机性，避免每类始终使用固定模板。
- 生成 `class_mapping.json` 和各 split 的索引 JSON。

一个典型命令如下：

```bash
python data/prepare_data.py \
  --rgb_folder ./raw/rgb \
  --hsi_folder ./raw/hsi \
  --output_folder ./data/processed_data
```

若没有真实 HSI，可用：

```bash
python data/prepare_data.py \
  --rgb_folder ./raw/rgb \
  --output_folder ./data/processed_data \
  --create_dummy_hsi
```

## 7. 训练流程

训练入口：[`scripts/train_qwen_fixed.py`]

该脚本的训练策略具有几个明确特点：

- 强制要求 GPU 环境，且默认优先考虑显存优化。
- 自动启用 AMP 混合精度以节省显存。
- 使用梯度累积扩大有效 batch size。
- 对不同模块设置不同学习率：
  - 编码器与融合层较高
  - Qwen / LoRA 部分较低
- 使用 `CosineAnnealingWarmRestarts` 学习率调度。
- 每轮验证并保存 `latest.pth` 与 `best_model.pth`。
- 支持早停。

示例命令：

```bash
python scripts/train_qwen_fixed.py \
  --data_root ./data/processed_data \
  --qwen_path ./models/qwen2.5-7b \
  --batch_size 2 \
  --accumulation_steps 4 \
  --epochs 50 \
  --lr 2e-5 \
  --output_dir ./outputs/qwen_fixed \
  --use_hsi
```

如果想禁用 HSI：

```bash
python scripts/train_qwen_fixed.py \
  --data_root ./data/processed_data \
  --qwen_path ./models/qwen2.5-7b \
  --output_dir ./outputs/qwen_rgb_text \
  --no-use_hsi
```

训练输出通常包括：

- `logs/`：TensorBoard 日志
- `checkpoints/latest.pth`
- `checkpoints/best_model.pth`
- `history.json`
- `curves.png`

## 8. 评估流程

评估入口：[`scripts/evaluate_qwen.py`]

评估脚本会：

- 加载测试集。
- 恢复训练好的模型权重。
- 计算准确率、精确率、召回率、F1。
- 保存混淆矩阵。
- 在开启特征返回时做 t-SNE 可视化。

示例命令：

```bash
python scripts/evaluate_qwen.py \
  --checkpoint ./outputs/qwen_fixed/checkpoints/best_model.pth \
  --data_root ./data/processed_data \
  --qwen_path ./models/qwen2.5-7b \
  --output_dir ./eval_results \
  --use_hsi
```

评估输出通常包括：

- `results.json`
- `predictions.npy`
- `labels.npy`
- `confusion_matrix.png`
- `tsne.png`

## 9. 推理流程

推理入口：[`scripts/inference_qwen.py`]

支持两种模式：

- 单张图片推理
- 文件夹批量推理

单张推理示例：

```bash
python scripts/inference_qwen.py \
  --checkpoint ./outputs/qwen_fixed/checkpoints/best_model.pth \
  --data_root ./data/processed_data \
  --qwen_path ./models/qwen2.5-7b \
  --rgb_image ./demo/test.jpg \
  --hsi_image ./demo/test.npy \
  --text "叶片出现黄色斑点和边缘卷曲" \
  --top_k 5 \
  --output ./pred_result.json
```

批量推理示例：

```bash
python scripts/inference_qwen.py \
  --checkpoint ./outputs/qwen_fixed/checkpoints/best_model.pth \
  --data_root ./data/processed_data \
  --qwen_path ./models/qwen2.5-7b \
  --image_dir ./inference_test_set \
  --pattern "*.jpg" \
  --output ./batch_predictions.json
```

推理输出 JSON 中会包含：

- 输入路径
- Top-K 类别
- 每个类别的置信度
- `top_prediction`

## 10. 配置系统

配置文件：[`config/model_config.py`]

该模块定义了几组 dataclass 配置：

- `EncoderConfig`
- `FusionConfig`
- `LLMConfig`
- `TrainingConfig`
- `DataConfig`
- `ModelConfig`

它支持：

- 从 Python 默认配置构建
- 从 JSON / YAML 加载
- 导出到 JSON / YAML

并内置了几套便捷配置：

- `get_baseline_config()`
- `get_small_config()`
- `get_large_config()`
- `get_llm_config()`

## 11. 指标与可视化

文件：

- [`utils/metrics.py`]
- [`utils/visualization.py`]

当前实现支持：

- Accuracy
- Precision / Recall / F1
- Per-class 指标
- Confusion Matrix
- ROC AUC
- 训练曲线
- 特征分布可视化
- 注意力图可视化

这意味着该项目不仅能训练模型，也方便做误差分析和模态解释。

## 12. 安装依赖

依赖文件：[`requirements.txt`]

建议使用 Python 3.10+，然后执行：

```bash
pip install -r requirements.txt
```

核心依赖包括：

- `torch`
- `transformers`
- `albumentations`
- `scikit-learn`
- `timm`
- `tensorboard`

如果要进一步做更标准的 LLM PEFT 训练，可按需补装：

- `accelerate`
- `bitsandbytes`
- `peft`


