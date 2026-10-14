<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Spleeter → NCNN FP16

已将 Deezer Spleeter 的 **2、4、5 声部**预训练网络转换为 NCNN。产物位于 `models/{2,4,5}stems-fp16/`，每个声部各有 `.param` 与 `.bin` 文件。模型权重以 FP16 存储，输入输出仍为 FP32。

| 模型 | 声部 | 来源 |
| --- | --- | --- |
| 2stems | vocals、accompaniment | [sherpa-onnx 从 Deezer checkpoint 导出的 FP32 ONNX](https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/spleeter) |
| 4stems | vocals、drums、bass、other | [Best-Practice 从 Deezer checkpoint 导出的 FP16 ONNX](https://huggingface.co/Best-Practice/spleeter-4stems-onnx) |
| 5stems | vocals、piano、drums、bass、other | [Deezer 官方 TensorFlow checkpoint](https://github.com/deezer/spleeter/releases/tag/v1.4.0)；使用 4 声部 ONNX 作为网络结构模板，替换全部权重并导出 logits |

源模型是 [Deezer Spleeter](https://github.com/deezer/spleeter) 的 v1.4.0 预训练权重。NCNN 转换使用 [PNNX](https://github.com/Tencent/ncnn/blob/master/tools/onnx/README.md)。4 声部 ONNX 的算子和权重均为 FP16；脚本先把它们提升到 FP32，使 PNNX 正确识别卷积，再用 `fp16=1` 写入 NCNN。这个步骤不能恢复原 ONNX 已经损失的精度。

## 重新转换

需要 Python 3、PNNX。4、5 声部还需要 `onnx` 与 `numpy` Python 包。5 声部 checkpoint 可用 TensorFlow 读取；也可用 [tensorflow-checkpoint-reader](https://github.com/shawwn/tensorflow-checkpoint-reader) 读取。后者还需要 `six` 与 `protobuf`。

```sh
python3 convert.py --model 2stems --pnnx /path/to/pnnx
python3 convert.py --model 4stems --pnnx /path/to/pnnx
python3 convert_5stems.py --pnnx /path/to/pnnx
```

脚本会下载所需 ONNX 源文件。已有 2 声部压缩包时可传 `--archive /path/to/sherpa-onnx-spleeter-2stems.tar.bz2`；已有 ONNX 文件时可传 `--onnx-dir /path/to/files`。4 声部目录中可放 `vocals.fp16.onnx` 等文件，也可放同名 FP32 `.onnx` 文件。

不安装 TensorFlow 时，可克隆纯 Python checkpoint 读取器，并运行：

```sh
git clone https://github.com/shawwn/tensorflow-checkpoint-reader.git /tmp/tf-checkpoint-reader
git -C /tmp/tf-checkpoint-reader checkout c1fe2d751a76e202478a92108d126a7433d352c4
python3 convert_5stems.py --pure-reader /tmp/tf-checkpoint-reader --pnnx /path/to/pnnx
```

5 声部脚本会下载官方 `5stems.tar.gz`，并使用 4 声部的 ELU U-Net 导出图作为结构模板；每层卷积、转置卷积和 BatchNorm 权重都从官方 checkpoint 重新填入。输出是五个 **softmax 前 logits** 图，不是最终幅度估计图。

## 推理接口

每个 NCNN 图的输入 blob 是 `in0`，输出 blob 是 `out0`。模型接收一个 `[2, 1, 512, 1024]` 的 FP32 幅度谱张量，维度依次为立体声通道、一个分段、512 帧、1024 个频率 bin；输出形状相同。PNNX 固定了每次一个分段的形状。源 ONNX 的动态 `num_splits` 不能直接沿用到这些 NCNN 产物。

2、4 声部的每个网络输出对应声部的幅度估计。5 声部的每个网络输出 logits，必须收集五个声部的输出，逐元素按声部维度做 softmax，再分别乘原始幅度谱，才能得到幅度估计。例如以 `vocals,piano,drums,bass,other` 为声部顺序：

```python
z = np.stack([logits[s] for s in ("vocals", "piano", "drums", "bass", "other")], axis=0)
e = np.exp(z - z.max(axis=0, keepdims=True))
probabilities = e / e.sum(axis=0, keepdims=True)
estimates = probabilities * mix_magnitude
```

随后按 [Spleeter 的计算](https://github.com/deezer/spleeter/blob/master/spleeter/model/__init__.py)做跨声部 soft ratio mask：对每个声部的幅度估计平方，加上 `1e-10 / 声部数`，除以全部声部估计平方之和再加 `1e-10`。完整音频分离还需要在宿主程序中完成 STFT、分段、相位复用、频带补齐和逆 STFT。2 声部与 4、5 声部的网络使用不同的激活函数，产物不能混用。[Spleeter 配置](https://github.com/deezer/spleeter/tree/master/spleeter/resources)规定采样率 44.1 kHz、周期 Hann 窗 4096 点、hop 1024 点、每段 512 帧以及前 1024 个频率 bin。4 声部的[导出说明](https://huggingface.co/Best-Practice/spleeter-4stems-onnx)包含音频重建代码。

FP16 权重存储不保证目标设备执行 FP16 算术，后者取决于 NCNN 构建和硬件。当前只完成模型转换与结构检查，尚未做 NCNN 与 TensorFlow 数值对照或音频分离质量测量。

## 许可证与来源

本仓库自编写的转换脚本及本文档以 [GNU GPL v3.0（仅此版本）](LICENSE) 发布。复制、修改或分发这些文件时，请遵守 GPLv3 的规定，并保留相应的版权与许可证声明。

`upstream/` 是 Deezer Spleeter 的上游副本，其代码仍按 [Deezer 的 MIT 许可证](upstream/LICENSE) 发布。上游示例音频和其他资源不自动采用 MIT 或 GPLv3。模型文件由第三方模型和 Deezer 预训练权重转换而来；它们的来源、所适用的原始许可及需要保留的署名列于 [第三方许可与署名](THIRD_PARTY_NOTICES.md)。根目录的 GPLv3 不会取代这些原始许可。`models/` 当前被 `.gitignore` 排除；若单独发布转换后的模型，应同时提供该文件、相关许可证文本、署名和转换说明。
