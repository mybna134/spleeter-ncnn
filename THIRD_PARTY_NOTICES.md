<!-- SPDX-License-Identifier: GPL-3.0-only -->

# 许可




|  | 许可 | 本仓库中的用途 |
| --- | --- | --- |
| `upstream/` 中的 Spleeter 源码 | Deezer SA，MIT；许可全文见 [`upstream/LICENSE`](upstream/LICENSE) 和 [`LICENSES/Deezer-MIT.txt`](LICENSES/Deezer-MIT.txt) | 保留的上游副本及参考实现 |
| `upstream/audio_example.mp3`、`upstream/audio_example_mono.mp3` | Steven M. Bryant 的 *Slow Motion Dream* 片段，参与者包括 CSoul、Alex Beroza 和 Robert Siekawitch；上游 README 标注为 [Creative Commons Attribution 3.0](LICENSES/CC-BY-3.0.txt)，原曲页面为 [ccMixter](http://dig.ccmixter.org/files/stevieb357/34740) | 上游测试和示例音频 | 
| `models/2stems-fp16/` | 由 [sherpa-onnx 的 Spleeter ONNX 发布包](https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/spleeter)转换。Apache-2.0 适用于其项目中相应的代码和通知；Deezer 预训练权重的许可和署名仍按 Deezer 的原始说明处理 | `convert.py` 生成的 NCNN 模型 |
| `models/4stems-fp16/` | 由 [Best-Practice 的 4 声部 ONNX 发布](https://huggingface.co/Best-Practice/spleeter-4stems-onnx)转换。该发布附有 Apache-2.0 及 Xiaomi Corporation 相关通知；其中使用的 Deezer 预训练权重仍按其原始说明处理 | `convert.py` 生成的 NCNN 模型 |
| `models/5stems-fp16/` | 使用 [Deezer v1.4.0 官方 5 声部 checkpoint](https://github.com/deezer/spleeter/releases/tag/v1.4.0)填充网络参数，并使用上面的 Best-Practice ONNX 图作为结构模板 | `convert_5stems.py` 生成的 NCNN 模型 |
