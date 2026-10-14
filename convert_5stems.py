#!/usr/bin/env python3
"""Export Deezer's 5-stem checkpoint as five NCNN FP16 logits networks."""

import argparse
import collections
import importlib.metadata
import inspect
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import urllib.request

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import onnx
from onnx import numpy_helper

from convert import convert_model, promote_fp16_onnx


STEMS = ("vocals", "piano", "drums", "bass", "other")
CONVS = ("conv", "conv1", "conv2", "conv3", "conv4", "conv5", "up7")
UPS = ("up1", "up2", "up3", "up4", "up5", "up6")
BNS = ("bn", "bn1", "bn2", "bn3", "bn4", "bn5", "bn6", "bn7", "bn8", "bn9", "bn10")
BN_OFFSETS = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11)
CHECKPOINT_URL = "https://github.com/deezer/spleeter/releases/download/v1.4.0/5stems.tar.gz"
CHECKPOINT_ARCHIVE_SIZE = 182_803_368
TEMPLATE_URL = "https://huggingface.co/Best-Practice/spleeter-4stems-onnx/resolve/main/vocals.fp16.onnx"


def prepare_sources(checkpoint: Path, template: Path) -> None:
    data_file = checkpoint.parent / f"{checkpoint.name}.data-00000-of-00001"
    if not checkpoint.with_suffix(".index").is_file() or not data_file.is_file():
        archive = checkpoint.parent.parent / "5stems.tar.gz"
        if not archive.is_file() or archive.stat().st_size != CHECKPOINT_ARCHIVE_SIZE:
            archive.parent.mkdir(parents=True, exist_ok=True)
            partial = archive.with_suffix(".partial")
            with urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            if partial.stat().st_size != CHECKPOINT_ARCHIVE_SIZE:
                raise RuntimeError("Incomplete 5-stem checkpoint download")
            partial.replace(archive)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as source:
            members = {Path(item.name).name: item for item in source if item.isfile()}
            for filename in ("checkpoint", "model.index", "model.data-00000-of-00001"):
                member = members.get(filename)
                if member is None:
                    raise RuntimeError(f"{filename} missing from checkpoint archive")
                with source.extractfile(member) as content, (checkpoint.parent / filename).open("wb") as output:
                    shutil.copyfileobj(content, output)
    if not template.is_file() or template.stat().st_size < 10_000_000:
        template.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TEMPLATE_URL, timeout=60) as response, template.open("wb") as output:
            shutil.copyfileobj(response, output)


def scope(family: str, index: int) -> str:
    return family if index == 0 else f"{family}_{index}"


def open_checkpoint(prefix: Path, pure_reader: Path | None):
    if pure_reader is None:
        import tensorflow as tf

        return tf.train.load_checkpoint(str(prefix))

    # Compatibility shims for the standalone reader on Python 3.12 and NumPy 2.
    sys.path.insert(0, str(pure_reader))
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_
    if not hasattr(inspect, "ArgSpec"):
        inspect.ArgSpec = collections.namedtuple("ArgSpec", "args varargs keywords defaults")
    version = importlib.metadata.version
    importlib.metadata.version = lambda name: "0" if name == "tensorflow_checkpoint_reader" else version(name)
    from tensorflow_checkpoint_reader.py_checkpoint_reader import NewCheckpointReader

    return NewCheckpointReader(str(prefix))


def set_initializer(model: onnx.ModelProto, name: str, value: np.ndarray, allow_new: bool = False) -> None:
    for index, tensor in enumerate(model.graph.initializer):
        if tensor.name == name:
            if tuple(tensor.dims) != value.shape:
                raise ValueError(f"{name}: ONNX {tuple(tensor.dims)} vs checkpoint {value.shape}")
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)
            )
            return
    if allow_new:
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name))
        return
    raise KeyError(f"Missing ONNX initializer {name}")


def build_logits(template: onnx.ModelProto, reader, stem_index: int) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.CopyFrom(template)
    for j, layer in enumerate(CONVS):
        tf_scope = scope("conv2d", 7 * stem_index + j)
        for key in ("weight", "bias"):
            tf_key = "kernel" if key == "weight" else "bias"
            array = reader.get_tensor(f"{tf_scope}/{tf_key}")
            if key == "weight":
                array = array.transpose(3, 2, 0, 1)
            set_initializer(model, f"{layer}.{key}", array)
    for j, layer in enumerate(UPS):
        tf_scope = scope("conv2d_transpose", 6 * stem_index + j)
        for key in ("weight", "bias"):
            tf_key = "kernel" if key == "weight" else "bias"
            array = reader.get_tensor(f"{tf_scope}/{tf_key}")
            if key == "weight":
                array = array.transpose(3, 2, 0, 1)
            set_initializer(model, f"{layer}.{key}", array)
    for j, layer in enumerate(BNS):
        tf_scope = scope("batch_normalization", 12 * stem_index + BN_OFFSETS[j])
        for target, source in (
            ("weight", "gamma"), ("bias", "beta"),
            ("running_mean", "moving_mean"), ("running_var", "moving_variance"),
        ):
            set_initializer(
                model, f"{layer}.{target}", reader.get_tensor(f"{tf_scope}/{source}"),
                allow_new=target.startswith("running_"),
            )
        bn_node_name = (
            "node__native_batch_norm_legit_no_training__0" if j == 0
            else f"node__native_batch_norm_legit_no_training_{j}__0"
        )
        bn_node = next(node for node in model.graph.node if node.name == bn_node_name)
        bn_node.input[3] = f"{layer}.running_mean"
        bn_node.input[4] = f"{layer}.running_var"

    # The 5-stem checkpoint uses unet.softmax_unet: up7 is a logit, followed
    # by softmax across all five U-Nets. Emit each logit and apply softmax on host.
    nodes = [node for node in model.graph.node if node.name not in ("node_sigmoid", "node_mul_164")]
    if len(nodes) != len(model.graph.node) - 2:
        raise RuntimeError("Unexpected ONNX template tail")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    transpose = next(node for node in model.graph.node if node.name == "node_permute_1")
    if list(transpose.input) != ["mul_164"]:
        raise RuntimeError("Unexpected ONNX template output")
    transpose.input[0] = "conv2d_6"
    del model.metadata_props[:]
    for key, value in {"source": "deezer/spleeter v1.4.0 5stems", "output": "pre-softmax logits", "stem": STEMS[stem_index]}.items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, value
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/5stems-checkpoint/model"))
    parser.add_argument("--template", type=Path, default=Path("models/4stems-onnx/vocals.fp16.onnx"))
    parser.add_argument("--pure-reader", type=Path, help="Path to shawwn/tensorflow-checkpoint-reader checkout")
    parser.add_argument("--output", type=Path, default=Path("models/5stems-fp16"))
    parser.add_argument("--pnnx", default=os.environ.get("PNNX", "pnnx"))
    args = parser.parse_args()
    prepare_sources(args.checkpoint, args.template)
    reader = open_checkpoint(args.checkpoint, args.pure_reader)
    with tempfile.TemporaryDirectory(prefix="spleeter5-") as temp:
        fp32 = promote_fp16_onnx(args.template, Path(temp) / "template.onnx")
        template = onnx.load(fp32)
        for index, stem in enumerate(STEMS):
            logits = build_logits(template, reader, index)
            onnx_path = Path(temp) / f"{stem}.onnx"
            onnx.save(logits, onnx_path)
            convert_model(args.pnnx, onnx_path, args.output, stem, False)


if __name__ == "__main__":
    main()
