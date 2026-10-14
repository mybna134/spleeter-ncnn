#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Convert Spleeter 2-stem and 4-stem ONNX exports to NCNN FP16 weights."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request


RELEASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "source-separation-models/sherpa-onnx-spleeter-2stems.tar.bz2"
)
ARCHIVE_SIZE = 74_682_545
STEMS = {
    "2stems": ("vocals", "accompaniment"),
    "4stems": ("vocals", "drums", "bass", "other"),
}
FOUR_STEM_URL = "https://huggingface.co/Best-Practice/spleeter-4stems-onnx/resolve/main/{}.fp16.onnx"


def download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == ARCHIVE_SIZE:
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading {RELEASE_URL}", flush=True)
    try:
        with urllib.request.urlopen(RELEASE_URL, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if partial.stat().st_size != ARCHIVE_SIZE:
            raise RuntimeError(f"Unexpected archive size: {partial.stat().st_size}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def extract_onnx(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as source:
        members = {Path(member.name).name: member for member in source if member.isfile()}
        for stem in STEMS["2stems"]:
            filename = f"{stem}.onnx"
            if filename not in members:
                raise RuntimeError(f"{filename} is missing from {archive}")
            with source.extractfile(members[filename]) as content, (destination / filename).open("wb") as output:
                shutil.copyfileobj(content, output)


def promote_fp16_onnx(source: Path, destination: Path) -> Path:
    """PNNX needs FP32 operators; FP16 weights are restored at NCNN export."""
    import onnx
    from onnx import numpy_helper
    import numpy as np

    model = onnx.load(source)
    if not any(t.data_type == onnx.TensorProto.FLOAT16 for t in model.graph.initializer):
        return source
    for tensor in model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT16:
            array = numpy_helper.to_array(tensor).astype(np.float32)
            tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))
    for node in model.graph.node:
        for attribute in node.attribute:
            if node.op_type == "Cast" and attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT16:
                attribute.i = onnx.TensorProto.FLOAT
            if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == onnx.TensorProto.FLOAT16:
                array = numpy_helper.to_array(attribute.t).astype(np.float32)
                attribute.t.CopyFrom(numpy_helper.from_array(array, attribute.t.name))
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    onnx.save(model, destination)
    return destination


def convert_model(pnnx: str, onnx: Path, destination: Path, stem: str, promote: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="spleeter-pnnx-") as temp:
        temporary = Path(temp)
        if promote:
            onnx = promote_fp16_onnx(onnx, temporary / f"{stem}.fp32.onnx")
        subprocess.run(
            [
                pnnx,
                str(onnx.resolve()),
                "inputshape=[2,1,512,1024]",
                "fp16=1",
                f"ncnnparam={temporary / (stem + '.param')}",
                f"ncnnbin={temporary / (stem + '.bin')}",
                f"pnnxparam={temporary / (stem + '.pnnx.param')}",
                f"pnnxbin={temporary / (stem + '.pnnx.bin')}",
                f"pnnxpy={temporary / (stem + '_pnnx.py')}",
                f"pnnxonnx={temporary / (stem + '.pnnx.onnx')}",
                f"ncnnpy={temporary / (stem + '_ncnn.py')}",
            ],
            check=True,
        )
        param = temporary / f"{stem}.param"
        weights = temporary / f"{stem}.bin"
        if not param.is_file() or not weights.is_file() or weights.stat().st_size == 0:
            raise RuntimeError(f"pnnx did not create a complete NCNN model for {stem}")
        unsupported = {"Conv", "ConvTranspose", "Tensor.to"}
        if any(line.split()[0] in unsupported or line.startswith("pnnx.")
               for line in param.read_text().splitlines()[2:]):
            raise RuntimeError(f"pnnx left unsupported operators in {param}")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(param, destination / param.name)
        shutil.copy2(weights, destination / weights.name)
        print(f"{stem}: {weights.stat().st_size / 1024 / 1024:.1f} MiB FP16 NCNN weights", flush=True)


def download_four_stems(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for stem in STEMS["4stems"]:
        target = destination / f"{stem}.fp16.onnx"
        if target.is_file() and target.stat().st_size > 10_000_000:
            continue
        partial = target.with_suffix(".partial")
        try:
            print(f"Downloading {stem} ONNX", flush=True)
            with urllib.request.urlopen(FOUR_STEM_URL.format(stem), timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            if partial.stat().st_size < 10_000_000:
                raise RuntimeError(f"Incomplete ONNX download for {stem}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=STEMS, default="2stems")
    parser.add_argument("--archive", type=Path, help="Existing sherpa-onnx Spleeter 2-stem archive")
    parser.add_argument("--onnx-dir", type=Path, help="Directory with vocals.onnx and accompaniment.onnx")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pnnx", default=os.environ.get("PNNX", "pnnx"))
    args = parser.parse_args()
    if args.archive and args.model != "2stems":
        parser.error("--archive is only for 2stems")
    if args.archive and args.onnx_dir:
        parser.error("Choose --archive or --onnx-dir")
    output = args.output or Path(f"models/{args.model}-fp16")
    if args.onnx_dir:
        source = args.onnx_dir
    elif args.model == "2stems":
        archive = args.archive or Path("models/sherpa-onnx-spleeter-2stems.tar.bz2")
        if not args.archive:
            download_archive(archive)
        with tempfile.TemporaryDirectory(prefix="spleeter-onnx-") as temp:
            source = Path(temp)
            extract_onnx(archive, source)
            for stem in STEMS["2stems"]:
                convert_model(args.pnnx, source / f"{stem}.onnx", output, stem, False)
        return
    else:
        source = Path("models/4stems-onnx")
        download_four_stems(source)
    for stem in STEMS[args.model]:
        onnx = source / f"{stem}.onnx"
        if not onnx.is_file() and args.model == "4stems":
            onnx = source / f"{stem}.fp16.onnx"
        if not onnx.is_file():
            raise FileNotFoundError(onnx)
        convert_model(args.pnnx, onnx, output, stem, args.model == "4stems")


if __name__ == "__main__":
    main()
