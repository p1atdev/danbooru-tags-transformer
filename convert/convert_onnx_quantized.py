import argparse

from pathlib import Path

import torch
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from optimum.exporters.onnx.model_configs import MistralOnnxConfig
from optimum.exporters.onnx import export, main_export
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

MODEL_NAME = "p1atdev/dart-v2-base"
SAVE_DIR = "./output/onnx-v2-base"


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize a model")
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default=MODEL_NAME,
        help="Model name",
    )
    parser.add_argument(
        "--save_dir",
        "-o",
        type=str,
        default=SAVE_DIR,
        help="Save directory",
    )
    parser.add_argument(
        "--quantize",
        "-q",
        action="store_true",
        help="Quantize the model",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force convert the model even if the model is not supported",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    save_dir = args.save_dir
    do_quantize = args.quantize
    is_force = args.force

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    onnx_config = MistralOnnxConfig(config, task="text-generation")

    if not is_force:
        model = ORTModelForCausalLM.from_pretrained(model_name, export=True)

        if do_quantize:
            qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
            quantizer = ORTQuantizer.from_pretrained(model)

            quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)

            print("Quantization done!")
        else:
            print("Skipping quantization")
            model.save_pretrained(save_dir)

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

        export(
            model,
            onnx_config,
            Path(save_dir) / "model.onnx",
            onnx_config.DEFAULT_ONNX_OPSET,
        )
        config.save_pretrained(save_dir)

        if do_quantize:
            ort_model = ORTModelForCausalLM.from_pretrained(
                save_dir, export=False, use_cache=False, use_io_binding=False
            )
            qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
            quantizer = ORTQuantizer.from_pretrained(ort_model)

            quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)

            print("Quantization done!")


if __name__ == "__main__":
    main()
