import argparse

from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from transformers import AutoTokenizer

MODEL_NAME = "p1atdev/dart-v2-llama-100m"
SAVE_DIR = "./output/onnx-llama-100m"


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
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    save_dir = args.save_dir
    do_quantize = args.quantize

    model = ORTModelForCausalLM.from_pretrained(model_name, export=True)

    if do_quantize:
        qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
        quantizer = ORTQuantizer.from_pretrained(model)

        quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)

        print("Quantization done!")
    else:
        print("Skipping quantization")
        model.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
