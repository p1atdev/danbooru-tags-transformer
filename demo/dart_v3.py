from typing import Literal

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
# from optimum.onnxruntime import ORTModelForCausalLM

import gradio as gr

ALL_MODELS = {
    # "p1atdev/dart-v2-sft": {
    #     "type": "sft",
    # },
    "p1atdev/dart-v3-llama-8L-241005_241006-sft": {
        "type": "sft",
    },
    "p1atdev/dart-v3-llama-8L-241005_241007-sft": {
        "type": "sft",
    },
    "p1atdev/dart-v3-llama-8L-241003": {
        "type": "pretrain",
    },
}


def prepare_models(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # file_name="model_quantized.onnx",
    )

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


def compose_prompt(
    type: str = "pretrain",
    copyright: str = "",
    character: str = "",
    general: str = "",
    rating: str = "<|rating:general|>",
    aspect_ratio: str = "<|aspect_ratio:tall|>",
    length: str = "<|length:long|>",
):
    if type == "pretrain":
        prompt = (
            f"<|bos|>"
            f"{rating}{aspect_ratio}{length}"
            f"<copyright>{copyright.strip()}</copyright>"
            f"<character>{character.strip()}</character>"
            f"<general>{general.strip()}"
        )
    else:
        prompt = (
            f"<|bos|>"
            f"{rating}{aspect_ratio}{length}"
            f"<copyright>{copyright.strip()}</copyright>"
            f"<character>{character.strip()}</character>"
            f"<general>{general.strip()}<|input_end|>"
        )

    return prompt


@torch.no_grad()
def generate_tags(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
):
    input_ids = tokenizer.encode_plus(prompt, return_tensors="pt").input_ids
    print(
        tokenizer.decode(
            input_ids[0],
            skip_special_tokens=False,
        )
    )
    output = model.generate(
        input_ids.to(model.device),
        do_sample=True,
        temperature=1,
        top_p=0.9,
        top_k=100,
        num_beams=1,
        num_return_sequences=1,
        max_length=256,
    )

    return ", ".join(
        [
            token
            for token in tokenizer.batch_decode(output[0], skip_special_tokens=True)
            if token.strip() != ""
        ]
    )


def main():
    print("Loading models...")

    models = {
        model_name: prepare_models(model_name) for model_name in ALL_MODELS.keys()
    }

    def on_generate(
        model_name: str,
        copyright: str,
        character: str,
        general: str,
        rating: str,
        aspect_ratio: str,
        length: str,
        identity: str,
    ):
        model = models[model_name]["model"]
        tokenizer = models[model_name]["tokenizer"]

        prompt = compose_prompt(
            type=ALL_MODELS[model_name]["type"],
            copyright=copyright,
            character=character,
            general=general,
            rating=f"<|rating:{rating}|>",
            aspect_ratio=f"<|aspect_ratio:{aspect_ratio}|>",
            length=f"<|length:{length}|>",
        )

        print(prompt)

        return generate_tags(model, tokenizer, prompt)

    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    input_copyright = gr.Textbox(
                        label="Copyright",
                        placeholder="vocaloid",
                    )
                    input_character = gr.Textbox(
                        label="Character tags",
                        placeholder="hatsune miku",
                    )
                    input_general = gr.Textbox(
                        label="General tags",
                        placeholder="1girl, ...",
                    )

                    input_rating = gr.Radio(
                        label="Rating",
                        choices=[
                            # "sfw",
                            "general",
                            "sensitive",
                            # "nsfw",
                            "questionable",
                            "explicit",
                        ],
                        value="general",
                    )
                    input_aspect_ratio = gr.Radio(
                        label="Aspect ratio",
                        choices=["ultra_wide", "wide", "square", "tall", "ultra_tall"],
                        value="tall",
                    )
                    input_length = gr.Radio(
                        label="Length",
                        choices=["very_short", "short", "medium", "long", "very_long"],
                        value="long",
                    )

                    model_name = gr.Dropdown(
                        label="Model",
                        choices=list(ALL_MODELS.keys()),
                        value=list(ALL_MODELS.keys())[0],
                    )

                generate_btn = gr.Button(value="Generate", variant="primary")

            with gr.Column():
                generated_tags = gr.Textbox(
                    label="Generated tags",
                )

        generate_btn.click(
            on_generate,
            inputs=[
                model_name,
                input_copyright,
                input_character,
                input_general,
                input_rating,
                input_aspect_ratio,
                input_length,
            ],
            outputs=[generated_tags],
        )

    ui.launch()


if __name__ == "__main__":
    main()
