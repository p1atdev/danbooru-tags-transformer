import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase


import gradio as gr

MODELS = [
    "p1atdev/dart-v2-llama-100m",
    "p1atdev/dart-v2-mistral-100m",
    "p1atdev/dart-v2-mixtral-100m",
]


def prepare_models(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


def compose_prompt(
    copyright: str = "",
    character: str = "",
    general: str = "",
    rating: str = "<|rating:sfw|>",
    aspect_ratio: str = "<|aspect_ratio:tall|>",
    length: str = "<|length:long|>",
):
    prompt = (
        f"<|bos|>"
        f"<copyright>{copyright.strip()}</copyright>"
        f"<character>{character.strip()}</character>"
        f"{rating}{aspect_ratio}{length}"
        f"<general>{general.strip()}"
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
        input_ids,
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
    models = {model_name: prepare_models(model_name) for model_name in MODELS}

    def on_generate(
        model_name: str,
        copyright: str,
        character: str,
        general: str,
        rating: str,
        aspect_ratio: str,
        length: str,
    ):
        model = models[model_name]["model"]
        tokenizer = models[model_name]["tokenizer"]

        prompt = compose_prompt(
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
                            "sfw",
                            "general",
                            "sensitive",
                            "nsfw",
                            "questionable",
                            "explicit",
                        ],
                        value="sfw",
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
                        choices=list(MODELS),
                        value=MODELS[0],
                    )

                generate_btn = gr.Button(value="Generate", variant="primary")

            with gr.Column():
                generated_tags = gr.Label(
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
