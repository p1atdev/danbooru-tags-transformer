import torch
import numpy as np

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient

import gradio as gr

LENGTH_TAG_MAP = {
    "very short": "<|very_short|>",
    "short": "<|short|>",
    "long": "<|long|>",
    "very long": "<|very_long|>",
}

RATING_TAG_MAP = {
    "general": "rating:general",
    "sensitive": "rating:sensitive",
    "questionable": "rating:questionable",
    "explicit": "rating:explicit",
}

BOS = "<|bos|>"
EOS = "<|eos|>"
RATING_BOS = "<rating>"
RATING_EOS = "</rating>"
COPYRIGHT_BOS = "<copyright>"
COPYRIGHT_EOS = "</copyright>"
CHARACTER_BOS = "<character>"
CHARACTER_EOS = "</character>"
GENERAL_BOS = "<general>"
GENERAL_EOS = "</general>"

INPUT_END = "<|input_end|>"

LENGTH_VERY_SHORT = "<|very_short|>"
LENGTH_SHORT = "<|short|>"
LENGTH_LONG = "<|long|>"
LENGTH_VERY_LONG = "<|very_long|>"

NEGATIVE_PROMPT = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
T2I_MODEL_NAME = "cagliostrolab/animagine-xl-3.1"


def prepare_models(model_name: str = "p1atdev/dart-v1-sft"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return tokenizer, model


def tokenize_tags(
    tokenizer, rating: str, copyright: str, character: str, general: str, length: str
):
    inputs = tokenizer.apply_chat_template(
        {
            "rating": rating,
            "copyright": copyright,
            "character": character,
            "general": general,
            "length": length,
        },
        return_tensors="pt",
        tokenize=True,
    )

    return inputs


@torch.no_grad()
def generate_tags(model, inputs):
    outputs = model.generate(inputs, num_return_sequences=2)
    return outputs


def hil_ui():
    tokenizer, model = prepare_models()

    RATING_EOS_ID = tokenizer.convert_tokens_to_ids(RATING_EOS)
    COPYRIGHT_EOS_ID = tokenizer.convert_tokens_to_ids(COPYRIGHT_EOS)
    CHARACTER_EOS_ID = tokenizer.convert_tokens_to_ids(CHARACTER_EOS)
    GENERAL_EOS_ID = tokenizer.convert_tokens_to_ids(GENERAL_EOS)
    PEOPLE_TAG_IDS_LIST = tokenizer.convert_tokens_to_ids(
        ["1girl", "1boy", "2girls", "2boys", "3girls", "3boys", "4girls", "4boys"]
    )

    def split_people_tokens_part(token_ids: list[int]):
        people_tokens = []
        other_tokens = []

        for token in token_ids:
            if token in PEOPLE_TAG_IDS_LIST:
                people_tokens.append(token)
            else:
                other_tokens.append(token)

        return people_tokens, other_tokens

    client = InferenceClient(model=T2I_MODEL_NAME)  # huggingface inference api

    def decode_animagine(token_ids: list[int]):
        def get_part(eos_token_id: int, remains_part: list[int]):
            part = []
            for i, token_id in enumerate(remains_part):
                if token_id == eos_token_id:
                    return part, remains_part[i:]

                part.append(token_id)

            raise Exception("The provided EOS token was not found in the token_ids.")

        # get each part
        rating_part, remains = get_part(RATING_EOS_ID, token_ids)
        copyright_part, remains = get_part(COPYRIGHT_EOS_ID, remains)
        character_part, remains = get_part(CHARACTER_EOS_ID, remains)
        general_part, _ = get_part(GENERAL_EOS_ID, remains)

        # separete people tags (1girl, 1boy, no humans...)
        people_part, other_general_part = split_people_tokens_part(general_part)

        # AnimagineXL v3 style order
        rearranged_tokens = (
            people_part
            + character_part
            + copyright_part
            + other_general_part
            + rating_part
        )

        decoded = tokenizer.decode(rearranged_tokens, skip_special_tokens=True)

        # fix "nsfw" tag
        decoded = decoded.replace("rating:nsfw", "nsfw")

        return decoded

    def generate_images(
        rating: str, copyright: str, character: str, general: str, length: str
    ):
        inputs = tokenize_tags(
            tokenizer,
            RATING_TAG_MAP[rating],
            copyright,
            character,
            general,
            LENGTH_TAG_MAP[length],
        )
        generated_sentences = generate_tags(model, inputs)
        tags_a, tags_b = map(decode_animagine, generated_sentences)

        print("A:", tags_a)
        print("B:", tags_b)

        # get images
        image_a = client.text_to_image(
            prompt=tags_a,
            negative_prompt=NEGATIVE_PROMPT,
            width=896,
            height=1216,
            guidance_scale=6.5,
            num_inference_steps=25,
        )
        image_b = client.text_to_image(
            prompt=tags_b,
            negative_prompt=NEGATIVE_PROMPT,
            width=896,
            height=1216,
            guidance_scale=6.5,
            num_inference_steps=25,
        )

        return image_a, tags_a, image_b, tags_b

    with gr.Blocks() as ui:
        with gr.Group():
            rating_radio = gr.Radio(
                label="Rating",
                choices=["general", "sensitive", "questionable", "explicit"],
                value="general",
                interactive=True,
            )
            copyright_textbox = gr.Textbox(
                label="Copyright", placeholder="vocaloid", value="", interactive=True
            )
            character_textbox = gr.Textbox(
                label="Character",
                placeholder="hatsune miku",
                value="",
                interactive=True,
            )
            general_textbox = gr.Textbox(
                label="General",
                placeholder="1girl, ...",
                value="1girl, blue hair",
                interactive=True,
            )

            length_radio = gr.Radio(
                label="Length",
                choices=["very short", "short", "long", "very long"],
                value="long",
                interactive=True,
            )

            generate_btn = gr.Button(
                value="Generate", variant="primary", interactive=True
            )

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    output_image_a = gr.Image(
                        label="Image 🅰️", height=500, interactive=False
                    )
                    with gr.Accordion(label="Details", open=False):
                        output_text_a = gr.Textbox(
                            label="The prompt used to generate this image",
                            placeholder="No tags generated yet",
                            value="",
                            interactive=False,
                        )

                    a_is_better_btn = gr.Button(
                        value="🅰️ This is better", interactive=True
                    )

            with gr.Column():
                with gr.Group():
                    output_image_b = gr.Image(
                        label="Image 🅱️", height=500, interactive=False
                    )

                    with gr.Accordion(label="Details", open=False):
                        output_text_b = gr.Textbox(
                            label="The prompt used to generate this image",
                            placeholder="No tags generated yet",
                            value="",
                            interactive=False,
                        )

                    b_is_better_btn = gr.Button(
                        value="🅱️ This is better", interactive=True
                    )

        neither_is_better_btn = gr.Button(value="😒 Neither is good", interactive=True)

        generate_btn.click(
            generate_images,
            inputs=[
                rating_radio,
                copyright_textbox,
                character_textbox,
                general_textbox,
                length_radio,
            ],
            outputs=[output_image_a, output_text_a, output_image_b, output_text_b],
        )

    ui.launch()


if __name__ == "__main__":
    hil_ui()
