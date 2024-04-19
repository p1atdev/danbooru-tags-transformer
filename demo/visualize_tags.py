import torch
import io

from transformers import AutoTokenizer, DebertaV2Model


TOKENIZER_NAME = "p1atdev/dart2vec-deberta_1"
MODEL_NAME = "p1atdev/dart2vec-deberta_1"


def prepare_embeddings():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = DebertaV2Model.from_pretrained(MODEL_NAME)
    embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()

    id2label = tokenizer.get_vocab()

    return embeddings[: len(id2label)], id2label


def main():
    embeddings, id2label = prepare_embeddings()

    print(embeddings.shape, len(id2label))

    # ファイルに保存
    out_v = io.open("data/vectors.tsv", "w", encoding="utf-8")
    out_m = io.open("data/metadata.tsv", "w", encoding="utf-8")
    for index, label in enumerate(id2label):
        vector = embeddings[index]
        out_v.write("\t".join([str(x) for x in vector]) + "\n")
        out_m.write(label + "\n")
    out_v.close()
    out_m.close()

    # the go to https://projector.tensorflow.org/ and load the files


if __name__ == "__main__":
    main()
