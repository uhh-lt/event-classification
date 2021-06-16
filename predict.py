import typer
import json
import os
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from torch.utils.data import DataLoader

import catma_gitlab as catma
from event_classify.datasets import JSONDataset, SpanAnnotation, EventType, SimpleEventDataset
from event_classify.eval import evaluate

app = typer.Typer()


def get_model(model_path: str):
    tokenizer: ElectraTokenizer = ElectraTokenizer.from_pretrained(
        os.path.join(model_path, "tokenizer")
    )
    model = ElectraForSequenceClassification.from_pretrained(
        os.path.join(model_path, "best-model"),
        num_labels=4,
    )
    return model, tokenizer


@app.command()
def main(segmented_json: str, out_path: str, model_path: str, device: str = "cuda:0", batch_size: int = 16, special_tokens: bool = True):
    """
    Add predictions to segmented json file.

    Args:
    model_path: Path to run directory containing saved model and tokenizer
    """
    dataset = JSONDataset(segmented_json, include_special_tokens=special_tokens)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda list_: SpanAnnotation.to_batch(list_, tokenizer),
    )
    model, tokenizer = get_model(model_path)
    model.to(device)
    _, _, predictions = evaluate(loader, model, device=device)
    dataset.save_json(out_path, [EventType(p.item()) for p in predictions])


@app.command()
def gold_spans(model_path: str, device: str = "cuda:0", special_tokens: bool = True):
    project = catma.CatmaProject(
        ".",
        "CATMA_DD5E9DF1-0F5C-4FBD-B333-D507976CA3C7_EvENT_root",
        filter_intrinsic_markup=False,
    )
    collection = project.ac_dict["Verwandlung_MV"]
    dataset = SimpleEventDataset(
        project,
        ["Verwandlung_MV"],
        include_special_tokens=special_tokens,
    )
    model, tokenizer = get_model(model_path)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=lambda list_: SpanAnnotation.to_batch(list_, tokenizer),
    )
    evaluate(
        loader,
        model,
        device=device,
        out_file=open("gold_spans.json", "w"),
        save_confusion_matrix=False,
    )

if __name__ == "__main__":
    app()
