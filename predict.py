import typer
import json
import os
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from torch.utils.data import DataLoader

from event_classify.datasets import JSONDataset, SpanAnnotation, EventType
from event_classify.eval import evaluate

app = typer.Typer()


@app.command()
def main(segmented_json: str, out_path: str, model_path: str, device: str = "cuda:0", batch_size: int = 16):
    """
    Add predictions to segmented json file.

    Args:
    model_path: Path to run directory containing saved model and tokenizer
    """
    dataset = JSONDataset(segmented_json)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda list_: SpanAnnotation.to_batch(list_, tokenizer),
        shuffle=True,
    )
    tokenizer: ElectraTokenizer = ElectraTokenizer.from_pretrained(
        os.path.join(model_path, "tokenizer")
    )
    model = ElectraForSequenceClassification.from_pretrained(
        os.path.join(model_path, "best-model"),
        num_labels=4,
    )
    model.to(device)
    _, _, predictions = evaluate(loader, model, device=device)
    dataset.save_json(out_path, [EventType(p.item()) for p in predictions])

if __name__ == "__main__":
    app()
