from collections import defaultdict
import logging
import json

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from event_classify.datasets import EventType


def plot_confusion_matrix(target, hypothesis, normalize="true"):
    cm = confusion_matrix(target, hypothesis, normalize=normalize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["non-event", "change of state", "process", "stative event"],
    )
    return disp.plot()


def evaluate(loader, model, device=None, out_file=None, save_confusion_matrix=False):
    model.to(device)
    if device is None:
        device = model.device
    model.eval()
    gold = []
    predictions = []
    texts = defaultdict(list)
    labled_annotations = []
    for input_data, gold_labels, annotations in tqdm(loader, desc="Evaluating"):
        out = model(**input_data.to(device))
        predicted_labels = out.logits.argmax(-1)
        for anno, label in zip(annotations, predicted_labels.cpu()):
            labled_annotations.append((label, anno))
        if out_file is not None:
            for anno, label in zip(annotations, predicted_labels.cpu()):
                texts[anno.document_text].append(anno.output_dict(label.item()))
        predictions.append(predicted_labels.cpu())
        if gold_labels is not None:
            gold.append(gold_labels.cpu())
    for label, annotation in labled_annotations:
        logging.debug(
            f"=== Gold: {annotation.event_type}, predicted: {EventType(label.item())}"
        )
        logging.debug(annotation.text)
    if len(gold) == 0:
        logging.warning("No gold labels given, not calculating classification report")
        report = None
    else:
        report = classification_report(
            torch.cat(gold), torch.cat(predictions), output_dict=True
        )
        logging.info(classification_report(torch.cat(gold), torch.cat(predictions)))
    if save_confusion_matrix and len(gold) > 0:
        _ = plot_confusion_matrix(torch.cat(gold), torch.cat(predictions))
        plt.savefig("confusion_matrix.pdf")
    if out_file is not None:
        out_list = []
        for text, annotations in texts.items():
            out_list.append(
                {
                    "text": text,
                    "title": None,
                    "annotations": annotations,
                }
            )
        json.dump(out_list, out_file)
    if report is not None:
        return report["weighted avg"]["f1-score"], report["macro avg"]["f1-score"], torch.cat(predictions)
    else:
        return None, None, torch.cat(predictions)

