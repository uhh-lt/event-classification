from collections import defaultdict
import logging
import json
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from event_classify.datasets import EventType


def plot_confusion_matrix(target, hypothesis, normalize="true"):
    cm = confusion_matrix(target, hypothesis, normalize=normalize)
    cm2 = confusion_matrix(target, hypothesis, normalize=None)
    tick_names = ["Non Event", "Change of State", "Process", "Stative Event"]
    ax = sns.heatmap(
        cm,
        vmin=0.0,
        vmax=1.0,
        cmap=plt.cm.Blues,
        xticklabels=tick_names,
        yticklabels=tick_names,
        square=True,
        annot=True,
        cbar=True
    )
    ax.set_ylabel("True Labels")
    ax.set_xlabel("Predicted Labels")
    # print(cm2)
    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=cm,
    #     display_labels=["non-event", "change of state", "process", "stative event"],
    # )
    # plotted = disp.plot(
    #     cmap=plt.cm.Blues,
    # )
    # plotted.figure.colorbar()
    return ax


@torch.no_grad()
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
        for anno, label in zip(annotations, out.event_kind.cpu()):
            labled_annotations.append((label, anno))
        if out_file is not None:
            for anno, label, gold_label in zip(annotations, out.event_kind, gold_labels.event_kind):
                out_data = anno.output_dict(label.item())
                out_data["gold_label"] = EventType(gold_label.item()).to_string()
                texts[anno.document_text].append(out_data)
        predictions.append(out.event_kind.cpu())
        if gold_labels is not None:
            gold.append(gold_labels.event_kind.cpu())
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
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif", size=12)
        _ = plot_confusion_matrix(torch.cat(gold), torch.cat(predictions))
        plt.tight_layout()
        plt.gcf().subplots_adjust(left=0.2)
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
        return report["weighted avg"]["f1-score"], report["macro avg"]["f1-score"], torch.cat(predictions).cpu()
    else:
        return None, None, torch.cat(predictions).cpu()

