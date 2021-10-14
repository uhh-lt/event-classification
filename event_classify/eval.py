from collections import defaultdict
import json
import logging

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
import torch
from tqdm import tqdm

from event_classify.datasets import EventType


def plot_confusion_matrix(target, hypothesis, normalize="true", tick_names=["Non Event", "Stative Event", "Process", "Change of State"]):
    cm = confusion_matrix(target, hypothesis, normalize=normalize)
    cm2 = confusion_matrix(target, hypothesis, normalize=None)
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
def evaluate(loader, model, device=None, out_file=None, save_confusion_matrix=False, epoch=None):
    model.to(device)
    if device is None:
        device = model.device
    model.eval()
    gold = []
    predictions = []
    texts = defaultdict(list)
    labled_annotations = []
    all_predictions = defaultdict(list)
    all_labels = defaultdict(list)
    for input_data, gold_labels, annotations in tqdm(loader, desc="Evaluating"):
        out = model(**input_data.to(device))
        for anno, label in zip(annotations, out.event_type.cpu()):
            labled_annotations.append((label, anno))
        if out_file is not None:
            not_non_event_index = 0 # Counted up whenever an event is not a non-event
            for i, anno in enumerate(annotations):
                out_data = anno.output_dict(out.event_type[i].item())
                out_data["gold_label"] = EventType(gold_labels.event_type[i].item()).to_string()
                out_data["properties"] = dict()
                if anno.event_type != EventType.NON_EVENT:
                    for prop in ["mental", "iterative"]:
                        out_data["properties"][prop] = getattr(gold_labels, prop)[not_non_event_index].item()
                for prop in ["thought_representation", "speech_type"]:
                    out_data["properties"][prop] = EventType(getattr(gold_labels, prop)[i].item()).to_string()
                texts[anno.document_text].append(out_data)
                if anno.event_type != EventType.NON_EVENT:
                    not_non_event_index += 1
        predictions.append(out.event_type.cpu())
        for name in ["mental", "iterative"]:
            all_predictions[name].append(torch.masked_select(getattr(out, name).cpu(), gold_labels.event_type != 0))
            all_labels[name].append(getattr(gold_labels, name).cpu())
            assert len(all_labels[name][-1]) == len(all_predictions[name][-1])
        for name in ["speech_type", "thought_representation"]:
            all_predictions[name].append(getattr(out, name).cpu())
            all_labels[name].append(getattr(gold_labels, name).cpu())
        if gold_labels is not None:
            gold.append(gold_labels.event_type.cpu())
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
    if (save_confusion_matrix or epoch is not None) and len(gold) > 0:
        plt.rc("text")
        plt.rc("font", family="serif", size=12)
        gold_data_converted = torch.tensor([EventType(gold_label.item()).get_narrativity_ordinal() for gold_label in torch.cat(gold)], dtype=torch.int)
        predictions_converted = torch.tensor([EventType(prediction.item()).get_narrativity_ordinal() for prediction in torch.cat(predictions)], dtype=torch.int)
        _ = plot_confusion_matrix(gold_data_converted, predictions_converted)
        plt.tight_layout()
        plt.gcf().subplots_adjust(left=0.2)
        plt.savefig(f"confusion_matrix_event-types_{epoch}.pdf")
        mlflow.log_artifact(f"confusion_matrix_event-types_{epoch}.pdf")
        plt.clf()
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
        extra_metrics = {}
        for name, values in all_predictions.items():
            plt.rc("text")
            plt.rc("font", family="serif", size=12)
            plot_confusion_matrix(torch.cat(values), torch.cat(all_labels[name]), tick_names="auto")
            extra_metrics[name + " macro f1"] = classification_report(
                torch.cat(values),
                torch.cat(all_labels[name]),
                output_dict=True
            )["macro avg"]["f1-score"]
            plt.tight_layout()
            plt.gcf().subplots_adjust(left=0.2)
            file_name = f"confusion_matrix_{name}_{epoch if epoch is not None else 'end'}.pdf"
            plt.savefig(file_name)
            plt.clf()
            mlflow.log_artifact(file_name)
        return report["weighted avg"]["f1-score"], report["macro avg"]["f1-score"], torch.cat(predictions).cpu(), extra_metrics
    else:
        return None, None, torch.cat(predictions).cpu(), {}

