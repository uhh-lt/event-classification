import json
import logging
import math
import os
from collections import Counter, defaultdict
from typing import List, Iterator

import catma_gitlab as catma
import hydra
import matplotlib.pyplot as plt
import torch
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import classification_report, confusion_matrix
# This is private API but the easiest way to plot the confusion matrix for now...
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, ElectraForSequenceClassification

from event_classify.config import Config, DatasetConfig
from event_classify.datasets import (EventType, SimpleEventDataset,
                                     SpanAnnotation)

CLASS_WEIGHTS = torch.tensor([0.9, 17.0, 0.48, 1.32])


def add_special_tokens(model, tokenizer):
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": ["<EE>", "<SE>"],
        }
    )
    model.resize_token_embeddings(len(tokenizer))


def plot_confusion_matrix(target, hypothesis, normalize="true"):
    cm = confusion_matrix(target, hypothesis, normalize=normalize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["non-event", "change of state", "process", "stative event"],
    )
    return disp.plot()


def train(train_loader, dev_loader, model, config: Config, writer: SummaryWriter):
    class_weights = CLASS_WEIGHTS.to(config.device)
    model.to(config.device)
    optimizer = SGD(model.parameters(), lr=config.learning_rate)
    f1s: List[float] = []
    scheduler = CyclicLR(
        optimizer,
        base_lr=config.learning_rate / 10,
        max_lr=config.learning_rate,
        step_size_up=(len(train_loader) * config.lr_scheduler_epochs) // 2,
    )
    for epoch in range(config.epochs):
        loss_epoch: float = 0.0
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, (input_data, labels, _) in enumerate(pbar):
            out = model(**input_data.to(config.device))
            loss = cross_entropy(
                out.logits, labels.to(config.device), weight=class_weights
            )
            loss_epoch += float(loss.item())
            pbar.set_postfix({"mean epoch loss": loss_epoch / (i + 1)})
            loss.backward()
            optimizer.step()
            model.zero_grad()
            scheduler.step()
        if dev_loader is not None:
            weighted_f1, macro_f1 = eval(dev_loader, model, config.device)
            if config.optimize == "weighted f1":
                f1: float = float(weighted_f1)
            elif config.optimize == "macro f1":
                f1: float = float(macro_f1)
            else:
                logging.warning("Invalid optimization metric, defaulting to weighted.")
                f1 = weighted_f1
            if (len(f1s) > 0 and f1 > max(f1s)) or len(f1s) == 0:
                torch.save(model, "best-model.pt")
            f1s.append(f1)
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Weighted F1", weighted_f1, epoch)
            writer.add_scalar("Macro F1", macro_f1, epoch)
            if len(f1s) > 0 and max(f1s) not in f1s[-config.patience :]:
                logging.info("Ran out of patience, stopping training.")
                return


def eval(loader, model, device=None, out_file=None, save_confusion_matrix=False):
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
        gold.append(gold_labels.cpu())
    for label, annotation in labled_annotations:
        logging.debug(
            f"=== Gold: {annotation.event_type}, predicted: {EventType(label.item())}"
        )
        logging.debug(annotation.text)
    logging.info(classification_report(torch.cat(gold), torch.cat(predictions)))
    if save_confusion_matrix:
        _ = plot_confusion_matrix(torch.cat(gold), torch.cat(predictions))
        plt.savefig("confusion_matrix.pdf")
    if out_file is not None:
        out_list = []
        for text, annotations in texts.items():
            out_list.append(
                {
                    "text": text.plain_text,
                    "title": text.title,
                    "annotations": annotations,
                }
            )
        json.dump(out_list, out_file)
    report = classification_report(
        torch.cat(gold), torch.cat(predictions), output_dict=True
    )
    return report["weighted avg"]["f1-score"], report["macro avg"]["f1-score"]


def get_datasets(config: DatasetConfig) -> tuple[Dataset]:
    project = catma.CatmaProject(
        hydra.utils.to_absolute_path(config.catma_dir),
        config.catma_uuid,
        filter_intrinsic_markup=False,
    )
    if config.in_domain:
        dataset = SimpleEventDataset(
            project, ["Verwandlung_MV", "Krambambuli_MW", "Effi_Briest_MW"]
        )
        total = len(dataset)
        train_size = math.floor(total * 0.8)
        dev_size = (total - train_size) // 2
        test_size = total - train_size - dev_size
        train_dataset, dev_dataset, test_dataset = random_split(
            dataset,
            [train_size, dev_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_dataset = SimpleEventDataset(
            project, ["Effi_Briest_MW", "Krambambuli_MW"]
        )
        test_dataset = SimpleEventDataset(
            ["Verwandlung_MV"],
        )
        dev_dataset = None

    return train_dataset, dev_dataset, test_dataset


def build_loaders(tokenizer: AutoTokenizer, datasets: List[Dataset], config: Config) -> Iterator[DataLoader]:
    for ds in datasets:
        if ds:
            yield DataLoader(
                ds,
                batch_size=config.batch_size,
                collate_fn=lambda list_: SpanAnnotation.to_batch(list_, tokenizer),
            )
        else:
            yield None


def print_target_weights(dataset):
    counts = Counter(el.event_type for el in dataset)
    logging.info("Recommended class weights:")
    for event_type, value in sorted(counts.items(), key=lambda x: x[0].value):
        logging.info(
            f"Class: {event_type}, {1 / (value / (sum(counts.values()) / len(counts)))}"
        )


@hydra.main(config_name="conf/config")
def main(config: Config):
    hydra_run_name = HydraConfig.get().run.dir.replace("outputs/", "").replace("/", "_")
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model,
    )
    model = ElectraForSequenceClassification.from_pretrained(
        config.pretrained_model,
        num_labels=4,
    )
    writer = SummaryWriter(
        os.path.join(hydra.utils.to_absolute_path("runs"), hydra_run_name)
    )
    add_special_tokens(model, tokenizer)
    datasets = get_datasets(config.dataset)
    print_target_weights(datasets[0])
    assert datasets[0] is not None
    assert datasets[-1] is not None
    train_loader, dev_loader, test_loader = list(
        build_loaders(tokenizer, datasets, config)
    )
    train(train_loader, dev_loader, model, config, writer)
    if dev_loader is not None:
        model = torch.load("best-model.pt")
    eval(
        test_loader,
        model,
        device=config.device,
        out_file=open("predictions.json", "w"),
        save_confusion_matrix=True,
    )


if __name__ == "__main__":
    main()
