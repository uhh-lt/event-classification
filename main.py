import json
import logging
import math
import os
from collections import Counter
from typing import List, Iterator, Optional

import catma_gitlab as catma
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import ElectraTokenizer, ElectraForSequenceClassification

from event_classify.eval import evaluate
from event_classify.config import Config, DatasetConfig
from event_classify.label_smoothing import LabelSmoothingLoss
from event_classify.datasets import SimpleEventDataset, SpanAnnotation

CLASS_WEIGHTS = torch.tensor([0.0003, 0.15, 0.0003, 0.0005])


def add_special_tokens(model, tokenizer):
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": ["<EE>", "<SE>"],
        }
    )
    model.resize_token_embeddings(len(tokenizer))


def train(train_loader, dev_loader, model, config: Config, writer: SummaryWriter):
    class_weights = CLASS_WEIGHTS.to(config.device)
    model.to(config.device)
    optimizer = SGD(model.parameters(), lr=config.learning_rate)
    f1s: List[float] = []
    scheduler: Optional[LambdaLR] = None
    if config.scheduler.enable:
        scheduler = LambdaLR(
            optimizer,
            lambda epoch: 1 - (epoch / config.scheduler.epochs),
        )
    loss_func = LabelSmoothingLoss(weight=CLASS_WEIGHTS)
    for epoch in range(config.epochs):
        loss_epoch: float = 0.0
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, (input_data, labels, _) in enumerate(pbar):
            out = model(**input_data.to(config.device))
            if config.label_smoothing:
                loss = cross_entropy(
                    out.logits, labels.to(config.device), weight=class_weights
                )
            else:
                loss = loss_func(out.logits, labels.to(config.device))
            loss_epoch += float(loss.item())
            pbar.set_postfix({"mean epoch loss": loss_epoch / (i + 1)})
            loss.backward()
            optimizer.step()
            model.zero_grad()
            writer.add_scalar("Loss", loss.item())
        if scheduler is not None:
            scheduler.step()
        if dev_loader is not None:
            weighted_f1, macro_f1, _ = evaluate(dev_loader, model, config.device)
            if config.optimize == "weighted f1":
                f1: float = float(weighted_f1)
            elif config.optimize == "macro f1":
                f1: float = float(macro_f1)
            else:
                logging.warning("Invalid optimization metric, defaulting to weighted.")
                f1 = weighted_f1
            if (len(f1s) > 0 and f1 > max(f1s)) or len(f1s) == 0:
                model.save_pretrained("best-model")
            f1s.append(f1)
            if scheduler is not None:
                writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Weighted F1", weighted_f1, epoch)
            writer.add_scalar("Macro F1", macro_f1, epoch)
            if len(f1s) > 0 and max(f1s) not in f1s[-config.patience :]:
                logging.info("Ran out of patience, stopping training.")
                return


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
            generator=torch.Generator().manual_seed(13),
        )
    else:
        train_dataset = SimpleEventDataset(
            project, ["Effi_Briest_MW", "Krambambuli_MW"]
        )
        test_dataset = SimpleEventDataset(
            project,
            ["Verwandlung_MV"],
        )
        dev_dataset = None

    return train_dataset, dev_dataset, test_dataset


def build_loaders(tokenizer: ElectraTokenizer, datasets: List[Dataset], config: Config) -> Iterator[DataLoader]:
    for ds in datasets:
        if ds:
            yield DataLoader(
                ds,
                batch_size=config.batch_size,
                collate_fn=lambda list_: SpanAnnotation.to_batch(list_, tokenizer),
                shuffle=True,
            )
        else:
            yield None


def print_target_weights(dataset):
    counts = Counter(el.event_type for el in dataset)
    logging.info("Recommended class weights:")
    output_classes = []
    output_weights = []
    for event_type, value in sorted(counts.items(), key=lambda x: x[0].value):
        weight = 1 / value
        logging.info(
            f"Class: {event_type}, {weight}"
        )


@hydra.main(config_name="conf/config")
def main(config: Config):
    hydra_run_name = HydraConfig.get().run.dir.replace("outputs/", "").replace("/", "_")
    tokenizer: ElectraTokenizer = ElectraTokenizer.from_pretrained(
        config.pretrained_model,
    )
    tokenizer.save_pretrained("tokenizer")
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
        model = ElectraForSequenceClassification.from_pretrained(
            "best-model"
        )
    logging.info("Dev set results")
    evaluate(
        dev_loader,
        model,
        device=config.device,
        out_file=open("predictions.json", "w"),
        save_confusion_matrix=True,
    )
    logging.info("Test set results")
    evaluate(
        test_loader,
        model,
        device=config.device,
        out_file=open("predictions.json", "w"),
        save_confusion_matrix=True,
    )


if __name__ == "__main__":
    main()
