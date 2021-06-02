import torch
from tqdm import tqdm
import catma_gitlab as catma
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from collections import Counter, defaultdict
import json
from transformers import AutoTokenizer, ElectraForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
# This is private API but the easiest way to plot the confusion matrix for now...
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

from datasets import SimpleEventDataset, SpanAnnotation, EventType

CLASS_WEIGHTS = torch.tensor([0.9, 17.0, 0.48, 1.32])


def plot_confusion_matrix(target, hypothesis, normalize="true"):
    cm = confusion_matrix(target, hypothesis, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["non-event", "change of state", "process", "stative event"])
    return disp.plot()


def train(train_loader, test_loader, model, device="cpu:0"):
    class_weights = CLASS_WEIGHTS.to(device)
    model.to(device)
    optimizer = SGD(model.parameters(), lr=0.001)
    for epoch in range(10):
        loss_epoch = 0
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, (input_data, labels, _) in enumerate(pbar):
            out = model(**input_data.to(device))
            loss = cross_entropy(out.logits, labels.to(device), weight=class_weights)
            loss_epoch += loss.item()
            pbar.set_postfix({"loss": loss_epoch / (i + 1)})
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch == 9:
            eval(test_loader, model, device, out_file=open("/tmp/out.json", "w"), show_confusion_matrix=True)
        else:
            eval(test_loader, model, device)


def eval(loader, model, device=None, out_file=None, show_confusion_matrix=False):
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
        print(f"=== Gold: {annotation.event_type}, predicted: {EventType(label.item())}")
        print(annotation.text)
    print(classification_report(torch.cat(gold), torch.cat(predictions)))
    if show_confusion_matrix:
        _ = plot_confusion_matrix(torch.cat(gold), torch.cat(predictions))
        plt.show()
    if out_file is not None:
        out_list = []
        for text, annotations in texts.items():
            out_list.append({
                "text": text.plain_text,
                "title": text.title,
                "annotations": annotations,
            })
        json.dump(out_list, out_file)



def main():
    project = catma.CatmaProject(
        ".",
        "CATMA_DD5E9DF1-0F5C-4FBD-B333-D507976CA3C7_EvENT_root",
        filter_intrinsic_markup=False,
    )
    dataset = SimpleEventDataset(project, ["Verwandlung_MV", "Krambambuli_MW", "Verwandlung_MW"])
    tokenizer = AutoTokenizer.from_pretrained(
        "german-nlp-group/electra-base-german-uncased"
    )
    model = ElectraForSequenceClassification.from_pretrained(
        "german-nlp-group/electra-base-german-uncased",
        num_labels=4,
    )
    counts = Counter(el.event_type for el in dataset)
    total = sum(counts.values())
    print(f"Number of examples: {total}")
    print("Recommended class weights:")
    for k, value in counts.items():
        print("\tClass:", k, 1 / (value / (total / len(counts))))
    train_data, test_data = random_split(dataset, [(total // 10 * 7), total - (total // 10 * 7)])
    train_loader = DataLoader(
        train_data,
        batch_size=8,
        collate_fn=lambda list_: SpanAnnotation.to_batch(list_, tokenizer)
    )
    test_loader = DataLoader(
        test_data,
        batch_size=8,
        collate_fn=lambda list_: SpanAnnotation.to_batch(list_, tokenizer)
    )
    train(train_loader, test_loader, model, device="cuda:0")


if __name__ == "__main__":
    main()
