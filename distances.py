"""
Calculate distances for d-prose jsonlines file
"""
from itertools import combinations_with_replacement
import json

from matplotlib import pyplot as plt
from tqdm import tqdm
import typer

from event_classify.util import smooth_bins
from fastdtw import fastdtw

app = typer.Typer()


def get_narrativity_scores(data):
    smoothed_scores = []
    bins = smooth_bins(data["annotations"])
    return data.get("dprose_id"), bins


@app.command()
def calculate_distances(dprose_json: str, out_path: str):
    series = {}
    jsonlines = open(dprose_json, "r")
    for i, line in enumerate(jsonlines):
        doc = json.loads(line)
        dprose_id, scores = get_narrativity_scores(doc)
        series[dprose_id] = scores
        if i > 3:
            break
    out_file = open(out_path, "w")
    for (outer_id, outer_scores), (inner_id, inner_scores) in tqdm(combinations_with_replacement(series.items(), 2)):
        if outer_id != inner_id:
            dist, _ = fastdtw(outer_scores, inner_scores)
            out_file.write(f"{outer_id},{inner_id},{dist}\n")
        # plt.plot(scores)
        # plt.show()


if __name__ == "__main__":
    app()
