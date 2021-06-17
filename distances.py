"""
Calculate distances for d-prose jsonlines file
"""
from itertools import combinations_with_replacement
import os
from typing import List

from fastdtw import fastdtw
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import typer
import ujson

from event_classify.datasets import EventType
from event_classify.util import smooth_bins

app = typer.Typer()


def get_narrativity_scores(data, smoothing_span=30):
    scores = []
    for span in data["annotations"]:
        scores.append(EventType.from_tag_name(span["predicted"]).get_narrativity_score())
    series = pd.Series(scores).rolling(smoothing_span, center=True, win_type='cosine').mean()
    series = series[series.notnull()]
    return int(data["dprose_id"]), series


@app.command()
def calculate_distances(dprose_json: str, out_path: str, display: List[int] = []):
    series = {}
    jsonlines = open(dprose_json, "r")
    for i, line in tqdm(enumerate(jsonlines)):
        doc = ujson.loads(line)
        dprose_id, scores = get_narrativity_scores(doc)
        series[dprose_id] = list(scores.values)
        if dprose_id in display:
            plt.plot(scores.values, label=doc["title"])
    plt.show()
    df = pd.DataFrame(None, columns=sorted(series.keys()), index=sorted(series.keys()))
    # Diagonal is zeros
    for el in sorted(series.keys()):
        df[el][el] = 0
    for (outer_id, outer_scores), (inner_id, inner_scores) in tqdm(combinations_with_replacement(series.items(), 2)):
        if outer_id != inner_id:
            dist, _ = fastdtw(outer_scores, inner_scores)
            df[outer_id][inner_id] = dist
            df[inner_id][outer_id] = dist
    df.to_pickle(out_path)
    print(df)




if __name__ == "__main__":
    app()
