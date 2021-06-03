import itertools
import sys

import spacy
from spacy.tokens import Doc, Token
from spacy.language import Language

Doc.set_extension("events", default=[])
Token.set_extension("custom_dep", default=None)


@Language.component("event_segmentation")
def event_segmentation(doc):
    processed = set()
    for token in doc:
        ranges = []
        if token.tag_.startswith("V") and token.tag_.endswith("FIN") and (token not in processed):
            recursed = recurse_children(token, blacklist=list(processed))
            processed |= set(recursed)
            span_indexes = [t.i for t in recursed]
            if token.head.tag_ == "KON":
                span_indexes.append(token.head.i)
            for r in to_ranges(span_indexes):
                ranges.append(doc[r.start : r.stop])
            doc._.events.append(ranges)
    return doc


def recurse_children(token, blacklist=()):
    return list(helper_recurse_children(list(token.lefts) + list(token.rights), blacklist)) + [
        token
    ]


def helper_recurse_children(tokens, blacklist=()):
    for t in tokens:
        children = recurse_children(t)
        child_tags = [c.tag_ for c in children]
        has_verb_child = any(t.endswith("FIN") and t.startswith("V") for t in child_tags)
        # Skip any empty tokens
        if t.text.strip() == "":
            continue
        # Skip periods at the end of sentences (this could probably break in edge cases)
        if t.tag_ == "$.":
            continue
        # import ipdb; ipdb.set_trace()
        if t.tag_ == "$," or t.tag_ == "$.":
            direct_children = list(t.rights) + list(t.lefts)
            if len(direct_children) == 0:
                continue
            if sum(len(list(recurse_children(t))) for t in direct_children) == 0:
        # Don't recures into relative clauses
        if t.dep_ == "rc":
            continue
        # Don't recurse into modifiers with their own finite verb, they will be picked up independently
        if t.dep_ == "mo" and t.tag_.startswith("V") and t.tag_.endswith("FIN"):
            continue
        if t.dep_ == "cd" and has_verb_child:
            continue
        if t.dep_ == "cj" and has_verb_child:
            continue
        if t._.custom_dep:
            # if t._.custom_dep.lower() == "gmod" and t.tag_.startswith("V") and t.tag_.endswith("FIN"):
            #     continue
            if t.tag_.lower() == "kon" and has_verb_child:
                continue
            if t._.custom_dep.lower() == "rel" and has_verb_child:
                continue
            if t._.custom_dep.lower() == "neb" and has_verb_child:
                continue
            if t._.custom_dep.lower() == "par" and has_verb_child:
                continue
        if t not in blacklist:
            yield t
        yield from helper_recurse_children(t.lefts, blacklist=blacklist)
        yield from helper_recurse_children(t.rights, blacklist=blacklist)


def to_ranges(tokens):
    indexes = sorted(tokens)
    start = None
    end = None
    for pos, index in enumerate(indexes):
        if start is None:
            start = index
        if len(indexes) == pos + 1:
            end = index + 1
        elif indexes[pos + 1] != index + 1:
            end = index + 1
        if start is not None and end is not None:
            yield range(start, end)
            start = None
            end = None
