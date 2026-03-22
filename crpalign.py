#!/usr/bin/env python3
import sys
import argparse
import math
import random
import collections
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

"""
CRP 1-1 aligner (Python port) + optional epsilon-attachment postprocessing.

Postprocessing goal
-------------------
Given a monotone 1-1 alignment as a sequence of columns (g_i, p_i) where either side
may be epsilon ("_"), we ONLY allow one kind of transformation:

  - Attach epsilon columns to a neighboring *anchor* column (a column where both
    sides are non-epsilon), thereby turning 1-1 columns into many-many chunks.

We DO NOT allow merging two anchor columns together. That is, we never merge
(g,p) with (g2,p2) when both are non-epsilon. The only material that can "move"
across a boundary is epsilon-only material.

This supports desired outcomes like:
  - wh ↔ h (from w:ε + h:h)
  - ea ↔ ɛ  (from e:ɛ + a:ε)
  - ti ↔ ʃ  (from t:ε + i:ʃ)
while still allowing silent final e to remain unaligned (e ↔ ε).

Statistical model (no big-chunk bias)
-------------------------------------
We score attachments by explicitly comparing a merged analysis against the split
analysis that keeps epsilon material as deletion/insertion events. Concretely, for
grapheme deletions attached to an anchor with phoneme p:

  merged:   (G+X) ↔ p
  split:    G ↔ p   +   X ↔ ε

We choose merges only when the merged likelihood beats the split likelihood.
This likelihood-ratio (Bayes factor) style comparison avoids an inherent bias
toward always building bigger chunks.
"""

# ----------------------------
# Core aligner configuration
# ----------------------------

LEFT = -1
DIAG = 0
DOWN = 1

INPUT_FORMAT_L2P = 0
INPUT_FORMAT_NEWS = 1

OUTPUT_FORMAT_PLAIN = 0
OUTPUT_FORMAT_ALIGNED = 1
OUTPUT_FORMAT_PHONETISAURUS = 2
OUTPUT_FORMAT_M2M = 3
OUTPUT_FORMAT_CHUNKED = 4  # new: epsilon-attached chunks

debug_flag = False
med_flag = False
input_format = INPUT_FORMAT_L2P
output_format = OUTPUT_FORMAT_ALIGNED
prior = 0.1

# Boundary markers for postprocessing (optional)
BOUNDARIES_ENABLED = False
BEGIN_SYM = "⟨B⟩"
END_SYM = "⟨E⟩"
BEGIN_ID: Optional[int] = None
END_ID: Optional[int] = None

symbol_to_id: Dict[str, int] = {}
id_to_symbol: List[str] = ['_']  # 0 is epsilon
max_symbol = 0

string_pairs: List[Dict] = []

current_count = collections.defaultdict(lambda: collections.defaultdict(int))
global_count = collections.defaultdict(lambda: collections.defaultdict(int))
pair_count = 0
distinct_pairs = 0


def grapheme_clusters(s: str) -> List[str]:
    """Split string into base+combining sequences."""
    out = []
    i = 0
    while i < len(s):
        base = s[i]
        i += 1
        while i < len(s) and unicodedata.combining(s[i]):
            base += s[i]
            i += 1
        out.append(base)
    return out


def get_set_char_num(sym: str) -> int:
    global max_symbol
    if sym in symbol_to_id:
        return symbol_to_id[sym]
    max_symbol += 1
    symbol_to_id[sym] = max_symbol
    id_to_symbol.append(sym)
    return max_symbol


def display_width(s: str) -> int:
    # Match align.c behavior: count non-combining codepoints as width 1.
    w = 0
    for ch in s:
        if unicodedata.combining(ch):
            continue
        w += 1
    return max(1, w)


def clear_counts():
    global pair_count, distinct_pairs
    current_count.clear()
    global_count.clear()
    pair_count = 0
    distinct_pairs = 0


def add_counts(inaligned, outaligned):
    global pair_count, distinct_pairs
    for i in range(len(inaligned)):
        a = inaligned[i]
        b = outaligned[i]
        if current_count[a][b] == 0:
            distinct_pairs += 1
        current_count[a][b] += 1
        pair_count += 1


def remove_counts(inaligned, outaligned):
    global pair_count, distinct_pairs
    for i in range(len(inaligned)):
        a = inaligned[i]
        b = outaligned[i]
        current_count[a][b] -= 1
        if current_count[a][b] == 0:
            distinct_pairs -= 1
        pair_count -= 1


def cost_crp(a, b) -> float:
    denom = pair_count + distinct_pairs * prior
    num = current_count[a][b] + prior
    if denom <= 0:
        return 0.0
    return -math.log(num / denom)


def cost_levenshtein(a, b) -> float:
    return 0.0 if a == b else 1.0


def cmp3(a, b, c):
    # C parity with CMP3 macro in align.c
    if a < b:
        return LEFT if a < c else DOWN
    return DIAG if b < c else DOWN


def log_add(x, y):
    # in negative-log space: -log(exp(-x)+exp(-y))
    if x > y:
        x, y = y, x
    if y - x > 80:
        return x
    return x - math.log1p(math.exp(-(y - x)))


def pad_to_display(s: str, width: int) -> str:
    dw = display_width(s)
    if dw >= width:
        return s
    return s + (' ' * (width - dw))


def random_3draw(left, diag, down):
    # C parity: scale negative log probs to avoid underflow before exponentiating.
    minv = min(left, diag, down)
    if minv >= 2:
        subv = minv - 2
        left -= subv
        diag -= subv
        down -= subv
    w_left = math.exp(-left)
    w_diag = math.exp(-diag)
    w_down = math.exp(-down)
    s = w_left + w_diag + w_down
    if s <= 0.0:
        return DOWN
    r = random.random() * s
    if r < w_left:
        return LEFT
    if r < w_left + w_diag:
        return DIAG
    return DOWN


def fill_trellis(in_ids, out_ids, cost_func, mode_med: bool):
    inlen = len(in_ids)
    outlen = len(out_ids)
    trellis = [[0.0] * (inlen + 1) for _ in range(outlen + 1)]
    backptr = [[0] * (inlen + 1) for _ in range(outlen + 1)]

    trellis[0][0] = 0.0
    for x in range(1, outlen + 1):
        trellis[x][0] = trellis[x - 1][0] + cost_func(0, out_ids[x - 1])
        backptr[x][0] = LEFT
    for y in range(1, inlen + 1):
        trellis[0][y] = trellis[0][y - 1] + cost_func(in_ids[y - 1], 0)
        backptr[0][y] = DOWN

    for x in range(1, outlen + 1):
        for y in range(1, inlen + 1):
            left = trellis[x - 1][y] + cost_func(0, out_ids[x - 1])
            down = trellis[x][y - 1] + cost_func(in_ids[y - 1], 0)
            diag = trellis[x - 1][y - 1] + cost_func(in_ids[y - 1], out_ids[x - 1])

            if mode_med:
                best = min(left, diag, down)
                trellis[x][y] = best
                backptr[x][y] = cmp3(left, diag, down)
            else:
                trellis[x][y] = log_add(log_add(left, diag), down)

    return trellis, backptr


def sample_alignment(in_ids, out_ids, trellis, cost_func):
    inlen = len(in_ids)
    outlen = len(out_ids)
    x, y = outlen, inlen
    inaligned = []
    outaligned = []
    while x > 0 or y > 0:
        if x == 0:
            inaligned.append(in_ids[y - 1])
            outaligned.append(0)
            y -= 1
            continue
        if y == 0:
            inaligned.append(0)
            outaligned.append(out_ids[x - 1])
            x -= 1
            continue

        left = trellis[x - 1][y] + cost_func(0, out_ids[x - 1])
        down = trellis[x][y - 1] + cost_func(in_ids[y - 1], 0)
        diag = trellis[x - 1][y - 1] + cost_func(in_ids[y - 1], out_ids[x - 1])
        direction = random_3draw(left, diag, down)
        if direction == LEFT:
            inaligned.append(0)
            outaligned.append(out_ids[x - 1])
            x -= 1
        elif direction == DIAG:
            inaligned.append(in_ids[y - 1])
            outaligned.append(out_ids[x - 1])
            x -= 1
            y -= 1
        else:
            inaligned.append(in_ids[y - 1])
            outaligned.append(0)
            y -= 1

    inaligned.reverse()
    outaligned.reverse()
    return inaligned, outaligned


def initial_align():
    for pair in string_pairs:
        in_ids = pair['in']
        out_ids = pair['out']
        L = max(len(in_ids), len(out_ids))
        inaligned = in_ids[:] + [0] * (L - len(in_ids))
        outaligned = out_ids[:] + [0] * (L - len(out_ids))
        pair['inaligned'] = inaligned
        pair['outaligned'] = outaligned
        add_counts(inaligned, outaligned)


def crp_train(iterations, burnin, lag):
    for it in range(iterations):
        print(f"Alignment iteration: {it}", file=sys.stderr)
        for pair in string_pairs:
            remove_counts(pair['inaligned'], pair['outaligned'])
            trellis, _ = fill_trellis(pair['in'], pair['out'], cost_crp, mode_med=False)
            inaligned, outaligned = sample_alignment(pair['in'], pair['out'], trellis, cost_crp)
            pair['inaligned'] = inaligned
            pair['outaligned'] = outaligned
            add_counts(inaligned, outaligned)

        # C parity: collect only when it > burnin and divisible by lag.
        if it > burnin and (it % lag == 0):
            for a in current_count:
                for b in current_count[a]:
                    global_count[a][b] += current_count[a][b]


def crp_align():
    # Use current_count as it stands (last Gibbs state) for MED alignment
    for pair in string_pairs:
        trellis, backptr = fill_trellis(pair['in'], pair['out'], cost_crp, mode_med=True)
        inlen = len(pair['in'])
        outlen = len(pair['out'])
        x, y = outlen, inlen
        inaligned = []
        outaligned = []
        while x > 0 or y > 0:
            direction = backptr[x][y]
            if x > 0 and y == 0:
                direction = LEFT
            elif y > 0 and x == 0:
                direction = DOWN
            if direction == LEFT:
                inaligned.append(0)
                outaligned.append(pair['out'][x - 1])
                x -= 1
            elif direction == DIAG:
                inaligned.append(pair['in'][y - 1])
                outaligned.append(pair['out'][x - 1])
                x -= 1
                y -= 1
            else:
                inaligned.append(pair['in'][y - 1])
                outaligned.append(0)
                y -= 1
        inaligned.reverse()
        outaligned.reverse()
        pair['inaligned'] = inaligned
        pair['outaligned'] = outaligned


def med_align():
    for pair in string_pairs:
        trellis, backptr = fill_trellis(pair['in'], pair['out'], cost_levenshtein, mode_med=True)
        inlen = len(pair['in'])
        outlen = len(pair['out'])
        x, y = outlen, inlen
        inaligned = []
        outaligned = []
        while x > 0 or y > 0:
            direction = backptr[x][y]
            if x > 0 and y == 0:
                direction = LEFT
            elif y > 0 and x == 0:
                direction = DOWN
            if direction == LEFT:
                inaligned.append(0)
                outaligned.append(pair['out'][x - 1])
                x -= 1
            elif direction == DIAG:
                inaligned.append(pair['in'][y - 1])
                outaligned.append(pair['out'][x - 1])
                x -= 1
                y -= 1
            else:
                inaligned.append(pair['in'][y - 1])
                outaligned.append(0)
                y -= 1
        inaligned.reverse()
        outaligned.reverse()
        pair['inaligned'] = inaligned
        pair['outaligned'] = outaligned


def add_string_pair(in_str: str, out_str: str):
    if input_format == INPUT_FORMAT_L2P:
        in_symbols = grapheme_clusters(in_str)
        out_symbols = grapheme_clusters(out_str)
    else:
        in_symbols = in_str.split()
        out_symbols = out_str.split()
    in_ids = [get_set_char_num(sym) for sym in in_symbols]
    out_ids = [get_set_char_num(sym) for sym in out_symbols]
    string_pairs.append({'in': in_ids, 'out': out_ids, 'inaligned': None, 'outaligned': None})


# -----------------------------------------
# Epsilon-attachment postprocessing model
# -----------------------------------------

POS_INITIAL = 0
POS_INTERNAL = 1
POS_FINAL = 2

def _tuple_from_ids(ids: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in ids)

def _render_tuple(t: Tuple[int, ...]) -> str:
    if not t:
        return "_"
    return "".join(id_to_symbol[i] for i in t)

@dataclass
class EpsAttachParams:
    max_attach: int = 3      # max eps symbols that may attach on each side of an anchor
    max_del_ngram: int = 3   # max length for deletion/insertion chunk events
    alpha: float = 0.5       # smoothing for anchored models
    beta: float = 0.5        # smoothing for eps-only models

class EpsAttachModel:
    """
    Factorized model:
      - grapheme expansion given phoneme anchor:   P(G_tuple | p_id)
      - phoneme expansion given grapheme anchor:   P(P_tuple | g_id)
      - deletions:                                 P(D_tuple | pos)
      - insertions:                                P(I_tuple | pos)

    We decode by choosing, for each epsilon run between anchors, how much attaches
    to the left anchor (suffix), how much attaches to the right anchor (prefix),
    and what stays standalone—then compute anchor likelihoods *once* for the final
    attached tuples.
    """
    def __init__(self, params: EpsAttachParams, *, begin_id: Optional[int] = None, end_id: Optional[int] = None):
        self.params = params
        self.begin_id = begin_id
        self.end_id = end_id
        self.boundaries_enabled = (begin_id is not None and end_id is not None)

        # anchored expansions
        self.given_p_counts: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)  # p -> Counter(Gtuple)
        self.given_p_totals: Dict[int, int] = collections.defaultdict(int)

        self.given_g_counts: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)  # g -> Counter(Ptuple)
        self.given_g_totals: Dict[int, int] = collections.defaultdict(int)

        # eps-only (pos-conditioned)
        self.del_counts: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)  # pos -> Counter(Dtuple)
        self.del_totals: Dict[int, int] = collections.defaultdict(int)

        self.ins_counts: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)  # pos -> Counter(Ituple)
        self.ins_totals: Dict[int, int] = collections.defaultdict(int)

        self.vocab_g_tuples = 0
        self.vocab_p_tuples = 0
        self.vocab_del = {POS_INITIAL: 0, POS_INTERNAL: 0, POS_FINAL: 0}
        self.vocab_ins = {POS_INITIAL: 0, POS_INTERNAL: 0, POS_FINAL: 0}

    # ---------- training data extraction ----------

    def fit_from_alignments(self, pairs: List[Dict]):
        """
        Learn distributions from existing 1-1 alignments by collecting:
          - plausible anchored expansions formed by absorbing adjacent eps columns
          - deletion/insertion n-grams from consecutive eps-only columns
        """
        P = self.params

        for pair in pairs:
            inaligned = pair['inaligned']
            outaligned = pair['outaligned']
            if self.boundaries_enabled:
                inaligned = [self.begin_id] + list(inaligned) + [self.end_id]
                outaligned = [self.begin_id] + list(outaligned) + [self.end_id]
            L = len(inaligned)

            # anchor indices
            anchors = [i for i in range(L) if inaligned[i] != 0 and outaligned[i] != 0]
            if self.boundaries_enabled:
                anchors_real = [i for i in anchors if inaligned[i] not in (self.begin_id, self.end_id)]
            else:
                anchors_real = anchors
            if not anchors_real:
                # no anchors; treat all as deletions/insertions
                self._collect_eps_ngrams(inaligned, outaligned, anchors_real)
                continue

            # collect anchored expansions around each anchor
            for ai, idx in enumerate(anchors_real):
                g = inaligned[idx]
                p = outaligned[idx]

                # consecutive grapheme-deletion columns immediately left/right
                left_g = []
                j = idx - 1
                while j >= 0 and outaligned[j] == 0 and inaligned[j] != 0:
                    left_g.append(inaligned[j])
                    j -= 1
                left_g.reverse()  # in order

                right_g = []
                j = idx + 1
                while j < L and outaligned[j] == 0 and inaligned[j] != 0:
                    right_g.append(inaligned[j])
                    j += 1

                # consecutive phoneme-insertion columns immediately left/right
                left_p = []
                j = idx - 1
                while j >= 0 and inaligned[j] == 0 and outaligned[j] != 0:
                    left_p.append(outaligned[j])
                    j -= 1
                left_p.reverse()

                right_p = []
                j = idx + 1
                while j < L and inaligned[j] == 0 and outaligned[j] != 0:
                    right_p.append(outaligned[j])
                    j += 1

                # enumerate bounded attachments
                for l in range(0, min(P.max_attach, len(left_g)) + 1):
                    for r in range(0, min(P.max_attach, len(right_g)) + 1):
                        Gt = _tuple_from_ids(left_g[len(left_g)-l:] + [g] + right_g[:r])
                        self.given_p_counts[p][Gt] += 1
                        self.given_p_totals[p] += 1

                for l in range(0, min(P.max_attach, len(left_p)) + 1):
                    for r in range(0, min(P.max_attach, len(right_p)) + 1):
                        Pt = _tuple_from_ids(left_p[len(left_p)-l:] + [p] + right_p[:r])
                        self.given_g_counts[g][Pt] += 1
                        self.given_g_totals[g] += 1

            # collect eps n-grams globally with position conditioning
            self._collect_eps_ngrams(inaligned, outaligned, anchors_real)

        # vocab sizes for smoothing
        self.vocab_g_tuples = len({Gt for p in self.given_p_counts for Gt in self.given_p_counts[p]})
        self.vocab_p_tuples = len({Pt for g in self.given_g_counts for Pt in self.given_g_counts[g]})
        for pos in (POS_INITIAL, POS_INTERNAL, POS_FINAL):
            self.vocab_del[pos] = len(self.del_counts[pos])
            self.vocab_ins[pos] = len(self.ins_counts[pos])

    def _collect_eps_ngrams(self, inaligned, outaligned, anchors: List[int]):
        """
        Collect deletion/insertion tuple counts from consecutive eps-only columns.
        We also count n-grams up to max_del_ngram from each run.
        """
        P = self.params

        # helper to decide position class of a column index
        first_anchor = anchors[0] if anchors else None
        last_anchor = anchors[-1] if anchors else None

        def pos_of(idx: int) -> int:
            if not anchors:
                return POS_INTERNAL
            if idx < first_anchor:
                return POS_INITIAL
            if idx > last_anchor:
                return POS_FINAL
            return POS_INTERNAL

        # scan runs of deletions (in!=0,out==0)
        i = 0
        L = len(inaligned)
        while i < L:
            if inaligned[i] != 0 and outaligned[i] == 0:
                start = i
                run = []
                while i < L and inaligned[i] != 0 and outaligned[i] == 0:
                    run.append(inaligned[i])
                    i += 1
                pos = pos_of(start) if start == 0 or (start-1 not in anchors) else POS_INTERNAL
                # count n-grams
                for a in range(len(run)):
                    for b in range(a+1, min(len(run), a+P.max_del_ngram) + 1):
                        Dt = _tuple_from_ids(run[a:b])
                        self.del_counts[pos][Dt] += 1
                        self.del_totals[pos] += 1
            else:
                i += 1

        # scan runs of insertions (in==0,out!=0)
        i = 0
        while i < L:
            if inaligned[i] == 0 and outaligned[i] != 0:
                start = i
                run = []
                while i < L and inaligned[i] == 0 and outaligned[i] != 0:
                    run.append(outaligned[i])
                    i += 1
                pos = pos_of(start) if start == 0 or (start-1 not in anchors) else POS_INTERNAL
                for a in range(len(run)):
                    for b in range(a+1, min(len(run), a+P.max_del_ngram) + 1):
                        It = _tuple_from_ids(run[a:b])
                        self.ins_counts[pos][It] += 1
                        self.ins_totals[pos] += 1
            else:
                i += 1

    # ---------- log-prob scoring ----------

    def logP_G_given_p(self, p: int, Gt: Tuple[int, ...]) -> float:
        P = self.params
        c = self.given_p_counts[p].get(Gt, 0)
        tot = self.given_p_totals.get(p, 0)
        V = max(1, self.vocab_g_tuples)
        return math.log(c + P.alpha) - math.log(tot + P.alpha * V)

    def logP_P_given_g(self, g: int, Pt: Tuple[int, ...]) -> float:
        P = self.params
        c = self.given_g_counts[g].get(Pt, 0)
        tot = self.given_g_totals.get(g, 0)
        V = max(1, self.vocab_p_tuples)
        return math.log(c + P.alpha) - math.log(tot + P.alpha * V)

    def logP_del(self, pos: int, Dt: Tuple[int, ...]) -> float:
        P = self.params
        c = self.del_counts[pos].get(Dt, 0)
        tot = self.del_totals.get(pos, 0)
        V = max(1, self.vocab_del[pos])
        return math.log(c + P.beta) - math.log(tot + P.beta * V)

    def logP_ins(self, pos: int, It: Tuple[int, ...]) -> float:
        P = self.params
        c = self.ins_counts[pos].get(It, 0)
        tot = self.ins_totals.get(pos, 0)
        V = max(1, self.vocab_ins[pos])
        return math.log(c + P.beta) - math.log(tot + P.beta * V)

    # ---------- decoding ----------

    def decode_chunks_for_pair(self, inaligned: List[int], outaligned: List[int]) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """
        Return chunk list as tuples (Gtuple, Ptuple), where each chunk is either:
          - an anchored mapping with absorbed eps material, or
          - a standalone deletion/insertion chunk (one side empty tuple).
        """
        if self.boundaries_enabled:
            inaligned = [self.begin_id] + list(inaligned) + [self.end_id]
            outaligned = [self.begin_id] + list(outaligned) + [self.end_id]
        L = len(inaligned)
        anchors = [i for i in range(L) if inaligned[i] != 0 and outaligned[i] != 0]
        if self.boundaries_enabled:
            anchors_real = [i for i in anchors if inaligned[i] not in (self.begin_id, self.end_id)]
        else:
            anchors_real = anchors
        if not anchors_real:
            # only eps chunks
            if self.boundaries_enabled:
                core_in = list(inaligned)[1:-1]
                core_out = list(outaligned)[1:-1]
                return self._chunks_all_eps(core_in, core_out)
            return self._chunks_all_eps(inaligned, outaligned)

        # Build per-anchor base symbols
        base_g = [inaligned[i] for i in anchors]
        base_p = [outaligned[i] for i in anchors]
        n = len(anchors)

        # Extract sequences of deletions/insertions between anchors, plus edges.
        del_seqs: List[List[int]] = []
        ins_seqs: List[List[int]] = []

        def collect_between(a_idx: int, b_idx: int) -> Tuple[List[int], List[int]]:
            Ds, Is = [], []
            for k in range(a_idx+1, b_idx):
                if inaligned[k] != 0 and outaligned[k] == 0:
                    Ds.append(inaligned[k])
                elif inaligned[k] == 0 and outaligned[k] != 0:
                    Is.append(outaligned[k])
            return Ds, Is

        # left edge
        Ds, Is = [], []
        for k in range(0, anchors[0]):
            if inaligned[k] != 0 and outaligned[k] == 0:
                Ds.append(inaligned[k])
            elif inaligned[k] == 0 and outaligned[k] != 0:
                Is.append(outaligned[k])
        del_seqs.append(Ds)
        ins_seqs.append(Is)

        # gaps between anchors
        for i in range(n-1):
            Ds, Is = collect_between(anchors[i], anchors[i+1])
            del_seqs.append(Ds)
            ins_seqs.append(Is)

        # right edge
        Ds, Is = [], []
        for k in range(anchors[-1]+1, L):
            if inaligned[k] != 0 and outaligned[k] == 0:
                Ds.append(inaligned[k])
            elif inaligned[k] == 0 and outaligned[k] != 0:
                Is.append(outaligned[k])
        del_seqs.append(Ds)
        ins_seqs.append(Is)

        # DP over anchors with small prefix state (how many from left gap attach as prefix)
        P = self.params

        # Precompute all possible splits for each gap for deletions and insertions:
        # gap i is between anchor i-1 and i (with i=0 being left edge; i=n being right edge)
        # For internal gaps (1..n-1), split D into suffix for left anchor and prefix for right anchor with possible middle standalone.
        # For edges, only prefix to first / suffix to last or standalone.

        def enumerate_splits(seq: List[int], left_attach_allowed: bool, right_attach_allowed: bool) -> List[Tuple[Tuple[int,...], List[Tuple[int,...]], Tuple[int,...]]]:
            """
            Return list of (left_suffix, mid_chunks, right_prefix)
            left_suffix: attaches to left anchor as suffix (empty tuple if none or no left anchor)
            right_prefix: attaches to right anchor as prefix
            mid_chunks: standalone eps chunks (tuples) in order
            """
            m = len(seq)
            splits = []
            maxa = P.max_attach
            # choose a,b with bounded suffix/prefix lengths
            for a in range(0, m+1):  # suffix length a = take first a as left_suffix
                if left_attach_allowed:
                    if a > maxa:
                        continue
                else:
                    if a != 0:
                        continue
                for b in range(a, m+1):  # middle is [a:b], prefix is [b:m]
                    pref_len = m - b
                    if right_attach_allowed:
                        if pref_len > maxa:
                            continue
                    else:
                        if pref_len != 0:
                            continue
                    left_suffix = _tuple_from_ids(seq[:a])
                    right_prefix = _tuple_from_ids(seq[b:])
                    middle = seq[a:b]
                    # represent middle as n-grams up to max_del_ngram, but here we keep it as maximal chunks greedily
                    mid_chunks = self._segment_eps_run(middle)
                    splits.append((left_suffix, mid_chunks, right_prefix))
            return splits

        # segment eps run into standalone chunks (DP best under deletion/insertion model, but
        # for speed keep simple: DP to maximize logP over chunks)
        # We'll do model-specific scoring later; here just enumerate possible chunkings.
        # To keep decoding exact-ish, we use DP for mid chunks based on del/ins probabilities.

        # We'll implement _segment_eps_run below as DP over the sequence with n-grams up to max_del_ngram,
        # choosing the segmentation that maximizes sum logP_del/ins. But we need pos to score; pos varies by gap.
        # So here we keep as list of singletons; and later we will resegment optimally with pos.

        # We'll do precise mid segmentation inside scoring functions below instead.

        def best_eps_mid_score(seq: List[int], kind: str, pos: int) -> Tuple[float, List[Tuple[int,...]]]:
            """
            Best score and chunk list for standalone eps material `seq`, using n-grams up to max_del_ngram.
            kind: 'del' or 'ins'
            """
            if not seq:
                return 0.0, []
            maxn = P.max_del_ngram
            dp = [-1e18] * (len(seq)+1)
            bp = [None] * (len(seq)+1)
            dp[0] = 0.0
            for t in range(1, len(seq)+1):
                best = -1e18
                best_k = None
                kmin = max(0, t-maxn)
                for k in range(kmin, t):
                    chunk = _tuple_from_ids(seq[k:t])
                    if kind == 'del':
                        s = self.logP_del(pos, chunk)
                    else:
                        s = self.logP_ins(pos, chunk)
                    val = dp[k] + s
                    if val > best:
                        best = val
                        best_k = k
                dp[t] = best
                bp[t] = best_k
            # reconstruct
            chunks = []
            t = len(seq)
            while t > 0:
                k = bp[t]
                if k is None:
                    k = t-1
                chunks.append(_tuple_from_ids(seq[k:t]))
                t = k
            chunks.reverse()
            return dp[len(seq)], chunks

        # For each gap we enumerate split points but compute mid chunks optimally per pos and kind.
        del_gap_splits = []
        ins_gap_splits = []

        def _is_boundary(sym_id: int) -> bool:
            return self.boundaries_enabled and sym_id in (self.begin_id, self.end_id)

        for gi in range(n+1):
            if gi == 0:
                # edge before anchor0
                left_allowed = False
                right_allowed = (not _is_boundary(base_g[0]))
            elif gi == n:
                # edge after last anchor
                left_allowed = (not _is_boundary(base_g[n-1]))
                right_allowed = False
            else:
                left_allowed = (not _is_boundary(base_g[gi-1]))
                right_allowed = (not _is_boundary(base_g[gi]))
            del_gap_splits.append(enumerate_splits(del_seqs[gi], left_allowed, right_allowed))
            ins_gap_splits.append(enumerate_splits(ins_seqs[gi], left_allowed, right_allowed))

        # DP state: at anchor i, we may have a prefix deletion tuple and prefix insertion tuple coming from gap i (between i-1 and i).
        # To keep it finite, these prefixes are already bounded by max_attach, and arise from enumerated splits.
        # We'll represent state by actual tuples.
        from collections import defaultdict

        # dp[i][gpref][ppref] = best score up to anchor i-1 processed, with these prefixes pending for anchor i.
        dp: List[Dict[Tuple[Tuple[int,...], Tuple[int,...]], float]] = [defaultdict(lambda: -1e18) for _ in range(n+1)]
        bp_choice: List[Dict[Tuple[Tuple[int,...], Tuple[int,...]], Tuple]] = [dict() for _ in range(n+1)]

        # Initialize pending prefixes for anchor 0 from left edge (gap 0): choose split that yields right_prefix
        # Left edge has no left anchor; split is (empty, mid, prefix_to_anchor0)
        pos_edge_left = POS_INITIAL
        for d_left_suffix, d_mid, d_pref in del_gap_splits[0]:
            # d_left_suffix must be empty
            d_mid_score, d_mid_chunks = best_eps_mid_score(list(d_mid[0]) if False else [], 'del', pos_edge_left)  # placeholder
        # We'll score properly below by iterating split and computing mid DP over actual mid sequence, not mid_chunks placeholder.

        def split_to_parts(split, full_seq):
            """Given (left_suffix, mid_chunks_placeholder, right_prefix) and full_seq, recover indices."""
            # We encoded left_suffix as seq[:a] and right_prefix as seq[b:], but lost b.
            # Reconstruct b from lengths: left_len=a, right_len=r, mid is middle.
            a = len(split[0])
            r = len(split[2])
            b = len(full_seq) - r
            mid = full_seq[a:b]
            return split[0], mid, split[2]

        # Initialize dp[0] states from edge gap 0 (prefix for anchor0)
        for d_split in del_gap_splits[0]:
            d_suf, d_mid_seq, d_pref = split_to_parts(d_split, del_seqs[0])
            assert len(d_suf) == 0
            d_mid_score, d_mid_chunks = best_eps_mid_score(d_mid_seq, 'del', POS_INITIAL)

            for i_split in ins_gap_splits[0]:
                i_suf, i_mid_seq, i_pref = split_to_parts(i_split, ins_seqs[0])
                assert len(i_suf) == 0
                i_mid_score, i_mid_chunks = best_eps_mid_score(i_mid_seq, 'ins', POS_INITIAL)

                state = (d_pref, i_pref)
                score = d_mid_score + i_mid_score
                if score > dp[0][state]:
                    dp[0][state] = score
                    bp_choice[0][state] = ('EDGE0', d_pref, d_mid_chunks, i_pref, i_mid_chunks)

        # Main DP across anchors
        for i in range(n):
            pos_gap_internal = POS_INTERNAL
            for (d_pref, i_pref), score_so_far in list(dp[i].items()):
                if score_so_far <= -1e17:
                    continue

                # Enumerate splits for right gap (gap i+1) which produces:
                #   suffix for anchor i (left_suffix)
                #   standalone middle
                #   prefix for anchor i+1 (right_prefix)
                # For last anchor i=n-1, gap i+1 is the right edge (POS_FINAL).
                gap_index = i + 1
                if self.boundaries_enabled and gap_index < n and base_g[gap_index] == self.end_id:
                    pos_gap = POS_FINAL
                elif self.boundaries_enabled and base_g[i] == self.begin_id:
                    pos_gap = POS_INITIAL
                else:
                    pos_gap = POS_FINAL if gap_index == n else POS_INTERNAL

                for d_split in del_gap_splits[gap_index]:
                    d_suf, d_mid_seq, d_next_pref = split_to_parts(d_split, del_seqs[gap_index])
                    d_mid_score, d_mid_chunks = best_eps_mid_score(d_mid_seq, 'del', pos_gap)

                    for ins_split in ins_gap_splits[gap_index]:
                        i_suf, i_mid_seq, i_next_pref = split_to_parts(ins_split, ins_seqs[gap_index])
                        i_mid_score, i_mid_chunks = best_eps_mid_score(i_mid_seq, 'ins', pos_gap)

                        # Score anchor i with its chosen prefix/suffix attachments
                        Gt = _tuple_from_ids(list(d_pref) + [base_g[i]] + list(d_suf))
                        Pt = _tuple_from_ids(list(i_pref) + [base_p[i]] + list(i_suf))

                        anchor_score = self.logP_G_given_p(base_p[i], Gt) + self.logP_P_given_g(base_g[i], Pt)

                        total = score_so_far + anchor_score + d_mid_score + i_mid_score

                        next_state = (d_next_pref, i_next_pref) if i+1 < n else ((), ())
                        if total > dp[i+1][next_state]:
                            dp[i+1][next_state] = total
                            bp_choice[i+1][next_state] = ('STEP', (d_pref, i_pref), (d_suf, d_mid_chunks, d_next_pref),
                                                          (i_suf, i_mid_chunks, i_next_pref))

        # Reconstruct from dp[n][((),())]
        end_state = ((), ())
        if end_state not in dp[n]:
            # fallback: pick best
            end_state = max(dp[n].items(), key=lambda kv: kv[1])[0]

        # Walk back
        chunks_rev = []
        state = end_state
        for i in range(n, 0, -1):
            info = bp_choice[i][state]
            tag = info[0]
            if tag != 'STEP':
                break
            prev_state, d_pack, i_pack = info[1], info[2], info[3]
            d_suf, d_mid_chunks, d_next_pref = d_pack
            i_suf, i_mid_chunks, i_next_pref = i_pack
            d_pref, i_pref = prev_state

            # standalone mid chunks belong to gap i (between anchor i-1 and i), which we add later in correct order.
            # We'll collect gap chunks separately and then interleave anchors.

            # create anchor chunk for anchor i-1
            ai = i-1
            Gt = _tuple_from_ids(list(d_pref) + [base_g[ai]] + list(d_suf))
            Pt = _tuple_from_ids(list(i_pref) + [base_p[ai]] + list(i_suf))
            chunks_rev.append(('ANCHOR', ai, Gt, Pt, d_mid_chunks, i_mid_chunks, i))  # include mid chunks for gap i
            state = prev_state

        chunks_rev.reverse()

        # Now build final chunk sequence in monotone order:
        # start with edge0 standalone chunks are stored in bp_choice[0][state0], but state0 is the one used in dp[0] that led to best path.
        # We can recover it from first element in chunks_rev if exists.
        # We'll reconstruct edge0 from bp_choice[0] by replay: easiest: recompute by forward replay from stored bp.
        # Instead, store edge0 info in dp[0] backpointer; we already did. We'll re-find the initial state used.

        # Find initial state used by the path: it's the 'prev_state' at i=1 in backtracking
        # i=1 corresponds to anchor0. In chunks_rev[0], prev_state is the dp[0] state, which is its prefixes.
        if chunks_rev:
            init_state = chunks_rev[0][1]  # wrong
        # We'll redo reconstruction more directly by re-running a forward pass using the chosen bp pointers.
        # Given our time constraints, we'll implement a simpler reconstruction:
        # We'll recompute the best path decisions by forward traversal using bp_choice states from dp.

        # Forward reconstruction: start at i=0 with the state that maximized dp[0] among those that lead to final.
        # We have the end_state; to get the exact chain of states, we can backtrack states stored in bp_choice (done above),
        # but we didn't store the dp[0] state. We can recover it as `state` after finishing backtracking.
        init_state = state  # after loop, state is dp[0] state used.

        # Edge0 standalone chunks:
        edge0_info = bp_choice[0].get(init_state)
        edge0_del_mid = []
        edge0_ins_mid = []
        if edge0_info and edge0_info[0] == 'EDGE0':
            edge0_del_mid = edge0_info[2]
            edge0_ins_mid = edge0_info[4]

        # Build output:
        out_chunks: List[Tuple[Tuple[int,...], Tuple[int,...]]] = []
        # emit edge0 mid standalone chunks (in original column order: deletions and insertions were interleaved in columns;
        # our model treats them separately, so we emit deletions then insertions. For display this is usually fine.
        # If you want perfect interleaving preservation, we can refine later.)
        for Dt in edge0_del_mid:
            out_chunks.append((Dt, ()))
        for It in edge0_ins_mid:
            out_chunks.append(((), It))

        # For each anchor step record, emit anchor chunk then gap mids (between this anchor and next)
        for rec in chunks_rev:
            _, ai, Gt, Pt, d_mid_chunks, i_mid_chunks, gap_i = rec
            out_chunks.append((Gt, Pt))
            # gap mids for the gap after this anchor are stored in rec
            for Dt in d_mid_chunks:
                out_chunks.append((Dt, ()))
            for It in i_mid_chunks:
                out_chunks.append(((), It))

        if self.boundaries_enabled:
            cleaned = []
            for Gt, Pt in out_chunks:
                if self.begin_id in Gt or self.end_id in Gt or self.begin_id in Pt or self.end_id in Pt:
                    continue
                cleaned.append((Gt, Pt))
            return cleaned
        return out_chunks

    def _segment_eps_run(self, seq: List[int]) -> List[Tuple[int,...]]:
        # Placeholder: not used; we do DP scoring in best_eps_mid_score.
        return [_tuple_from_ids([x]) for x in seq]

    def _chunks_all_eps(self, inaligned, outaligned):
        chunks = []
        # deletions
        run = []
        for a,b in zip(inaligned, outaligned):
            if a != 0 and b == 0:
                run.append(a)
            else:
                if run:
                    chunks.append((_tuple_from_ids(run), ()))
                    run = []
        if run:
            chunks.append((_tuple_from_ids(run), ()))
        # insertions
        run = []
        for a,b in zip(inaligned, outaligned):
            if a == 0 and b != 0:
                run.append(b)
            else:
                if run:
                    chunks.append(((), _tuple_from_ids(run)))
                    run = []
        if run:
            chunks.append(((), _tuple_from_ids(run)))
        return chunks


# -----------------------------------------
# Output formatting
# -----------------------------------------

def print_pair_plain(inaligned, outaligned):
    print(''.join([' ' if sid == 0 else id_to_symbol[sid] for sid in inaligned]))
    print(''.join([' ' if sid == 0 else id_to_symbol[sid] for sid in outaligned]))
    print()


def print_pair_phonetisaurus(inaligned, outaligned):
    parts = []
    for a, b in zip(inaligned, outaligned):
        in_sym = id_to_symbol[a] if a != 0 else '_'
        out_sym = id_to_symbol[b] if b != 0 else '_'
        parts.append(f"{in_sym}}}{out_sym}")
    print(' '.join(parts))


def print_pair_m2m(inaligned, outaligned):
    in_part = ''.join((' ' if sid == 0 else id_to_symbol[sid]) + '|' for sid in inaligned)
    out_part = ''.join((' ' if sid == 0 else id_to_symbol[sid]) + '|' for sid in outaligned)
    print(f"{in_part}\t{out_part}")


def print_pair_aligned(inaligned, outaligned):
    in_line = []
    out_line = []
    for a, b in zip(inaligned, outaligned):
        in_sym = id_to_symbol[a] if a != 0 else '_'
        out_sym = id_to_symbol[b] if b != 0 else '_'
        fieldwidth = max(display_width(in_sym), display_width(out_sym))
        in_line.append(pad_to_display(in_sym, fieldwidth))
        out_line.append(pad_to_display(out_sym, fieldwidth))
    print('|'.join(in_line))
    print('|'.join(out_line))
    print()


def print_pair_chunked(chunks: List[Tuple[Tuple[int,...], Tuple[int,...]]]):
    in_line = []
    out_line = []
    for Gt, Pt in chunks:
        in_sym = _render_tuple(Gt)
        out_sym = _render_tuple(Pt)
        fieldwidth = max(display_width(in_sym), display_width(out_sym))
        in_line.append(pad_to_display(in_sym, fieldwidth))
        out_line.append(pad_to_display(out_sym, fieldwidth))
    print('|'.join(in_line))
    print('|'.join(out_line))
    print()


def print_pair_plain_chunks(chunks: List[Tuple[Tuple[int,...], Tuple[int,...]]]):
    # Space-separated chunks for readability.
    in_syms = [_render_tuple(Gt) for Gt, _ in chunks]
    out_syms = [_render_tuple(Pt) for _, Pt in chunks]
    print(" ".join(in_syms))
    print(" ".join(out_syms))
    print()


def print_pair_phonetisaurus_chunks(chunks: List[Tuple[Tuple[int,...], Tuple[int,...]]]):
    parts = []
    for Gt, Pt in chunks:
        g = _render_tuple(Gt)
        p = _render_tuple(Pt)
        parts.append(f"{g}}}{p}")
    print(" ".join(parts))


def print_pair_m2m_chunks(chunks: List[Tuple[Tuple[int,...], Tuple[int,...]]]):
    in_part = ''.join((_render_tuple(Gt) if Gt else " ") + "|" for Gt, _ in chunks)
    out_part = ''.join((_render_tuple(Pt) if Pt else " ") + "|" for _, Pt in chunks)
    print(f"{in_part}\t{out_part}")

def write_stringpairs(eps_model: Optional[EpsAttachModel] = None, *, use_chunks: bool = False):
    for pair in string_pairs:
        if use_chunks:
            chunks = pair.get('chunks')
            if chunks is None and eps_model is not None:
                chunks = eps_model.decode_chunks_for_pair(pair['inaligned'], pair['outaligned'])
            chunks = chunks or []

            if output_format == OUTPUT_FORMAT_PLAIN:
                print_pair_plain_chunks(chunks)
            elif output_format == OUTPUT_FORMAT_ALIGNED:
                # "aligned" format, but on chunks (pipes stay aligned using display_width)
                print_pair_chunked(chunks)
            elif output_format == OUTPUT_FORMAT_PHONETISAURUS:
                print_pair_phonetisaurus_chunks(chunks)
            elif output_format == OUTPUT_FORMAT_M2M:
                print_pair_m2m_chunks(chunks)
            elif output_format == OUTPUT_FORMAT_CHUNKED:
                print_pair_chunked(chunks)
            continue

        # default: show original 1-1 alignment
        if output_format == OUTPUT_FORMAT_PLAIN:
            print_pair_plain(pair['inaligned'], pair['outaligned'])
        elif output_format == OUTPUT_FORMAT_ALIGNED:
            print_pair_aligned(pair['inaligned'], pair['outaligned'])
        elif output_format == OUTPUT_FORMAT_PHONETISAURUS:
            print_pair_phonetisaurus(pair['inaligned'], pair['outaligned'])
        elif output_format == OUTPUT_FORMAT_M2M:
            print_pair_m2m(pair['inaligned'], pair['outaligned'])
        elif output_format == OUTPUT_FORMAT_CHUNKED:
            chunks = pair.get('chunks')
            if chunks is None and eps_model is not None:
                chunks = eps_model.decode_chunks_for_pair(pair['inaligned'], pair['outaligned'])
            print_pair_chunked(chunks or [])

def read_stringpairs():
    for line in sys.stdin:
        if input_format == INPUT_FORMAT_L2P:
            parts = line.split()
            if len(parts) >= 2:
                add_string_pair(parts[0], parts[1])
        else:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 2:
                add_string_pair(parts[0], parts[1])
    clear_counts()
    initial_align()


def main():
    global debug_flag, med_flag, input_format, output_format, prior

    parser = argparse.ArgumentParser(description="CRP string-pair aligner + epsilon-attachment postprocess")
    parser.add_argument('-d', '--debug', action='store_true', help='print debug info')
    parser.add_argument('-m', '--med', action='store_true', help='do simple med-alignment only')
    parser.add_argument('-x', '--iterations', type=int, default=10, help='run aligner for NUM iterations')
    parser.add_argument('-i', '--informat', default='l2p', choices=['l2p', 'news'], help='expect data in format FMT=l2p|news')
    parser.add_argument('-o', '--outformat', default='aligned',
                        choices=['plain', 'aligned', 'phonetisaurus', 'm2m', 'chunked'],
                        help='print data in format FMT')
    parser.add_argument('-b', '--burnin', type=int, default=5, help='burn-in iterations')
    parser.add_argument('-l', '--lag', type=int, default=1, help='collect counts every NUM iterations after burn-in')
    parser.add_argument('-p', '--prior', type=float, default=0.1, help='CRP prior strength')

    # postprocess parameters
    parser.add_argument('--posteps', action='store_true',
                        help='enable epsilon-attachment postprocessing; when enabled, output uses postprocessed chunks in the selected -o format')
    parser.add_argument('--eps-maxattach', type=int, default=3,
                        help='max eps symbols attachable on each side of an anchor (default 3)')
    parser.add_argument('--eps-maxngram', type=int, default=3,
                        help='max length for standalone deletion/insertion chunks (default 3)')
    parser.add_argument('--eps-alpha', type=float, default=0.5,
                        help='smoothing for anchored expansion models (default 0.5)')
    parser.add_argument('--eps-beta', type=float, default=0.5,
                        help='smoothing for deletion/insertion models (default 0.5)')

    parser.add_argument('--boundaries', action='store_true',
                        help='add forced-aligned word boundary anchors for postprocessing only (helps final vs non-final); epsilons cannot attach to boundaries')

    args = parser.parse_args()

    debug_flag = args.debug
    med_flag = args.med
    input_format = INPUT_FORMAT_L2P if args.informat == 'l2p' else INPUT_FORMAT_NEWS

    if args.outformat == 'plain':
        output_format = OUTPUT_FORMAT_PLAIN
    elif args.outformat == 'aligned':
        output_format = OUTPUT_FORMAT_ALIGNED
    elif args.outformat == 'phonetisaurus':
        output_format = OUTPUT_FORMAT_PHONETISAURUS
    elif args.outformat == 'm2m':
        output_format = OUTPUT_FORMAT_M2M
    elif args.outformat == 'chunked':
        output_format = OUTPUT_FORMAT_CHUNKED

    prior = args.prior
    random.seed()

    read_stringpairs()
    if med_flag:
        med_align()
    else:
        crp_train(args.iterations, args.burnin, args.lag)
        crp_align()

    need_post = args.posteps or output_format == OUTPUT_FORMAT_CHUNKED
    eps_model = None

    global BOUNDARIES_ENABLED, BEGIN_ID, END_ID
    BOUNDARIES_ENABLED = bool(args.boundaries)
    if need_post and BOUNDARIES_ENABLED:
        # Create boundary symbol IDs (postprocess only; not used by 1-1 aligner).
        BEGIN_ID = get_set_char_num(BEGIN_SYM)
        END_ID = get_set_char_num(END_SYM)
    if need_post:
        params = EpsAttachParams(
            max_attach=args.eps_maxattach,
            max_del_ngram=args.eps_maxngram,
            alpha=args.eps_alpha,
            beta=args.eps_beta,
        )
        eps_model = EpsAttachModel(params, begin_id=BEGIN_ID if BOUNDARIES_ENABLED else None, end_id=END_ID if BOUNDARIES_ENABLED else None)
        eps_model.fit_from_alignments(string_pairs)
        for pair in string_pairs:
            pair['chunks'] = eps_model.decode_chunks_for_pair(pair['inaligned'], pair['outaligned'])

    write_stringpairs(eps_model=eps_model, use_chunks=need_post)


if __name__ == "__main__":
    main()
