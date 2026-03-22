import random
import unittest

import crpalign as cp


def reset_crpalign_state() -> None:
    cp.symbol_to_id.clear()
    cp.id_to_symbol[:] = ["_"]
    cp.max_symbol = 0

    cp.string_pairs.clear()
    cp.current_count.clear()
    cp.global_count.clear()
    cp.pair_count = 0
    cp.distinct_pairs = 0

    cp.input_format = cp.INPUT_FORMAT_L2P
    cp.prior = 0.1

    cp.BOUNDARIES_ENABLED = False
    cp.BEGIN_ID = None
    cp.END_ID = None


def load_sample_pairs(limit: int = 80):
    pairs = []
    with open("englishpron.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # File format: PRON WORD
                pairs.append((parts[1], parts[0]))
            if len(pairs) >= limit:
                break
    return pairs


def train_base_aligner(pairs, *, seed: int = 7) -> None:
    for graphemes, phonemes in pairs:
        cp.add_string_pair(graphemes, phonemes)

    cp.clear_counts()
    cp.initial_align()

    random.seed(seed)
    cp.random.seed(seed)

    # Keep this short so tests run quickly while still producing non-trivial alignments.
    cp.crp_train(iterations=4, burnin=1, lag=1)
    cp.crp_align()


def assert_chunk_invariants(testcase: unittest.TestCase, model: cp.EpsAttachModel) -> None:
    for pair in cp.string_pairs:
        inaligned = pair["inaligned"]
        outaligned = pair["outaligned"]
        chunks = model.decode_chunks_for_pair(inaligned, outaligned)

        # Invariant 1: non-epsilon symbol order is preserved exactly.
        in_cols = [x for x in inaligned if x != 0]
        out_cols = [x for x in outaligned if x != 0]
        in_chunks = [x for gt, _ in chunks for x in gt]
        out_chunks = [x for _, pt in chunks for x in pt]
        testcase.assertEqual(in_cols, in_chunks)
        testcase.assertEqual(out_cols, out_chunks)

        # Invariant 2: number of anchor columns is preserved (no anchor-anchor merges).
        anchor_cols = sum(1 for a, b in zip(inaligned, outaligned) if a != 0 and b != 0)
        anchor_chunks = sum(1 for gt, pt in chunks if gt and pt)
        testcase.assertEqual(anchor_cols, anchor_chunks)

        # Invariant 3: never emit a fully empty chunk.
        testcase.assertTrue(all(gt or pt for gt, pt in chunks))


class TestEpsilonAttachment(unittest.TestCase):
    def setUp(self) -> None:
        reset_crpalign_state()

    def test_chunk_invariants_without_boundaries(self) -> None:
        pairs = load_sample_pairs(limit=80)
        train_base_aligner(pairs, seed=7)

        model = cp.EpsAttachModel(cp.EpsAttachParams(max_attach=3, max_del_ngram=3, alpha=0.5, beta=0.5))
        model.fit_from_alignments(cp.string_pairs)

        assert_chunk_invariants(self, model)

    def test_chunk_invariants_with_boundaries(self) -> None:
        pairs = load_sample_pairs(limit=80)
        train_base_aligner(pairs, seed=11)

        begin_id = cp.get_set_char_num(cp.BEGIN_SYM)
        end_id = cp.get_set_char_num(cp.END_SYM)
        model = cp.EpsAttachModel(
            cp.EpsAttachParams(max_attach=3, max_del_ngram=3, alpha=0.5, beta=0.5),
            begin_id=begin_id,
            end_id=end_id,
        )
        model.fit_from_alignments(cp.string_pairs)

        assert_chunk_invariants(self, model)

        # Boundary anchors are internal to decoding and must not appear in output chunks.
        for pair in cp.string_pairs:
            chunks = model.decode_chunks_for_pair(pair["inaligned"], pair["outaligned"])
            for gt, pt in chunks:
                self.assertNotIn(begin_id, gt)
                self.assertNotIn(end_id, gt)
                self.assertNotIn(begin_id, pt)
                self.assertNotIn(end_id, pt)


if __name__ == "__main__":
    unittest.main()
