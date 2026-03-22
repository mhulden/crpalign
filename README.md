# crpalign — Chinese Restaurant Process Aligner

`crpalign` is a monotone 1-1 string-pair aligner trained with a Chinese Restaurant Process (CRP)-style model and Gibbs sampling.

The repository includes:

- `align.c`: reference C implementation
- `align.py`: Python ctypes wrapper over `libalign.so` (C backend)
- `crpalign.py`: Python implementation with optional epsilon-attachment postprocessing for many-many chunk output
- `example.py`: small wrapper example using `align.py`

Typical use cases:

- grapheme ↔ phoneme alignment (G2P / L2P)
- token ↔ token alignment for related strings

## Build

### C executable

```bash
gcc -O3 -Wall -Wextra -o crpalign align.c -lm
```

### Shared library for Python wrapper (`align.py`)

```bash
gcc -O3 -Wall -Wextra -fPIC -shared align.c -o libalign.so -lm
```

Notes:

- `-lm` is required for `log`, `exp`, etc.
- `-fPIC` is recommended for shared-object builds across toolchains.

## C CLI usage (`align.c`)

Basic usage:

```bash
./crpalign [options] < infile.txt > aligned.txt
```

### Input formats

- `-i l2p` (default): first two whitespace-separated fields per line are used
- `-i news`: first two tab-separated fields per line are used

Examples:

```bash
./crpalign -i l2p -o aligned < pairs.txt > aligned.out
./crpalign -i news -o aligned < tabpairs.txt > aligned.out
```

### Core options

- `-x NUM`, `--iterations=NUM`: Gibbs iterations (default `10`)
- `-b NUM`, `--burnin=NUM`: burn-in iterations (default `5`)
- `-l NUM`, `--lag=NUM`: collect global counts every `NUM` iterations after burn-in (default `1`)
- `-p NUM`, `--prior=NUM`: CRP prior (default `0.1`)
- `-m`, `--med`: run minimum-edit-distance alignment only
- `-d`, `--debug`: debug output

### Output formats

Choose with `-o FMT`:

- `aligned`: two pipe-separated lines, `_` is epsilon
- `plain`: two plain lines without `|`
- `phonetisaurus`: `IN}OUT` tokens
- `m2m`: compact one-line pair view

## Python wrapper usage (`align.py`)

`align.py` uses the C backend via ctypes and exposes `Aligner`.

### Quick example

```python
import align

wordpairs = [
    ("youth", "juθ"),
    ("yacht", "jɑt"),
]

ap = align.Aligner(
    wordpairs,
    align_symbol="_",
    iterations=50,
    burnin=1,
    lag=1,
    mode="crp",   # or "med"
)

for left, right in ap.alignedpairs:
    print(" ".join(left))
    print(" ".join(right))
    print()
```

You can also run the included example:

```bash
python3 example.py
```

### Wrapper API

```python
align.Aligner(wordpairs, align_symbol=' ', iterations=10, burnin=1, lag=1, mode='crp')
```

- `wordpairs`: iterable of `(left, right)` pairs
  - each side can be a string (character-level alignment)
  - or a token sequence (token-level alignment)
- `align_symbol`: symbol used for epsilon in output
- `mode`: `'crp'` or `'med'`
- result is in `Aligner.alignedpairs` as a list of `(left_aligned, right_aligned)` sequences

Practical notes:

- `align.py` loads `libalign.so` relative to the module location.
- The C backend has fixed-size limits; wrapper checks enforce up to 255 non-epsilon symbols and up to 255 symbols per input side.

## Python CLI with postprocessing (`crpalign.py`)

`crpalign.py` implements the same shared 1-1 aligner behavior and adds optional epsilon-attachment postprocessing for chunked many-many output.

Example:

```bash
python3 crpalign.py -i l2p -o aligned < pairs.txt > aligned.out
```

Python-only addon options:

- `--posteps`  
  Enables epsilon-attachment postprocessing. This keeps the 1-1 alignment anchors and reassigns epsilon columns to nearby anchors (or leaves them standalone) to produce chunked many-many mappings.
  
  When postprocessing is enabled:
  - output is postprocessed in whichever `-o` format you select (`aligned`, `plain`, `phonetisaurus`, `m2m`, or `chunked`)
  - base CRP training/decoding is unchanged; this is a post-alignment layer

- `--boundaries`  
  Adds forced boundary anchors (`⟨B⟩`, `⟨E⟩`) during postprocessing only. Epsilons are not allowed to attach to these boundary anchors.
  This flag has effect only when postprocessing is active (`--posteps` or `-o chunked`).
  
  Practical effect: helps differentiate initial/internal/final behavior (for example, final silent material vs non-final behavior) without changing the core 1-1 aligner.

- `--eps-maxattach` (default `3`)  
  Maximum number of epsilon symbols that may attach on each side of an anchor.
  
  Lower values:
  - more conservative chunks
  - less risk of over-merging
  - faster decoding
  
  Higher values:
  - allow larger many-many chunks
  - can improve recall for longer multi-symbol correspondences
  - increase search space and runtime

- `--eps-maxngram` (default `3`)  
  Maximum n-gram length for standalone deletion/insertion epsilon chunks.
  
  This controls both:
  - which epsilon n-grams are counted during postprocess model fitting
  - the max chunk length considered when segmenting standalone epsilon runs at decode time

- `--eps-alpha` (default `0.5`)  
  Smoothing for anchor-conditioned expansion models:
  - grapheme tuple given phoneme anchor: `P(G_tuple | p)`
  - phoneme tuple given grapheme anchor: `P(P_tuple | g)`
  
  Lower `alpha`: trusts observed anchored expansions more strongly.  
  Higher `alpha`: smoother, less peaky anchored expansion probabilities.

- `--eps-beta` (default `0.5`)  
  Smoothing for epsilon-only standalone chunk models:
  - deletion chunks by position class
  - insertion chunks by position class
  
  Lower `beta`: prefers frequently seen epsilon chunks.  
  Higher `beta`: more tolerant of unseen/rare epsilon chunks.

Recommended starting point:

```bash
python3 crpalign.py -i l2p -o aligned --posteps --boundaries \\
  --eps-maxattach 3 --eps-maxngram 3 --eps-alpha 0.5 --eps-beta 0.5 \\
  < pairs.txt > aligned.out
```

### Tuning recipes

Conservative chunking (less aggressive attachments):

```bash
python3 crpalign.py -i l2p -o chunked --posteps --boundaries \\
  --eps-maxattach 2 --eps-maxngram 2 --eps-alpha 0.7 --eps-beta 0.7 \\
  < pairs.txt > out.conservative.txt
```

Balanced default:

```bash
python3 crpalign.py -i l2p -o chunked --posteps --boundaries \\
  --eps-maxattach 3 --eps-maxngram 3 --eps-alpha 0.5 --eps-beta 0.5 \\
  < pairs.txt > out.balanced.txt
```

More permissive search space:

```bash
python3 crpalign.py -i l2p -o chunked --posteps --boundaries \\
  --eps-maxattach 4 --eps-maxngram 4 --eps-alpha 0.3 --eps-beta 0.3 \\
  < pairs.txt > out.permissive.txt
```

Note: parameter effects are data-dependent. Increasing search-space knobs does not always produce fewer chunks.

### Quick quality loop on `englishpron.txt`

Use a reproducible subset first:

```bash
head -n 400 englishpron.txt > /tmp/englishpron.400.txt
```

Run a few presets:

```bash
python3 crpalign.py -i l2p -o chunked --posteps --boundaries \\
  --eps-maxattach 2 --eps-maxngram 2 --eps-alpha 0.7 --eps-beta 0.7 \\
  < /tmp/englishpron.400.txt > /tmp/out.cons.txt

python3 crpalign.py -i l2p -o chunked --posteps --boundaries \\
  --eps-maxattach 3 --eps-maxngram 3 --eps-alpha 0.5 --eps-beta 0.5 \\
  < /tmp/englishpron.400.txt > /tmp/out.bal.txt

python3 crpalign.py -i l2p -o chunked --posteps --boundaries \\
  --eps-maxattach 4 --eps-maxngram 4 --eps-alpha 0.3 --eps-beta 0.3 \\
  < /tmp/englishpron.400.txt > /tmp/out.perm.txt
```

Compare quickly:

```bash
sed -n '1,60p' /tmp/out.cons.txt
sed -n '1,60p' /tmp/out.bal.txt
sed -n '1,60p' /tmp/out.perm.txt
```

Example sanity-check on `afterthought` (from `englishpron.txt` subset):

With `--eps-maxattach 3` or `4`, you can get:

```text
æ|f|t|ɚ |θ |ɔ   |t
a|f|t|er|th|ough|t
```

With `--eps-maxattach 2`, the same region may split:

```text
æ|f|t|ɚ |θ|ɔ   |_|t
a|f|t|er|t|houg|h|t
```

So for `...ough...`-type correspondences, `maxattach >= 3` is often a practical minimum.
`--eps-maxattach 4` still gives extra headroom for longer clusters in other words.

## Model summary

Each alignment column is a pair `(in_symbol, out_symbol)` where either side may be epsilon.

CRP-style probability for pair type `(a:b)`:

$$
P(a:b) = \frac{c(a:b)+\alpha}{N + K\alpha}
$$

with negative log costs used in dynamic programming.

Training is blocked Gibbs sampling:

1. remove current alignment counts for one pair
2. compute forward trellis (log-sum of path probabilities)
3. sample a new path by backward sampling
4. add sampled alignment counts back

After training, final output alignment is deterministic MED-style decoding under the current Gibbs-state counts.

The code also accumulates a separate global count table after burn-in; it is not used for final decoding in the current implementations.

## License

Apache License 2.0.
