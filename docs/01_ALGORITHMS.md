# 01 — The Algorithms: recurrences, initialisation, traceback

This page is the math reference. It mirrors exactly what
[`src/cpu_reference.cpp`](../src/cpu_reference.cpp) implements, so read them
together.

## Notation

- `A` and `B` are the two sequences, of lengths `m` and `n`.
- `A[0..m-1]`, `B[0..n-1]` are their characters (**0‑indexed**).
- `H` is the DP score matrix with **`(m+1)` rows and `(n+1)` columns**. The extra
  row 0 and column 0 represent the *empty prefix*.
- `H[i][j]` = the best score of aligning the first `i` letters of `A` with the
  first `j` letters of `B`.
- The scoring scheme: `match` (≥0), `mismatch` (≤0), `gap` (≤0), and
  `sub(x, y) = match if x == y else mismatch`.

> **The ±1 index shift.** Matrix cell `(i, j)` compares `A`'s `i`‑th letter,
> which is `A[i-1]`, with `B`'s `j`‑th letter `B[j-1]`. This off‑by‑one between
> *matrix coordinates* and *string indices* is the #1 source of bugs — watch for
> the `-1` everywhere a letter is read.

## The core recurrence (shared by both algorithms)

Any optimal alignment of the prefixes `A[0..i)` and `B[0..j)` ends in exactly one
of three ways. Each corresponds to a neighbouring cell:

```
        H[i-1][j-1]   H[i-1][j]
              \           |
               \          |  (up: A[i-1] vs gap)
   (diagonal:   \         |
    A[i-1] vs    \        v
    B[j-1])       \---->  H[i][j]
                  ^
                  |  (left: gap vs B[j-1])
        H[i][j-1]
```

So:

```
diag = H[i-1][j-1] + sub(A[i-1], B[j-1])     # align the two letters
up   = H[i-1][j]   + gap                      # A[i-1] aligns to a gap in B
left = H[i][j-1]   + gap                      # gap in A aligns to B[j-1]
```

This works because of **optimal substructure**: whichever final move is best, the
remaining alignment (to its upper‑left) is itself optimal for the smaller
prefixes — and we have *already computed and stored* its score.

## Needleman–Wunsch (global)

**Initialisation** (the borders). Aligning `i` letters against an *empty*
sequence forces `i` gaps:

```
H[0][0] = 0
H[i][0] = i * gap          for i = 1..m      (left column)
H[0][j] = j * gap          for j = 1..n      (top row)
```

**Recurrence:**

```
H[i][j] = max(diag, up, left)
```

**Answer:** `H[m][n]` (bottom‑right). A global alignment must consume both whole
sequences, so the answer is always in the corner.

**Traceback:** start at `(m, n)`, repeatedly step to whichever neighbour produced
the current value, emitting a letter/gap each step, until you reach `(0, 0)`.

### Worked example (matches the demo)

`A = GCATGCU`, `B = GATTACA`, with `match=+2, mismatch=-1, gap=-2`:

```
        *   G   A   T   T   A   C   A
   *    0  -2  -4  -6  -8 -10 -12 -14
   G   -2   2   0  -2  -4  -6  -8 -10
   C   -4   0   1  -1  -3  -5  -4  -6
   A   -6  -2   2   0  -2  -1  -3  -2
   T   -8  -4   0   4   2   0  -2  -4
   G  -10  -6  -2   2   3   1  -1  -3
   C  -12  -8  -4   0   1   2   3   1
   U  -14 -10  -6  -2  -1   0   1   2
```

The bottom‑right cell is `2` → optimal global score **2**. Tracing back yields:

```
GCATGCU
|..|.|.
GATTACA
```

(3 matches `+6`, 4 mismatches `-4`, netting `2`; the match bar `|..|.|.` above
shows exactly those 3 `|` matches and 4 `.` mismatches, and the algorithm
guarantees no other global layout beats it.)

## Smith–Waterman (local)

Two changes to Needleman–Wunsch — that's all.

**Initialisation:** the entire first row and column are **0** (a local alignment
may begin anywhere, for free):

```
H[i][0] = 0      for all i
H[0][j] = 0      for all j
```

**Recurrence:** add a fourth option worth `0` ("start a fresh local alignment
here"), which means the score can never go negative:

```
H[i][j] = max(0, diag, up, left)
```

**Answer:** the **maximum value anywhere** in the matrix (not the corner).

**Traceback:** start at that maximum cell and follow predecessors **until you hit
a cell whose value is 0** — that zero marks where the local alignment began.

### Worked example (matches the demo)

`A = TTTGTTGACAAA`, `B = CCGTTGACGG`, local, same scoring:

```
        *   C   C   G   T   T   G   A   C   G   G
   *    0   0   0   0   0   0   0   0   0   0   0
   T    0   0   0   0   2   2   0   0   0   0   0
   T    0   0   0   0   2   4   2   0   0   0   0
   T    0   0   0   0   2   4   3   1   0   0   0
   G    0   0   0   2   0   2   6   4   2   2   2
   T    0   0   0   0   4   2   4   5   3   1   1
   T    0   0   0   0   2   6   4   3   4   2   0
   G    0   0   0   2   0   4   8   6   4   6   4
   A    0   0   0   0   1   2   6  10   8   6   5
   C    0   2   2   0   0   0   4   8  12  10   8
   A    0   0   1   1   0   0   2   6  10  11   9
   A    0   0   0   0   0   0   0   4   8   9  10
   A    0   0   0   0   0   0   0   2   6   7   8
```

The maximum is `12`. Tracing back from it to the nearest `0` recovers exactly the
shared 6‑letter motif:

```
GTTGAC
||||||
GTTGAC
```

Note how the unrelated flanks (`TTT…AAA` vs `CC…GG`) are simply discarded — that
is the essence of *local* alignment.

## Complexity

| | Time | Space |
|---|---|---|
| Fill the matrix | `O(m·n)` | `O(m·n)` if you keep it for traceback |
| Score only | `O(m·n)` | `O(min(m, n))` with two rolling rows |
| Traceback | `O(m + n)` | — |

The score‑only / rolling‑rows trick keeps only the previous and current rows,
because cell `(i, j)` only ever reads row `i-1` and the cell to its left.
[`cpu_score_only`](../src/cpu_reference.cpp) puts the **shorter** sequence on the
columns, so its memory is `O(min(m, n))`. The GPU batched kernel uses the same
rolling‑row idea but sizes its rows to the B‑sequence length (`O(n)` per thread),
because a whole batch shares one fixed orientation. Either way you get the score
but lose the history needed to trace back — perfect for *"which database
sequences score highly?"*, after which you re‑align just the winners in full.

## Tie‑breaking

When several moves give the same optimal value, *which* alignment you report is a
matter of policy. This project always prefers **diagonal → up → left**. Both the
CPU and GPU tracebacks use the identical rule (in fact the *same function*), so
they always return the same one of the possibly‑many optimal alignments.

Next: [02 — CUDA parallelisation](02_CUDA_PARALLELIZATION.md).
