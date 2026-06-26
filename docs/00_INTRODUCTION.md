# 00 — Introduction: What is pairwise sequence alignment?

> Goal of this page: give you just enough biology and intuition that the rest of
> the project makes sense, even if you have never touched bioinformatics.

## The problem in one sentence

Given two strings of letters, **line them up so that their similar parts sit on
top of each other**, inserting "gaps" where needed, and score how good that
line‑up is.

That's it. The strings are usually biological:

- **DNA** over the 4‑letter alphabet `A C G T`
- **RNA** over `A C G U`
- **Proteins** over the 20 amino‑acid letters

…but the algorithms don't care what the letters *mean*. They work on any
sequences. (You could align two English words with them.)

## Why biologists care

Evolution edits sequences over time by three basic operations:

- **Substitution** — one letter changes into another (`A → G`)
- **Insertion** — a letter appears
- **Deletion** — a letter disappears

(Insertions and deletions are jointly called **indels**.)

If two species share a gene inherited from a common ancestor, their versions of
that gene will be *similar but not identical* — peppered with substitutions and
indels. **Alignment reverses the guesswork**: it proposes which letters
correspond, which were substituted, and where indels happened. From a good
alignment you can measure how related two sequences are, find conserved regions
(which are often functionally important), and build evolutionary trees.

## Representing an alignment: gaps

An alignment writes the two sequences on two rows of equal length, inserting the
gap character `-` to represent an indel. Example (DNA):

```
A:  G C A T G - C U
B:  G - A T T A C A
    |   | | .   | .
```

(A gap column is left blank — that is exactly what this project's
`format_alignment()` prints, a space under any `-`.) Reading column by column:

- `G/G` — a **match** (same letter)
- `C/-` — a **gap** (a `C` in A with nothing in B: a deletion in B / insertion in A)
- `A/A`, `T/T` — matches
- `G/T` — a **mismatch** (substitution)
- `-/A` — a gap the other way
- `C/C` — match
- `U/A` — mismatch

The two rows **always have the same length**, and you can never have a gap
aligned to a gap (`-/-`) — that would be meaningless.

## Scoring: turning an alignment into a number

To choose the *best* alignment we need to measure quality. We assign each column
a score and **sum them**:

| Column type | Example | Score (this project's default) |
|---|---|---|
| match | `A/A` | `+2` |
| mismatch | `A/G` | `-1` |
| gap | `A/-` or `-/A` | `-2` |

The exact numbers are a **scoring scheme** you choose. Rewards for matches pull
the alignment toward stacking identical letters; penalties for mismatches and
gaps discourage sloppy line‑ups. The algorithms find the column layout with the
**maximum total score**.

> This project uses a **linear gap model**: every gap position costs the same
> (`gap`). Real tools usually use an **affine** model — a large cost to *open* a
> run of gaps and a small cost to *extend* it — because a single 5‑long indel is
> biologically more likely than five separate 1‑long ones. Affine gaps are noted
> throughout as an extension exercise.

## Two flavours: global vs local

There are two fundamentally different questions you might ask, and they give the
two algorithms this project implements:

### Global — Needleman–Wunsch
*"Align the **entire length** of both sequences, end to end."*

Use it when the two sequences are roughly the same thing overall — e.g. the same
gene in two closely related species. Every letter of both sequences must appear
in the alignment (possibly against a gap).

### Local — Smith–Waterman
*"Find the single **best‑scoring sub‑region** the two sequences share, and
ignore everything else."*

Use it when you suspect only a *part* is shared — e.g. a short conserved motif
buried inside two otherwise‑different sequences, or matching a small query
against a huge genome. The unrelated flanks are simply dropped.

The same example sequences can give very different answers under the two modes —
global is forced to stretch across the whole length, local zooms in on the best
island of similarity. The demo shows both.

## Where the GPU comes in

Computing the optimal alignment is done with **dynamic programming**: fill a 2‑D
table where each cell is the best score for a pair of prefixes, building larger
answers from smaller ones. That table has `(m+1) × (n+1)` cells for sequences of
length `m` and `n`, and filling it is `O(m·n)` work — which gets expensive fast,
and is begging to be parallelised.

The catch (and the fun): each cell depends on its neighbours, so you can't
naively compute all cells at once. **How to extract parallelism from that
dependency is exactly what the CUDA parts of this project teach.** Continue to:

- [01 — Algorithms](01_ALGORITHMS.md): the precise recurrences and worked examples.
- [02 — CUDA parallelisation](02_CUDA_PARALLELIZATION.md): how we make it parallel.
