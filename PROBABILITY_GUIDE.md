# The predictor's probability math, from the ground up

A reading guide. No prior probability is assumed. Every formula is first
shown as plain counting on a tiny example you can check by hand, and only
then written as a formula. The code being explained is
`src/kvcache/predictor.py`; the reasoning is in `PREDICTOR.md`.

---

## 0. The one-paragraph version

When the cache is full we must throw something out. We want to throw out the
conversation least likely to be used again. To guess that, we look at the
past: among earlier conversations in the same situation as this one, what
fraction came back soon? That fraction is our probability. Everything below
is careful bookkeeping of "the same situation", and of one trap: the longer
someone has been silent, the more the situation changes.

---

## 1. Probability as a fraction of past cases

If 30 of 100 past cases did X, we say "X has probability 0.30". Nothing
more. This is called an **empirical** probability: it comes from counting,
not from a theory.

Everything in `predictor.py` is of this kind. There is no equation for how
humans behave; there are tables of counts.

---

## 2. What we record about the past

For every past conversation and every turn *k* in it, we write down what
happened right after that turn:

- **ENDED**: the conversation never continued, or
- **GAP = g seconds**: the next turn arrived g seconds later.

Cases are grouped by how many turns the conversation had so far (1, 2, 3,
4-5, 6-9, 10+), because a conversation that has already gone on for ten turns
behaves differently from one that just started.

### A toy table

Take one group, "2 turns so far". Suppose we saw 20 such cases:

- 6 ENDED
- 14 continued, with these gaps in seconds, sorted:

```
20  30  40  45  60  60  90  100  120  150  200  300  600  1500
```

Two numbers summarise it:

- `p_end = 6 / 20 = 0.30`: the share that ended.
- the list of 14 gaps: what the continuers did.

---

## 3. "What is the chance it returns within x seconds?"

Just count. Out of all 20 cases, how many returned within x seconds of the
last message?

- within 60 s: the gaps 20, 30, 40, 45, 60, 60 are all at most 60, so 6 cases.
  That is 6 / 20 = **0.30**.
- within 300 s: 12 gaps are at most 300, so 12 / 20 = **0.60**.
- within 1,800 s: all 14 gaps, so 14 / 20 = **0.70**.

Call this `F(x)`: the fraction of all cases that returned within x seconds.

```
F(60) = 0.30      F(300) = 0.60      F(1800) = 0.70
```

Notice `F` never reaches 1. The 6 cases that ended never return, so it stops
at 0.70. That is exactly `1 - p_end`.

### Where the code's formula comes from

The code stores the ended share and the gaps separately, so it computes `F`
in two pieces:

```
F(x) = (1 - p_end)  x  (fraction of the CONTINUERS whose gap <= x)
```

Check with x = 60: the continuers are 14 of 20, so 1 - p_end = 0.70. Of the
14 continuers, 6 had a gap of at most 60 s, so that fraction is 6/14 = 0.4286.
Then 0.70 x 0.4286 = **0.30**. Same answer as counting directly.

Why does the formula say "multiply"? Returning within x seconds needs two
things to be true together: (1) the conversation continues at all, and (2)
its gap is at most x. When you want "both A and B", and B is measured only
among those where A holds, you multiply:

```
P(A and B) = P(A) x P(B, among those where A holds)
```

That is the whole derivation. It is the chain rule of probability, and here
it is only the fractions 14/20 and 6/14 multiplying to 6/20.

The split is not just cosmetic. ENDED cases have no gap, so they cannot sit in
the list of gaps. Keeping them as a separate number is what makes the next
section work.

---

## 4. The trap: someone has already been silent for a while

At eviction time we are not looking at a conversation that just finished. It
finished some time ago and has been **idle** for `a` seconds. We already know
it did not return during those `a` seconds. That knowledge changes the odds.

Rule of thumb: **throw away every past case that contradicts what you already
know, then count again among the rest.**

We want: "given still silent after `a` seconds, chance of returning in the
next `H` seconds". `H` is the window that matters for the cache (see section
7).

### Counting version

Toy table, idle `a = 60` s, window `H = 240` s (so we look at 60 to 300 s):

1. Cases still silent at 60 s: everyone except those whose gap was at most
   60. That is 20 - 6 = **14** cases (the 6 ended ones, plus 8 continuers
   with gaps 90, 100, 120, 150, 200, 300, 600, 1500).
2. Of those 14, how many returned in the window (60, 300]? The gaps 90, 100,
   120, 150, 200, 300: **6** cases.
3. Answer: 6 / 14 = **0.4286**.

### Formula version

```
P(return within H | idle a) = ( F(a + H) - F(a) )  /  ( 1 - F(a) )
```

- `F(a + H) - F(a)` = fraction of ALL cases returning in the window
  = 0.60 - 0.30 = 0.30 (that is 6 of the 20).
- `1 - F(a)` = fraction of ALL cases still silent at `a`
  = 1 - 0.30 = 0.70 (that is 14 of the 20).
- Ratio: 0.30 / 0.70 = 0.4286. Same as counting.

The division is exactly "throw away the cases that already returned". It is
called **conditioning**. That is the only idea in this section.

Note that `1 - F(a)` includes the ENDED cases. From the outside you cannot
tell "ended forever" from "just slow", so both stay in the pool of the still
silent. This is why keeping `p_end` separate matters: `F` correctly stops
below 1.

### The idle time changes the answer

Same window length (240 s here, or 300 s in the second row), different idle:

| Idle so far | Cases still silent | Return in window | Probability |
|---|---|---|---|
| 0 s (just finished), window 300 s | 20 | 12 | 0.60 |
| 60 s, window (60, 300] | 14 | 6 | 0.43 |
| 300 s, window (300, 600] | 8 | 1 (gap 600) | 0.125 |
| 1,800 s, window (1800, 2100] | 6 | 0 | 0.00 |

The longer the silence, the lower the chance. In the last row only the
ENDED cases are left, and they never come back.

### Why silence is informative here (and not always)

This effect exists because human gaps are **heavy-tailed**: many quick
replies, and a few very long ones. If someone has not replied after a long
wait, they are probably one of the ones who left.

If gaps were **memoryless** (a fixed chance each second, like radioactive
decay), the probability would be the same at every idle time, and idle time
would tell you nothing. Real chat traffic is not like that, which is why this
predictor works better than a constant guess. It also means that on traffic
whose arrival times are generated by a formula, this whole idea gives
nothing.

---

## 5. The second factor: will the next prompt still fit?

A returning conversation only reuses its cache if its next prompt still
starts with the cached text. If the conversation has grown past the model's
context limit, old turns are dropped, the start of the prompt changes, and
the cache no longer matches.

The next prompt is roughly: cached length + the user's next message. It fits
if that stays under the limit. We do not know the next message length, so we
use past follow-up message lengths, again by counting.

### Toy example

Past follow-up message lengths (tokens), sorted:

```
10  20  30  50  80  120  200  400  900  3000
```

Context limit 4,096; this conversation has 3,900 tokens cached. Room left:
4,096 - 3,900 = 196 tokens. How many past follow-ups were at most 196 tokens?
6 of 10 (10, 20, 30, 50, 80, 120). So

```
P(fits) = 6 / 10 = 0.60
```

For a short conversation (say 500 tokens cached, room 3,596) nearly every
past message fits, and the probability is close to 1. For a full
conversation the room is 0 and the probability is 0.

---

## 6. Putting the two together

```
value = P(returns within H | idle a, turns so far)  x  P(next prompt fits)
```

With the toy numbers from above: 0.4286 x 0.60 = **0.257**.

Multiplying assumes the two events do not influence each other: whether
someone comes back soon says nothing about how long their message will be.
That is an approximation, and a reasonable one to start with. The code
does not test it.

### Several conversations sharing one cached prefix

A leaf of the cache tree can belong to more than one conversation (a shared
system prompt belongs to all of them). It is worth keeping if **at least one**
of them comes back and reuses it. Easiest to compute as the opposite: it is
wasted only if **none** comes back.

```
value(leaf) = 1 - (1 - v1) x (1 - v2) x ... x (1 - vn)
```

Example: two owners with values 0.30 and 0.10. Chance neither returns:
0.70 x 0.90 = 0.63. So the leaf's value is 1 - 0.63 = **0.37**. A prefix
shared by hundreds of conversations ends up with a value close to 1, so it
is almost never evicted.

This also assumes the owners return independently of each other.

---

## 7. Where `H` (the window) comes from

Sections 3 and 4 needed a window length `H` ("returns within H seconds").
Why not a fixed number like 5 minutes? Because the right window depends on
how crowded the cache is.

**The reasoning.** Keeping a conversation's cache only pays off if it comes
back *while the cache could still plausibly have held it*. How long the
cache can hold anything depends on load:

- Cache holds 1,000 conversations' worth and a new one arrives every
  second: an idle conversation survives about 1,000 s even under plain LRU.
- Same cache, but a new one arrives every 10 ms: it survives about 10 s.

If a user returns after 60 s and the cache turns over every 10 s, nothing
we could have done at eviction time would have saved that entry (we would
have needed to hold it through 6 turnovers, at the price of many others). So
the return that counts is one within roughly the cache's turnover time.

**How the code measures the turnover time.** LRU evicts the least recently
touched leaf first, so the leaf that has been untouched the longest is the
next to go, and its age tells us how long things have been surviving:

```
H = now - (last-touched time of the least recently used cached leaf)
```

Heavy load: things get evicted quickly, so the oldest leaf is young, `H` is
small. Plenty of memory: the oldest leaf is old, `H` is large. Worked
example: it is t = 1,000 s, the leaves were last touched at t = 700, 850 and
990; `H = 1000 - 700 = 300 s`.

**Effect on the number.** A small `H` makes `P(return within H)` smaller
for every conversation. The same `H` is used for all of them within one
eviction decision.

**Be honest about what this is.** `H` is a design choice, a heuristic, not
something derived or fitted. It is cheap (no extra state), adapts to load
without a tuning knob, and it worked in our runs, but we have not
compared it with alternatives (a fixed 5 minutes, a multiple of the
oldest age, a tuned value). That comparison is an open ablation. Note also
that the estimate is noisy: one very old, unlucky leaf sets `H` for the
whole eviction decision.

---

## 8. From a value to an eviction

The cache is a tree; only leaves can be evicted. Compute `value` for each
leaf, evict the lowest first, and when a leaf goes its parent may become a
leaf and a candidate. Ties are broken by "least recently used". Stock
Preble ranks leaves by last-touched time; this ranks them by `value`.

---

## 9. Checking that the probabilities are any good

A number like 0.4286 is only useful if it is right. We test it on
conversations the tables never saw (later days).

**Calibration.** Collect all the cases where the predictor said "about 40%".
About 40% of them should really have returned. Do this for bins from 0 to
100% and compare "predicted" to "observed". On WildChat, predicted 0.148 was
observed at 0.147, and predicted 0.390 at 0.411, so the probabilities can be
read as probabilities.

**Brier score.** For each case, take (prediction - what happened)^2, where
what happened is 1 if it returned and 0 if not, then average. Lower is
better. Always predicting the overall average gives a baseline (0.199 on
WildChat); the predictor scores 0.160.

**AUC.** Pick a random case that did return and a random one that did not.
AUC is the chance the predictor gave the returner the higher number. 0.5 is
a coin flip, 1.0 is perfect. The predictor gets 0.786 on WildChat. This
tests ranking only, and ranking is what eviction needs.

---

## 10. What is learned, what is assumed

| Piece | Learned from data, or assumed? |
|---|---|
| `p_end`, the lists of gaps, the list of follow-up lengths | Learned: pure counts from the training trace |
| Turn groups (1, 2, 3, 4-5, 6-9, 10+) | Chosen by hand |
| The split into "ends / continues", conditioning on idle time, multiplying the two factors, combining owners | Probability rules (the structure is general) |
| Return and fit are independent; owners are independent | Assumed, not tested |
| `H` = age of the oldest cached leaf | A design choice |

Because the numbers are counted from a trace, they describe that trace. To
use the predictor on other traffic, rebuild the tables from that traffic (a
matter of seconds). Whether tables from one dataset also work on another is
an experiment, not a given; see `PREDICTOR.md` for what has been measured.

---

## 11. Glossary

- **Empirical probability**: a fraction counted from past data.
- **Conditioning**: recomputing a probability after discarding the cases that
  contradict what you now know.
- **ECDF** (empirical cumulative distribution function): "the fraction of
  observed values that are at most x". Sorting the list and counting.
- **Heavy tail**: a few values are far larger than the typical one (a few
  very long silences).
- **Memoryless**: the chance of something happening next does not depend on
  how long you have already waited.
- **Calibrated**: predictions match observed frequencies.
- **Horizon (`H`)**: the length of the "returns soon" window.
- **Leaf / owner**: a leaf is a cache-tree node with no children; its owners
  are the conversations whose prompts passed through it.
