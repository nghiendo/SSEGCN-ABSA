# Research Gap and Proposed Improvements for SG-MBSC-ABSA

## 1. Research Gap

The current SG-MBSC-ABSA project is already stronger than a plain SSEGCN-style design in one important way: it does not stop at syntax-semantic graph enhancement. Instead, it adds a structured sentiment reasoning head with:

- one shared branch
- three sentiment-specific expert branches
- gated shared-private mixing
- branch supervision
- prototype-based contrastive learning
- a residual base classifier path

That means the project already addresses an important practical limitation in ABSA systems: even if the graph encoder captures useful context, the model still needs a mechanism to route information differently for positive, neutral, and negative sentiment.

However, when we compare the current project against the logical target of surpassing both SSEGCN and DAGF, three research gaps remain.

### Gap 1: Expert specialization is supervised, but not explicitly de-redundant

The current model encourages class alignment through branch cross-entropy and prototype contrastive loss. This is useful, but it does not directly guarantee that:

- `pos` learns something different from `neu`
- `neu` learns something different from `neg`
- `neg` learns something different from `pos`

In other words, the current project teaches each expert where to move, but it does not strongly penalize experts for collapsing toward similar representations.

Logical issue:

- Current implication: `class-aligned supervision -> better specialization`
- Missing implication: `class-aligned supervision -> non-redundant specialization`

This gap matters because a multi-expert design only becomes fully convincing when the experts are both:

- correct for their own class
- meaningfully different from one another

### Gap 2: Fusion is adaptive in the middle, but still relatively static at the output

The current project already has adaptive fusion at the representation level through gating. That is good. But the final prediction stage still follows a relatively simple formula:

- `shared_logits + expert_logits + base_weight * base_logits`

This means the model does not yet explicitly learn, for each input sample, questions such as:

- when should the shared branch dominate?
- when should expert reasoning dominate?
- when should the residual base path dominate?

Logical issue:

- Current implication: `gated representation fusion -> context sensitivity`
- Missing implication: `context sensitivity -> optimal final decision fusion`

For a web-dev analogy, this is like having smart internal middleware routing, but the final response merger still behaves like a mostly fixed aggregator.

### Gap 3: The encoder is aspect-aware, but not yet an explicit multi-graph fusion engine

The current graph encoder is already better than a single rigid syntax graph. It uses:

- semantic attention
- aspect-conditioned scoring
- short-distance priors
- iterative edge refinement

But from a DAGF-style perspective, this is still mostly one evolving adjacency space, not a clearly separated fusion of multiple named graph sources.

Logical issue:

- Current implication: `one refined aspect-aware graph -> better relation modeling`
- Missing implication: `multiple complementary graph sources -> broader relation coverage`

This matters because DAGF-like arguments become strong exactly when we can say:

- one graph captures syntactic structure
- one graph captures semantic correlation
- one graph captures aspect-centric or sentiment-prior relations
- the model adaptively decides which graph source matters more in each case

At the moment, SG-MBSC is strong in head-level sentiment reasoning, but not yet equally strong in explicit graph-source diversity.

## 2. Logical Positioning of the Current Project

The current project is already logically strong in the following chain:

1. syntax-only is not enough
2. semantic-only is not enough
3. aspect-aware graph reasoning is needed
4. shared reasoning alone is not enough for polarity separation
5. sentiment-specific expert routing is needed

This makes the current model stronger than a plain graph-enhancement architecture.

But to move from "strong" to "hard to beat," the next logical chain should be:

1. aspect-aware graph reasoning is needed
2. sentiment-specific experts are needed
3. experts must be explicitly non-redundant
4. multiple graph sources should be fused adaptively
5. final prediction paths should also be fused adaptively

That is the exact bridge from the current project to a more defensible claim of surpassing SSEGCN and competing with or surpassing DAGF.

## 3. Proposed Improvements

### Improvement A: Add expert diversity regularization

#### Goal

Ensure that `pos`, `neu`, and `neg` experts do not learn near-duplicate representations.

#### Why this is logically needed

Current supervision already says:

- the correct expert should align with the correct class

But we also need to say:

- incorrect experts should not behave like weak copies of the correct one

#### Practical solution options

1. Orthogonality loss between expert representations
2. Pairwise cosine similarity penalty between expert states
3. Diversity penalty on expert projection weights
4. Routing disagreement regularization across experts

#### Recommended first version

Use a pairwise cosine penalty on gated or pre-gated expert states:

- penalize high similarity between `pos` and `neu`
- penalize high similarity between `pos` and `neg`
- penalize high similarity between `neu` and `neg`

This is the cleanest first step because it is:

- lightweight
- local to the SG head
- easy to ablate

### Improvement B: Add adaptive output fusion

#### Goal

Replace the current mostly fixed final logit addition with sample-wise learned fusion.

#### Why this is logically needed

The current project already learns sample-specific gating in intermediate representations. It is logically consistent to extend that idea to the final decision stage.

Some samples may need:

- more shared evidence
- more expert evidence
- more base residual evidence

A single fixed output merge is weaker than a learned input-dependent output merge.

#### Practical solution

Learn a small fusion network that outputs normalized weights:

- `alpha(x)` for shared logits
- `beta(x)` for expert logits
- `gamma(x)` for base logits

with:

- `alpha(x) + beta(x) + gamma(x) = 1`

Then compute:

`final_logits = alpha(x)*shared_logits + beta(x)*expert_logits + gamma(x)*base_logits`

#### Benefit

This improves the model logically because it upgrades the final decision from:

- fixed addition

to:

- context-aware decision composition

### Improvement C: Add explicit multi-graph construction and graph-source fusion

#### Goal

Move the encoder closer to a true DAGF-level argument while preserving the current SG-MBSC head advantages.

#### Why this is logically needed

Right now, the project is strongest in sentiment-specialized reasoning after graph encoding.
To compete directly with DAGF-style claims, it should also become stronger in graph-source diversity before head reasoning.

#### Candidate graph sources

1. Syntactic graph
   - derived from dependency or short-distance priors
2. Semantic graph
   - derived from self-attention or learned token correlation
3. Aspect-centric graph
   - derived from aspect-token affinity
4. Sentiment prior graph
   - derived from class-biased or expert-conditioned relevance

#### Fusion design

Use adaptive graph-source weighting per sample, or even per token pair.

A simple first version:

- build several adjacency tensors
- encode each graph branch separately or partially separately
- learn graph weights based on aspect-conditioned context
- fuse them before SG head routing

#### Benefit

This strengthens the argument against DAGF by making the project strong in both places:

- graph construction
- sentiment-specific decision routing

## 4. Recommended Implementation Order

### Phase 1: Fastest high-value gain

1. Add expert diversity regularization
2. Add adaptive output fusion

Reason:

- local changes
- lower engineering cost
- easy to test through ablation
- directly aligned with current SG head architecture

### Phase 2: Stronger claim against DAGF

3. Add explicit multi-graph construction and graph-source fusion

Reason:

- this is the larger architectural step
- it is the most important change if the thesis claim is "stronger than DAGF"

## 5. Suggested Thesis Claim

A clean thesis-style claim for the current project plus next-step improvements is:

"The current SG-MBSC-ABSA architecture already improves over syntax-semantic graph enhancement alone by introducing aspect-aware sentiment-specific routing through shared-private expert branches. To further strengthen the model beyond SSEGCN and toward or beyond DAGF, the next necessary steps are explicit expert de-redundancy, adaptive output fusion, and explicit multi-graph source fusion."

## 6. Final Recommendation

If we want the next version of this project to be logically stronger in a defensible way, the most important upgrade path is:

1. make experts different, not just class-aligned
2. make final fusion adaptive, not mostly fixed
3. make graph sources explicit and complementary, not mostly implicit in one refined adjacency

That sequence is practical, aligned with the current codebase, and strong enough to support a real research argument rather than a cosmetic architectural change.
