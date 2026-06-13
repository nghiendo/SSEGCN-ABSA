# Logic Audit Table for SG-MBSC-ABSA

## Scope

This audit table consolidates the logical questions we need for comparing the current project against SSEGCN and DAGF, then maps each question to:

- the answer already present in the project
- concrete code evidence
- the remaining action needed, if any

Reference files used in this audit:

- `SG_MBSC_COMPONENT_ARCHITECTURE.mmd`
- `models/sg_mbsc_absa.py`
- `models/sg_mbsc_encoder.py`
- `models/sg_mbsc_bert_encoder.py`
- `data_utils.py`
- `train.py`

## Audit Table

| # | Logical question | Answer in current project | Code evidence | Required action / gap |
|---|---|---|---|---|
| 1 | Does the graph model sentiment relations better than syntax-only or semantic-only reasoning? | Yes, partially. The encoder builds attention-based adjacency conditioned by aspect information and short-distance priors, so relations are not purely dependency-based or purely semantic. | `models/sg_mbsc_encoder.py:81-86`, `models/sg_mbsc_encoder.py:123-142`; BERT path: `models/sg_mbsc_bert_encoder.py:82-86`, `models/sg_mbsc_bert_encoder.py:115-133` | Explicitly describe this as an aspect-conditioned relation graph. If we want to surpass DAGF more convincingly, add true multi-graph construction rather than relying mainly on one refined graph. |
| 2 | Does the model reduce over-reliance on a rigid parser? | Yes, to some extent. It uses learned attention scores and short-mask priors instead of a single hard parser graph. | `models/sg_mbsc_encoder.py:125-139`; `data_utils.py:253-261` builds multi-level `short_mask` | Add ablations: no `short_mask`, no `aspect_scores`, no iterative edge refinement. This is needed to prove the parser-softening claim empirically. |
| 3 | Is graph reasoning aspect-specific? | Yes. Aspect information enters both edge construction and the base residual representation. | `models/sg_mbsc_encoder.py:81-86`; `models/sg_mbsc_bert_encoder.py:82-86`; `models/sg_mbsc_absa.py:159-164`; `models/sg_mbsc_absa.py:176-180` | Keep this as a central claim. Add qualitative case studies or attention/edge visualization for different aspects in the same sentence. |
| 4 | Does the architecture filter noisy or irrelevant signals instead of propagating every edge equally? | Yes. This is one of the strongest parts of the project. Shared-private gating lets each sentiment branch mix only the amount of branch-specific information it needs. | `models/sg_mbsc_absa.py:70-75`; diagram flow in `SG_MBSC_COMPONENT_ARCHITECTURE.mmd` lines 67-78 | Frame the gate explicitly as a relevance filter, not just a fusion block. Future extension: add sparse or confidence-aware gating. |
| 5 | Does the project separate shared knowledge from sentiment-specific knowledge? | Yes. The head explicitly separates one shared branch and three sentiment-specific expert branches: `pos`, `neu`, `neg`. | `models/sg_mbsc_absa.py:22-36`; diagram lines 24-41 in `SG_MBSC_COMPONENT_ARCHITECTURE.mmd` | Keep as a key methodological contribution. To strengthen it further, add anti-redundancy constraints between experts. |
| 6 | Are the sentiment experts actually supervised to become different, rather than just split architecturally? | Yes, partially. The project uses branch supervision and prototype-based contrastive loss to push class-aligned experts. | `models/sg_mbsc_absa.py:93-110`, `models/sg_mbsc_absa.py:118-124`, `models/sg_mbsc_absa.py:126-134`; diagram lines 93-108 | Add explicit expert diversity regularization such as orthogonality, pairwise cosine penalties, or mutual-information reduction. Current supervision is class-aligned but not fully anti-redundant. |
| 7 | Is fusion adaptive to the input context? | Yes, mainly at representation level. Gates are sample-dependent, and shared plus expert states are merged into a joint embedding before classification. | `models/sg_mbsc_absa.py:73-80`; diagram lines 80-85 | Output-level fusion is still relatively static because logits are added directly. Add sample-wise adaptive weights for shared, expert, and base logits. |
| 8 | Is there a residual/base prediction path to avoid full dependence on expert branches? | Yes. The model keeps an aspect-pooled residual base classifier path. | `models/sg_mbsc_absa.py:84-85`; diagram lines 17-18, 41-42, 87-91 | Change the global `sg_base_weight` into a learned or sample-dependent coefficient. That would make the residual path more logically adaptive. |
| 9 | Does the model use locality control or distance priors to suppress long noisy relations? | Yes. `short_mask` is constructed with multiple short-distance levels and injected into attention scoring. | `data_utils.py:240-261`; `models/sg_mbsc_encoder.py:138-139`; `models/sg_mbsc_bert_encoder.py:128-129` | Turn this from fixed heuristic bins into learned distance bias or relation-aware distance encoding. |
| 10 | Is the SG head truly multi-branch reasoning, not just an ordinary classifier on top of pooled features? | Yes. Each branch has its own projection, attention pooling, gate, and expert path. | `models/sg_mbsc_absa.py:25-48`, `models/sg_mbsc_absa.py:65-87`; diagram lines 22-42 | Keep this as evidence that the head performs structured sentiment reasoning. A future step is top-k or temperature-based routing. |
| 11 | Does the project improve over SSEGCN in sentiment-specific reasoning? | Yes. SSEGCN is strong at graph enhancement, but this project adds sentiment-specialized experts, gates, branch CE, and prototype contrastive learning. | `models/sg_mbsc_absa.py:22-134`; architecture diagram SG head block | Support the claim with ablations: remove gate, remove branch CE, remove prototype loss, remove base residual. |
| 12 | Has the project already surpassed DAGF in multi-graph fusion? | Not yet in a strict sense. The current encoder mainly refines one adjacency tensor through layers rather than explicitly constructing and fusing several named graphs. | `models/sg_mbsc_encoder.py:86-107`; `models/sg_mbsc_bert_encoder.py:86-107` | Add explicit multi-graph construction, e.g. syntactic graph, semantic graph, aspect-centric graph, sentiment prior graph, followed by adaptive graph-source fusion. |
| 13 | Is there still a risk of redundancy among the three experts? | Yes. The current contrastive mechanism pulls each selected expert toward its prototype, but it does not directly force expert-to-expert diversity. | `models/sg_mbsc_absa.py:126-134` | Add diversity loss: orthogonality, pairwise cosine repulsion, disagreement regularization, or decorrelated routing. |
| 14 | Is the final fusion stage still somewhat static? | Yes. Final logits are computed by direct addition of shared logits, expert logits, and optionally weighted base logits. | `models/sg_mbsc_absa.py:81-85` | Add adaptive output fusion: `alpha(x)*shared + beta(x)*expert + gamma(x)*base`, with normalized weights per sample. |
| 15 | What is the strongest logical justification of the current project? | The project first extracts aspect-aware graph features, then routes them through shared-private sentiment gates, and finally supervises sentiment-specific experts using branch loss and prototype contrastive loss. | Combined evidence: `models/sg_mbsc_encoder.py:81-86`, `models/sg_mbsc_encoder.py:123-142`, `models/sg_mbsc_absa.py:70-87`, `models/sg_mbsc_absa.py:93-134`, diagram lines 57-108 | Use this as the central thesis statement. Then position the remaining gaps as: anti-redundancy, adaptive output fusion, and explicit multi-graph fusion. |

## Practical Summary

The current project already answers three major logical questions well:

1. aspect-specific reasoning
2. relevance filtering
3. sentiment-specific expert learning

The current project answers two others only partially:

1. adaptive fusion at the final output layer
2. explicit multi-graph fusion comparable to DAGF

## Priority Actions

### High priority

- Add adaptive output fusion
- Add diversity regularization among `pos`, `neu`, `neg` experts

### Medium priority

- Run clean ablations for:
  - no gate
  - no branch CE
  - no prototype contrastive loss
  - no `short_mask`
  - no aspect-conditioned attention term

### High priority if the goal is to compete directly with DAGF

- Add explicit multi-graph construction and graph-source fusion, for example:
  - syntactic graph
  - semantic graph
  - aspect-centric graph
  - sentiment prior graph
