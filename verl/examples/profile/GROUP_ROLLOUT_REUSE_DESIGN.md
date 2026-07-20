# Group-Wide Redundancy Elimination for GRPO Rollouts

Status: design boundary for the current research line. This document separates
the intended paper contribution from workload-specific prototypes and optional
optimizations.

## 1. Research objective

Within one GRPO group, branches and turns repeatedly encounter identical or
overlapping external context. The system should reuse every context-derived
computation whose dependencies are provably reusable, compute only the missing
delta for partial overlap, skip work that is provably unnecessary, and locally
recompute anything that depends on branch-specific causal history.

Context includes:

- original images, screenshots, crops, refocus outputs, and search images;
- fetched documents and document chunks;
- deterministic tool calls and their returned artifacts;
- context-derived work during acquisition, preprocessing, encoding, prefill,
  and decode.

The paper is not centered on relevance planning or fixed top-k pruning. Its
central question is:

> Given a context-derived computation in a GRPO rollout, which dependencies are
> shared across branches or turns, at what granularity can its result be reused,
> and which work must remain branch-local?

Ordinary exact text-prefix reuse already provided by the SGLang prefix cache is
a baseline, not a paper contribution.

## 2. Computation-node model

The unit of reasoning is a context-derived computation node, not merely a file,
request, image slot, or turn number. A node is an operator applied to inputs and
produces a representation at a specific stage:

```text
node = operator(inputs, model/processor version, operator configuration)
```

For every node, the runtime chooses one execution action:

| Action | Required evidence | Execution |
|---|---|---|
| `EXACT` | Equivalent dependency signatures | Reuse the complete artifact |
| `PARTIAL` | An explicit overlap map and reusable dependency closure | Reuse matched units and compute the delta |
| `SKIP` | Static or runtime evidence that the branch will not consume the work | Do not execute or access the node |
| `LOCAL` | Branch-history dependence, failed validation, or no safe match | Recompute normally in the recipient |

`LOCAL` is the correctness fallback. `SKIP` is redundancy elimination but is
not reuse. Metrics and paper claims must report them separately.

Similarity is a candidate-generation signal, not sufficient proof of reusable
state. An approximate `PARTIAL` action additionally needs a defined mapping,
representation stage, error criterion, and fallback.

## 3. Dependency-aware identity

`content_id` identifies canonical content or a canonical subunit. It is useful
for lookup but does not by itself establish computation equivalence.

The authoritative key is a dependency signature:

```text
dependency_signature = H(
    operator_id,
    representation_stage,
    immutable operator inputs,
    content/subunit identities,
    coordinate or chunk mapping,
    model_version,
    processor_version,
    operator_config,
    external-state version when applicable,
)
```

The signature must omit rollout-local placement only when the operator is truly
placement-independent. It must include causal history whenever the produced
state depends on that history.

Examples:

- A final ViT embedding depends on decoded image content, resize/crop and grid
  construction, processor configuration, vision weights, and output stage. It
  does not depend on where the image is later placed in the LLM prompt.
- A document tokenizer result depends on canonical text, tokenizer version, and
  tokenization configuration, but not on its later absolute LLM position.
- A decoder-layer K/V block depends on its input hidden state, layer weights,
  position semantics, and causal history. Equal document text is insufficient.
- A tool result depends on tool identity, arguments, tool version, external
  state or snapshot, and side-effect semantics.

## 4. Reusable artifact

Completed reusable work is represented as a `ReuseArtifact`:

```text
ReuseArtifact
  operator_id
  representation_stage
  content_id
  dependency_signature
  dependency_scope
  coordinate_or_chunk_map
  value

ProducerMetadata
  group_id
  policy_epoch
  branch_id
  turn_id

SharingPolicy
  validity_scope
```

Field roles:

- `dependency_signature` answers whether two executions are mathematically
  equivalent. Model/processor versions and operator configuration belong here
  whenever they affect the value.
- `group_id` and `policy_epoch` select a sharing/lifetime namespace. They do not
  make otherwise identical computations mathematically different.
- Producer branch and turn are provenance for coordination and attribution;
  they never alter artifact identity.
- `representation_stage` distinguishes raw content, processed features, patch
  embeddings, encoder layers, final encoder output, tokenization, document
  embeddings, and any decoder-side stage.
- `dependency_scope` records whether the result is content-local,
  coordinate-local, window-local, common-prefix-dependent, or branch-local.
- `coordinate_or_chunk_map` connects canonical subunits to a recipient image,
  document, or tool artifact.
- `validity_scope` defines group, policy, model, processor, external snapshot,
  and optional TTL boundaries.

The value may be a tensor, token ids, parsed document, chunk metadata, tool
artifact, or another typed representation. Different modalities do not need a
common tensor format.

## 5. Stage-by-stage safety boundary

### 5.1 Images and visual tool outputs

Exact reuse:

- The same processed image under the same processor and vision weights can
  reuse its full ViT output across branches, turns, and later prompt positions.
- Recipient position ids, mRoPE, LLM hidden states, and decoder K/V remain local.

Partial reuse:

- Original/crop/refocus pairs first need an image-space or processor-grid overlap
  map.
- Patch-embedding reuse requires equal processed patch inputs and compatible
  coordinate semantics; raw pixel overlap alone is not enough after resize or
  resampling.
- Reuse inside a local/window attention block is safe only while the matched
  token's dependency closure remains inside a matched region.
- After global mixing, an unchanged-looking token may depend on changed image
  regions. It must become `LOCAL` unless equivalence or a bounded approximation
  is established.

This dependency expansion across vision layers is part of the research problem,
not an implementation nuisance. It determines the safe reuse stage and explains
why activation similarity alone cannot authorize deep reuse.

### 5.2 Documents and document chunks

Document reuse must be reported by representation stage. Identical or overlapping
search content does not imply that every downstream LLM state is reusable.

| Stage | Exact/partial reuse opportunity | Remaining work |
|---|---|---|
| Search/fetch | Reuse the same deterministic result or page snapshot | Refresh only outside validity/TTL |
| Parse/normalize | Reuse canonical text and structural extraction | Reparse changed pages |
| Chunk | Exact chunk hashes; overlap map for shared spans | Construct branch-local ordering |
| Tokenize | Reuse token ids for identical canonical chunks | Generate local positions and separators |
| Independent document encoder | Reuse complete or chunk embeddings | Materialize selected chunks locally |
| Causal LLM prefill | Existing common-prefix reuse where applicable | Recompute under divergent branch history |

For the current decoder-only VLM, document fetch, parsing, chunking, and
tokenization are straightforward reusable stages. Large GPU prefill savings from
arbitrary-position document reuse require either common-prefix equivalence or a
history-independent document/context encoder. Deep document K/V must not be
claimed reusable solely from equal text.

The first document implementation should therefore instrument each stage
separately. It should not aggregate CPU/tool savings and LLM prefill savings into
one ambiguous "document reuse" number.

Granularity must also be explicit. If pages A and B partially overlap but one
canonical chunk is byte/text identical, the page-level relation is `PARTIAL`,
the matched chunk executes with `EXACT`, and unmatched chunks execute `LOCAL`.
`PARTIAL` execution is reserved for a computation node whose own input is only
partially reusable and therefore needs an overlap map plus delta computation.

### 5.3 Deterministic tools

A tool execution can be `EXACT` only when its result is reusable under the
declared validity scope:

- the tool is deterministic or tied to an immutable external snapshot;
- its arguments and configuration match;
- it is free of side effects, or execution reuse has explicitly safe semantics;
- authorization and user/session boundaries permit sharing.

Otherwise only a returned immutable artifact may be content-addressed, or the
tool call remains `LOCAL`. The serialized observation still receives
recipient-local positions and causal contextualization.

### 5.4 Prefill and decode

Different causal histories change deep hidden states and usually K/V. RoPE phase
adjustment can repair a K position phase but cannot repair a changed hidden state
or V. Direct deep-KV graft across divergent histories is therefore `LOCAL` by
default.

Decode remains in scope through rigorously separated mechanisms:

- Exact K/V storage or computation sharing requires equivalent dependency
  signatures, commonly an exact shared prefix or an explicitly
  history-independent context-memory representation.
- Block selection or sparse attention is a `SKIP` action: it avoids K/V access
  and attention work but does not make non-equivalent K/V reusable.
- Physical sharing is valid only for numerically equivalent immutable blocks.
- Content equality alone cannot deduplicate standard decoder self-attention K/V
  after branch histories diverge.

Dynamic sparse execution must demonstrate net end-to-end benefit after page-table
maintenance, scheduling, kernel, and batching overhead. The original physical-slot
search path was negative; the current semantic-position/fused implementation has a
positive real-scale Geo3K rollout result (Section 8), but remains an approximate
`SKIP` policy whose task-quality boundary still needs validation.

## 6. Runtime boundary

The first library-level runtime contains only proven common mechanisms. It is
intentionally smaller than the eventual layout:

```text
rollout_reuse/
  artifact.py       # identity, scope, action, provenance
  registry.py       # exact lookup, invalidation, single-flight, accounting
  routing.py        # keep a group inside one artifact owner
```

Matcher, materialization, and modality adapters are added only when an actual
second execution path needs them. SGLang remains the GPU artifact backend; the
agent registry does not duplicate ViT tensors.

The control contract is shared; modality-specific matching and representation
operators remain adapters.

`ContextBank` may name the artifact registry. `ContextView` may name the
recipient-local materialization. A `ContextPlan` is an optional producer of
specific `SKIP` decisions and is not the organizing abstraction of the paper or
runtime.

Do not build a large modality-independent relevance planner before at least two
workloads demonstrate a shared, useful selection rule. Fixed top-k is a workload
policy and must not be presented as a reuse algorithm.

## 7. Non-goals for the first version

- Confidence-based branch dropping or killing low-probability GRPO rollouts.
- Changing GRPO sampling, reward distributions, or relative-advantage semantics.
- A learned speculative branch controller.
- Treating semantic similarity alone as permission to reuse hidden state.
- Claiming deep decoder K/V equivalence from content identity.
- Folding ordinary text prefix-cache hits into the contribution.
- Adding unrelated micro-optimizations to make one workload look faster.

Speculation may later pre-execute deterministic tool calls or preprocessing, but
it is outside the first paper and must not suppress exploration branches.

## 8. Current implementation status

Implemented and aligned with the design:

- No custom ViT cache remains. The per-item/similarity/window/token-sparse ViT
  experiments were deleted after consistently losing to the unmodified encoder.
- LLM-prefill CacheBlend retains similarity/kvdev selection and donor K/V grafting;
  recipient-local prompt order, positions, causal work, and mRoPE remain local.
- Cache invalidation after model-weight updates, memory release, and explicit
  cache flush.
- Sparse decode resolves semantic drop positions once at recipient prefill, caches
  stable batch plans/buffers, and fuses mask gating plus page-table compaction in one
  prewarmed Triton kernel. Production CUDA Graph replay writes fixed captured buffers;
  stable batches append only the new causal tail. The current direct-source kernel
  reads live `req_to_token` rows without first gathering a dense intermediate. It does
  not claim K/V equivalence; it is a `SKIP` action.
- MMSearch reward and rollout attribution needed to check quality.

Integrated exact-reuse control path:

- Agent-owned MMSearch retrieval and observation tokenization use the common
  dependency-aware registry and failure-safe single-flight.
- GRPO `uid`, policy step, true rollout branch number, turn, and per-image
  content ids propagate toward the execution backend.
- Group-preserving agent-worker routing and SGLang replica affinity keep
  concurrent branches inside the owner of their reusable artifacts.
- MMSearch reports retrieval/tokenization `EXACT` separately from the
  already-materialized context `SKIP` action.
- SGLang records server/worker PID, physical replica, CacheBlend reuse, and
  sparse-context attribution without enabling a custom ViT tensor cache.

Prototype-only mechanisms:

- Fixed top5-to-top2 is a temporary `SKIP` experiment. It is not reuse and is not
  a main-paper default.
- MMSearch currently uses deterministic synthetic search operators; real fetch,
  parse, and chunk stages still need adapters with external snapshot semantics.

Not implemented:

- Dependency-aware partial visual reuse for crop/refocus overlap.
- Real-document fetch/parse/chunk artifacts and chunk-overlap matching.
- A common deterministic-tool adapter and validity policy.
- A unified `EXACT/PARTIAL/SKIP/LOCAL` decision trace.

Current sparse-decode evidence (2026-07-19/20):

- Geo3K stress, Qwen2.5-VL-7B, two replicas, real `64×4`, two complete PPO steps;
  custom ViT cache off, identical kvdev prefill, `fast_apply=0`,
  `compact_prefill=0`, and about 28.2–28.4% of decode context dropped.
- CUDA Graph dense controls averaged `127.797s` rollout over the two-run ABBA
  controls, about 3.6% faster than the earlier forced-eager dense mean. Formal
  launchers therefore use CUDA Graph rather than measuring an artificial eager path.
- Sparse without an absolute profitability floor averaged `129.622s` rollout
  (1.43% slower than graph dense). Long/high-batch decoder forwards were nevertheless
  2.1–2.7% faster; batch=3 tails were slightly slower and erased the kernel win.
- A 4096 aggregate-dropped-token floor produced `68.991s + 59.016s = 128.007s`,
  only 0.16% slower than the graph dense mean and 1.25% faster than the prior sparse
  mean. This removes the demonstrated regression but is not end-to-end positive.
- The next implementation removes the per-token dense page-table gather by fusing
  request-row lookup with compact/append. It has passed CPU/static regression but has
  no formal GPU number yet: the single planned `64×4` run was stopped during Ray init
  when physical GPUs 1/2 became externally saturated, before rollout began.
- Complete PPO time remains effectively flat because actor/logprob/update work is
  unchanged and noisier than the decoder saving. Do not claim whole-loop speedup.
- Output-token means are essentially matched, but sparse attention is approximate
  and this stress run has no useful reward discrimination. Do not promote it as a
  quality-preserving default until a task-quality A/B passes.

## 9. Implementation sequence

1. Validate sparse decode quality on a reward-bearing workload, then repeat the
   real `64×4` paired rollout measurement; keep the policy default-off until both
   performance and quality pass.
2. Add document/tool acquisition artifacts to MMSearch and report fetch, parse,
   chunk, tokenize, and LLM-prefill costs separately.
3. Demonstrate the same exact-reuse lifecycle in at least one non-MMSearch
   workload, such as repeated MMDU images or OSWorld screenshots.
4. Implement chunk-level document overlap and coordinate-aware visual overlap.
5. Keep visual `PARTIAL` as future research rather than restoring the deleted ViT
   cache implementation.
6. Keep fixed top-k and decode sparse execution as isolated policies. Promote
   either only when it adds reproducible end-to-end value without unacceptable
   GRPO quality/semantics changes.

## 10. Existing implementation mapping

| Existing implementation | Preservation/integration | Common contract | Required change |
|---|---|---|---|
| Custom per-item ViT cache | Removed after consistently negative end-to-end results | none | do not re-enable in launch configs |
| LLM prefill CacheBlend | Preserved as the merged-token compute-reuse backend | donor K/V + request/group scope | provenance and cross-branch attribution |
| Missing-image batching/order reconstruction | Upstream behavior preserved unchanged | `LOCAL` materialization | no custom tensor store |
| Weight-update/flush invalidation | Preserved | policy/model validity boundary | policy-scoped cache keys as fail-safe |
| MMSearch retrieval cache | Existing computation wrapped | tool-result artifact | dependency signature + single-flight |
| Observation tokenization cache | Existing computation wrapped | token-id artifact | tokenizer/template dependencies + single-flight |
| Already-seen context suppression | Preserved behavior | `SKIP` | stop reporting it as reuse |
| Fixed top-k | Optional only | `SKIP` policy | disabled in exact-reuse default |
| Reward scorer | Preserved unchanged | quality attribution | no reuse dependency |
| Sparse decode | Kept as an isolated approximate `SKIP` policy | semantic drop positions + fused page-table compaction | default-off; quality gate and repeat paired A/B |

## 11. Required attribution

Every experiment must attribute results by action, modality, and representation
stage. Minimum counters include:

- exact, partial, skip, local counts and eligible units;
- bytes/tokens/patches/chunks reused and recomputed;
- operator time avoided versus lookup, matching, copying, and scheduling cost;
- prefill tokens, decode attention blocks, memory, batching, and E2E wall time;
- fallback reason and representation stage;
- task reward/accuracy and, for approximate reuse, an explicit error measure.

The paper may claim reuse benefit only for `EXACT` and validated `PARTIAL`
actions. `SKIP` benefit must be reported separately. Cross-date measurements
must not be used as strict single-variable attribution without a current-tree
paired control.
