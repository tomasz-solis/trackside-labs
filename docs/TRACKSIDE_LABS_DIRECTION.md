# Trackside Labs — Product Direction & Operating Charter

> **Investigate the unexpected. Quantify what established mechanisms can explain. Show what the evidence changes about the next prediction.**

**Status:** Accepted product direction and engineering guardrails  
**Updated:** 24 October 2026  
**Purpose:** A durable decision record for the creator and coding agents. This is not a sprint plan, marketing brief, feature wish list, or commitment to publish on every race weekend.

---

## 0. The compact version

Trackside Labs is an **autonomous, visual F1 performance-intelligence laboratory**. It combines two interests:

- **Performance understanding (~60% of the intellectual emphasis):** detect meaningful deviations from expected performance; investigate where teams and drivers gain and sacrifice time; distinguish observations, plausible mechanisms and unknowable details.
- **Forward-looking forecasting (~40%):** maintain qualifying, sprint and race predictions; test whether new evidence changes the next expectation; measure whether that change helped.

The percentages express *interest and product identity*, not quotas for development time, screen area or published findings.

**Existing features are preserved:** qualifying, sprint and race predictions, constructor comparisons, and the accuracy tracker continue to exist and remain fully functional. They are **secondary in navigation and editorial emphasis**, not second-class technically. The existing frozen prediction history and evaluation logic must survive the redesign.

**A serious web redesign is part of this direction.** Keep the Python modelling, simulation, ingestion and research stack; replace the Streamlit public interface with a modern, responsive, themable TypeScript frontend backed by a read-only Python API and versioned, precomputed artifacts. The new frontend must enable expressive visual storytelling, accessible chart interactions and indexable pages without requiring routine human work.

The project has **two complementary loops**:

- **Human discovery → reusable knowledge:** the creator notices an unfamiliar pattern, investigates it, designs a revealing visual, tests an analytical method and optionally promotes it for unattended reuse. Discovery itself remains exploratory and fun; turning every experiment into a feature is not required.
- **Autonomous knowledge → new questions:** approved models analyse familiar mechanisms and their conditional impacts at every eligible race weekend. Unusual residuals may appear publicly as carefully bounded **Trackside investigations**, while genuinely novel possible explanations go to a private, optional research inbox.

A known mechanism is **not a guaranteed explanation of a particular case**. Safety Car pit-loss effects, tyre degradation and temperature responses all depend on conditions, uncertain parameters and data coverage; approved models must represent those limits.

The distinctive product experience is an **evidence-to-expectation loop**:

**What we expected → What could change our mind → What the new evidence shows → Where the trade-off appears → What changes next → What actually happened.**

The loop runs automatically for ordinary weekends. **Exceptional findings become optional research opportunities; they never become mandatory race-weekend tasks.**

The creator designs the questions, analytical methods, causal hypotheses and visual stories. Software repeatedly executes approved methods. **Verified observations can be public even when their cause remains unknown**; an unverified *causal explanation* cannot. No AI-generated public articles, no content calendar, no required human sign-off on routine output. Private machine-assisted research suggestions may be used with clear attribution and verification requirements.

**Quality standard:** An audience of one is enough. The website must remain personally interesting, scientifically defensible, visually compelling and operationally quiet. A visitor should leave with **something specific to watch for**, not merely another ranking. The creator should be able to open a prepared research case and start testing a new idea, not spend the evening cleaning data.

---

## 1. Identity and the kind of differentiation we are pursuing

### The promise

**See where the conventional picture of F1 performance may be incomplete—and see the evidence before the outcome is known.**

This is neither a generic telemetry viewer nor primarily a leaderboard of predicted finishing positions. A visitor should quickly understand:

1. **Expectation:** What was likely *before* new evidence arrived?
2. **Deviation:** What differs from that expectation, relative to a fair comparison?
3. **Trade-off:** Where was performance gained and lost? Is there a net advantage?
4. **Strength of evidence:** Is the finding trustworthy, consequential and new?
5. **Pressure test:** What plausible future evidence or condition would most change our expectation?
6. **Implication:** What should we expect or watch differently next, if anything?
7. **Resolution:** Did the subsequent observations support the expectation?

The product must make these questions legible through **visual comparisons**, not a wall of explanatory text. The audience experience is **anticipation → exploration → resolution**: a question to carry into the next session, a chart they can explore, and an eventual outcome. Ordinary results should remain valuable even when no surprise is detected.

### What we learn from others—and what we do not imitate

[F1 Strategy](https://f1-strategy.com/) is a useful example of rigorous, contextual, testable analysis. Other F1 products already offer telemetry, driver/car profiles, automated insights, prediction models and accuracy tracking. We should assume these are **table stakes, not unique selling points**.

Trackside Labs does **not** compete on the largest collection of dashboards, the most detailed pit-stop simulator, race-by-race editorial volume, live pit-wall coverage or the boldest prediction headline.

Its attempted distinction is the **repeatable, visual, pre-registered link between surprise, measured performance trade-offs, forward implications and eventual evaluation**. In particular:

- The **gap from a recorded pre-session expectation** is the starting point, not an interesting-looking chart discovered afterward.
- A **gain and its corresponding cost** appear together; a single favourable sector is not evidence of an overall advantage.
- The system records **whether new evidence truly changed a forecast**, rather than inventing a narrative after the event.
- A **surprise has a public research life cycle**: observed, checked, optionally published **Under Investigation**, tested, supported, weakened or closed unresolved. The status changes with evidence; a promised article or human approval is not required.
- Forecasts include **what might change the expectation**, derived from sensitivity analysis or predeclared scenario tests—not a retrospective explanation invented once a result is known.
- Visitors can **investigate one consequential trade-off themselves**, using approved comparable data and an explicitly defined interpretation rather than an unlimited telemetry playground.
- If nothing passes the evidence bar, **the site remains useful without manufacturing a surprise**.

This is a direction to test, **not a claim that no other website does anything similar**. The F1-flavoured **“Under Investigation”** label is a useful invitation, not a moat. The potential differentiation is the **public, time-stamped, visually testable lifecycle of uncertain ideas**, including rejected and inconclusive ones, alongside a reliable autonomous explanation of established mechanisms. Differentiation must be demonstrated in the finished experience.

### Audience and success

The primary user is the creator: an F1 enthusiast who enjoys discovering signals, predictive modelling, statistical reasoning and visual data storytelling. The secondary audience is analytically curious F1 fans.

Success is measured by **insight quality, forecast learning, clarity of the visual evidence, reader comprehension, autonomy and sustained curiosity**. Traffic, SEO, followers and occasional social posts do not control the research roadmap; however, basic indexability, deep links and shareable graphics should be automated technical capabilities rather than manual promotion work.

---

## 2. Two independent layers: routine intelligence and exceptional research

### A. Routine intelligence — always available when data permits

This is the dependable website. It runs every supported weekend **without creator intervention**:

- The next race and its available session status.
- A timestamped **pre-session or pre-weekend baseline**.
- Existing qualifying, sprint and race forecasts at supported cutoffs, with model version and scenario information.
- Approved performance comparisons and trade-off views as data becomes eligible.
- A clear view of **what could change an expectation** before the next session, plus **what actually changed it**, including *no meaningful change*.
- A grounded **What to Watch** cue that highlights a measurable question for the next session; if the model has no defensible cue, say so.
- The existing **constructor comparison** experience, with transparent comparisons and defensible shared units.
- The existing **prediction accuracy tracker**, with frozen issued forecasts, baseline comparisons, results and calibration evidence.
- Forecast history, outcome evaluation, data coverage and freshness.

An ordinary race weekend is a complete product outcome. A week without a new discovery does not create an empty or broken homepage.

### B. Exceptional intelligence — public when evidentially safe, research-led when genuinely new

A separate detector looks for deviations that are **material, credible, not merely repeats of known patterns, and plausibly useful for a future test**.

- **Level 2 — Known mechanism, surprising case:** the system may automatically publish a bounded **Under Investigation** card when an approved method has verified the observation. It can show the known mechanism, the mismatch, measured/modelled impact, permitted alternative hypotheses and what will test them. It does **not** claim to know which explanation is true.
- **Level 3 — Unexplained behaviour:** a carefully verified, materially interesting **observation** may be published automatically as **Under Investigation · Cause unknown**, with a traceable visual and next observation. New causal narratives, intent attribution or machine-suggested explanations **stay private until explicitly reviewed**. Lower-quality or potentially misleading candidates stay private entirely.
- **Private research inbox:** a rare case includes reproducible evidence, candidate explanations, conditional impacts, tests, existing related findings and a ready-to-run experiment. The system may prepare a private draft for verification, never publish it as a fact.

These are **publication routes, not a promise that the website will explain everything**. No candidate blocks routine forecasts or asks for race-weekend approval. Cases may remain unopened, and published investigations close as supported, weakened or unresolved according to approved evidence rules—not editorial deadlines. Full rules are in the Publication Contract below.

### Separation of notifications

- **Research notifications:** opt-in, quiet and batched (for example, a digest when requested); never urgent.
- **Operational alerts:** only when automated containment fails *and* the failure threatens data integrity, availability over multiple supported checkpoints, security or a configured cost limit.
- Neither kind of alert asks for approval before ordinary sessions can process.

**Design test:** Could the creator ignore two consecutive race weekends and return to a coherent, truthful site with no accumulated editorial chores? If not, the design fails.

---

## 3. The core loop: expectation → pressure test → evidence → next expectation

The website should make a **time-ordered, reproducible chain of claims** rather than present disconnected analytics views. Its job is to help someone follow a *question* into the next F1 session, not just inspect a completed session afterward.

| Stage | Automated output | Important constraint |
|---|---|---|
| **Before the weekend** | Frozen baseline pace/rank or outcome distributions; a **Forecast Pressure Test** and a specific **What to Watch** question if supported | Use only data available before the cutoff; do not imply the scenario is a causal intervention |
| **After eligible practice** | Observed-vs-expected performance; approved trade-off charts; comparison quality and a status update to the watch question | Unknown fuel, tyres, run plans, wind and track evolution can block conclusions |
| **Before qualifying / Sprint Qualifying** | Updated or unchanged production forecast, associated evidence, and any shadow-model alternative | Never overwrite earlier forecasts; sprint weekends have different session structures |
| **After qualifying or sprint** | Outcome/resolution of the earlier question; forecast evaluation; updated next eligible prediction | Distinguish qualifying classification from penalised starting grid |
| **After the race** | Prediction scorecard, resolution trail, learning for the next event | Do not rewrite the historical information state or automatically claim an engineering explanation |

The engine is **event- and data-readiness-driven**, not a fixed Friday/Saturday/Sunday posting ritual. Different weekends have different schedules; cancellations, delays, missing sessions and incomplete coverage are supported states.

**A forecast is still produced from its approved baseline when no trustworthy new signal exists.** Evidence should change a production expectation only when a tested rule permits it.

### Forecast Pressure Test: what could change our mind?

Before the next supported session, identify at most a *small number* of conditions that could materially change the relevant forecast. The Pressure Test is a **sensitivity or conditional-scenario view**, not an automatically invented story.

For each displayed condition, store:

- **Variable and scope:** an observed measurable input or explicitly modelled scenario, with units, direction and applicable session.
- **Reference state:** the frozen forecast and model/data cutoff.
- **Conditional result:** forecast distribution *if* the input/scenario changes by a stated, defensible amount; show the difference and uncertainty.
- **Plausibility and evidence status:** where the condition came from, whether it is empirically calibrated, a scenario-only assumption or unresolved.
- **Next observation:** what the upcoming session can actually measure and when it becomes comparable.
- **Resolution:** what was observed, whether the condition occurred, and whether the forecast changed as anticipated; *unresolved* is an acceptable outcome.

**Safeguards:** Only use features with a justified relation to the model or an explicitly labelled experimental calculation. A scenario sensitivity is not automatically a prediction of occurrence, proof of causality or an endorsement of a setup choice. Avoid extrapolating beyond supported input ranges. Do not promise that the next session can resolve unknown fuel load, ERS deployment or hidden setup parameters. If nothing passes the usefulness/evidence gate, show **“No well-supported pressure point for this session”** and the standard forecast.

### Two forecast tracks

**Production forecasts:** existing dependable qualifying, sprint and race predictions, automatically frozen, displayed and evaluated. Changes are controlled through versioned, reviewed methods. Pressure Tests on production forecasts must use the production model's meaningful, permitted sensitivities.

**Experimental forecasts:** shadow calculations using candidate signals. They can appear in a clearly labelled research comparison, including conditional sensitivity, but do **not** silently replace official site forecasts. Candidates graduate only after a deliberate promotion decision and a defensible evaluation.

An untested finding may lead to a testable **conditional expectation** without changing the production forecast. **Never backfill a pre-session Pressure Test after the session using evidence that was not available at its cutoff.**

---

## 4. What counts as an insight

An insight is a **structured, testable analytical object**, not an article or a free-form AI explanation.

**Minimum fields:**

- `question` — the precise question asked.
- `expected` — a frozen baseline or explicit reference group.
- `observed` — a reproducible measured quantity.
- `difference` — effect size and units, including gains *and* costs where applicable.
- `comparison_context` — session, tyres if known, lap criteria, conditions, and coverage.
- `quality` — uncertainty, sample sufficiency, exclusions and limitations.
- `evidence_state` — observation, candidate predictive signal or validated predictive signal.
- `mechanism_status` — approved mechanism applicable / known mechanism with unexplained residual / mechanism unknown / independently supported cause / not identifiable. **Not an evidence-quality score.**
- `evidence_quality` — measurement integrity, effective sample size, comparability, uncertainty and whether the observation survived independent checks.
- `publication_state` — internal only / public established / public under investigation / suppressed; distinct from the research lifecycle status and the forecast-track status.
- `next_test` — an observable future check, if one is justified.
- `pressure_test` — optional frozen input/scenario, conditional forecast response, uncertainty and scope of validity.
- `external_evidence` — optional sourced contextual facts or manually curated statements with provenance and availability cutoffs.
- `resolution` — what the subsequent eligible session resolved, contradicted or left unknown.
- `forecast_link` — unchanged / shadow only / production changed, plus versioned delta if applicable.
- `source_and_versions` — information cutoff, source, method and visual template versions.
- `identity_and_status` — stable fingerprint, first-seen time, latest evidence, public status, research state, planned next test and a bounded closure/revisit rule.

### Publication gates

The system should apply **gates before ranking**, instead of declaring the most extreme outlier the week's top finding:

1. **Valid data:** alignment, completeness, session identity and quality rules pass.
2. **Fair comparison:** matching/normalisation is defensible; important confounders are visible.
3. **Material effect:** exceeds a predeclared practical threshold, not just statistical noise.
4. **Adequate support:** uncertainty and repeatability meet this signal type's rules.
5. **Novelty:** not a duplicate or reappearance of an already tracked finding.
6. **Future relevance:** offers a legitimate next observation or forecast implication *if* the card claims one.
7. **Reader utility:** the card can articulate one specific watch question or a well-supported reason none is available, without implying an unmeasured mechanism.
8. **Communication safety:** an under-investigation badge cannot be mistaken for official FIA/stewards activity or an allegation of cheating; the title and visual must accurately distinguish measured effects from conditional estimates.

If a gate fails, show a qualified descriptive measurement in the appropriate detail view, quarantine the candidate, or show no insight. **Do not fill an arbitrary number of homepage slots.**

A strong statistical anomaly with weak future relevance can be stored without becoming a featured insight. Multiple-testing and correlated-feature effects must be considered when searching across many signals.

**Ranking uses two separate ideas:** (a) *unexpectedness* relative to the recorded baseline and its uncertainty, and (b) *consequence*—how much the finding could matter for a future session or forecast. A large surprise that changes nothing is not automatically the headline. A small but persistent change with a meaningful forecast implication may be more important. Do not collapse these into an unexplained 0–100 “confidence” score.

### Evidence language (approved automated wording)

| Level | Allowed claim | Not allowed |
|---|---|---|
| **Observed** | “The measured gap to the reference changed by X under the stated comparison.” | “The team deliberately traded speed for battery charge.” |
| **Candidate** | “This pattern may affect Y if it persists; Z would test it.” | “This will make the car faster next weekend.” |
| **Validated predictive** | “In the stated out-of-sample tests, this feature improved target Y versus baseline by X.” | “The observed feature proves the underlying engineering cause.” |

The site should **not** dynamically invent a public causal story. It may render deterministic labels, values, equations, uncertainty descriptions and carefully approved conditional templates. **Research suggestions are a separate, private class of output**, not claims suitable for direct publication.

---

## 5. Publication Contract — known mechanisms and live investigations

The website may be **open about uncertainty without becoming speculative**. Three intelligence levels describe the *type of knowledge available*, **not** a three-step scale of probability, confidence or permission to make causal claims. Evidence quality and public visibility are independent axes.

### Three levels of analysis

| Level | What the system actually knows | Automatic public output | Human involvement |
|---|---|---|---|
| **1 · Established mechanism** | An approved model describes a repeatable conditional relationship within defined limits | Routine measured/modelled impact, assumptions, uncertainty, visualisation, forecast implications and later evaluation | None per weekend |
| **2 · Known mechanism, unexpected behaviour** | The mechanism is familiar, but the observed magnitude, conditions or residual is unusual | **Under Investigation** if measurement gates pass: verified anomaly, known model's expected range, possible *preapproved* explanations as alternatives, competing tests and bounded impact estimates | Optional deeper inquiry; no publishing approval required for approved templates |
| **3 · Novel or unexplained behaviour** | The observation is credible but existing models cannot adequately account for it | **Under Investigation · Cause unknown** **only** when independent data-quality checks, materiality and clear prospective test pass. Publish the observation and what remains unknown; **do not automatically publish novel causal speculation** | Optional private research; explicit review needed before adding a new explanatory claim or approved model |

An untrusted measurement is **not Level 3**. First check missing data, lap selection, sensor artefacts, session conditions, model applicability and multiple-testing effects. If these cannot be ruled out sufficiently, hold the candidate privately or suppress it. The absence of a good explanation is not evidence of a new physical mechanism.

Mechanisms such as **Safety Car/VSC pit-loss effects**, **tyre degradation** and **temperature-related tyre behaviour** are starting *research candidates* for Level 1, not unconditional truths with fixed numerical coefficients. Some effects are established qualitatively but require an estimated circuit/weekend-specific parameter; others are non-monotonic or too confounded to quantify reliably. Publish the model's eligible conditional estimate, its assumptions and uncertainty—or abstain. Counterfactual race positions and “the Safety Car cost driver X a podium” require much stronger assumptions than pit-loss arithmetic.

### Explicit publication decisions

| Finding | Default location | What is allowed |
|---|---|---|
| Routine, approved mechanism with usable data | **Public automatically** | Modelled/observed effect with uncertainty and input coverage |
| Verified departure within a known approved model | **Public automatically**, if it passes insight and relevance gates | **Under Investigation**; observed-vs-expected gap; bounded alternatives, next test and contextual forecast sensitivity |
| Verified and novel unexplained pattern | **Public automatically only if stricter Level 3 gates pass**; otherwise private | **Under Investigation · Cause unknown**; measured phenomenon and conditional implications; no invented mechanism |
| Possible causal story, setup intent, new mechanism, or text suggested by an AI assistant | **Private draft pending creator approval** | Candidate explanations with evidence and disconfirming tests; may be attached publicly only after explicit review and sourcing |
| Insufficient data, invalid comparison, immaterial/noisy outlier, duplicate or expired hypothesis | **Private, withheld, or archived** | No fabricated public insight, no urgency and no required creator action |

**Stricter Level 3 gates:** an auditable surprise against a frozen expectation; independent or repeated confirmation where feasible; diagnostic checks for artefacts; a material effect with stated uncertainty; a meaningful next observation; an honest way to visualise it; and no unapproved attribution of intent, illegal conduct or regulatory breach. A single giant residual, model failure or interesting-looking trace is **not enough**. The number of public Level 3 cards may be **zero for an entire season**.

### Public-facing status and lifecycle

A public investigation carries a persistent **Signal Passport**, including the original frozen claim and measurement, model/data version and an explicit status timeline:

- **Under Investigation** — the *Trackside research question* is open; observed evidence is shown, cause not established. It must say **“Independent Trackside Labs data analysis — not an FIA/stewards investigation”** in the expanded view. Do not reproduce official FIA race-control visual treatment or suggest a team has committed an infringement.
- **Supported** — a specified, **predeclared observable prediction or relationship** received supporting evidence; this is **not** synonymous with causality proven.
- **Weakened** — later evidence contradicts or substantially reduces the original inference.
- **Unresolved** — a check did not distinguish explanations, required data never arrived, or an investigation aged out of its relevance window.
- **Retired / superseded** — a method, comparison or publication was invalidated or replaced, with a visible correction trail.

Statuses are **not forced into positive/negative verdicts**. In particular, an unresolved investigation isn't a confirmed mystery or proof of an undiscovered engineering technique. Each public case has a next eligible checkpoint, expiry/revisit rule and a predetermined automatic status transition where possible. No case can become a standing demand for a creator-written follow-up. Freeze historical titles/claims as of publication and append dated updates and corrections; do not rewrite what was previously known.

The public homepage features **at most one primary investigation and optionally one supporting case**, based on demonstrated materiality and relevance; these are **capacity limits, not quotas**. If none qualifies, the normal forecast-and-trade-off experience remains complete. Deduplicate recurring observations into an existing Passport instead of publishing a fresh headline every session.

### A prepared private investigation, not a publishing assignment

When a new or ambiguous case warrants curiosity, the system may prepare a private brief in this form:

> **Something unusual may be happening.** Verified observation and effect size; what our approved models predicted; where the difference appears in the visuals; potential explanations **labelled as hypotheses**; which are consistent or inconsistent with available evidence; estimated consequence **only where an approved conditional model supports it**; what additional data might discriminate among the alternatives; reusable charts and experiment bundle. **Review before publishing any new explanation.**

Optional model-assisted text generation is allowed **only inside this private brief**, with traceable numerical inputs and explicit uncertainty. It may propose tests, not assert facts. No public draft is automatically published, and approval of one interpretation does not grant approval of similar future claims. Do not use an LLM to make routine production operation depend on token availability or a human review cycle.

### What would make this a real differentiator?

**Not the motorsport wording or the volume of anomalies.** The distinctive, testable feature is a public **open scientific loop**: original pre-event expectation → observed mismatch → visual trade-off and competing tests → frozen forward implication → subsequent evidence → supported, weakened or unresolved outcome. Visitors can see which ideas survived, including the wrong and inconclusive ones, without a long article. A brief qualification label is useful only if the evidence and evolving visual record justify following it.

Test the idea with a real Level 1 analysis and a credible Level 2 case **before** implementing automatic public Level 3. The differentiator has not been demonstrated if visitors see another generic anomaly feed or cannot tell observation from conjecture.

---

## 6. Human control: hypotheses, causality and research direction

**The creator controls what the system is allowed to infer—not each individual publication.**

For every approved analytical method, maintain a lightweight **Hypothesis Definition / Mechanism Model** (a version-controlled Python configuration or YAML file is sufficient initially). This includes **Level 1 models**, not only anomalous-signal detectors:

| Field | Intent |
|---|---|
| ID, version and status | Stable lineage; draft / experimental / approved / retired |
| Research question | What are we trying to detect or predict? |
| Observables and comparison rules | Exactly how to measure it and who/what to compare |
| Known confounders and exclusions | Conditions that limit or invalidate the comparison |
| Materiality and quality gates | Rules for detecting and suppressing findings |
| Permitted claims and alternatives | What is observed, what mechanisms remain hypotheses |
| Next test and target | What future data would support or weaken it |
| Visual story template | How evidence should be shown |
| Forecast permission | Descriptive only / shadow forecast / approved production feature |
| Mechanism applicability | Conditions and input ranges where a relationship is supported; outside-range abstention |
| Public publication authority | Routine approved output / bounded Level 2 / verified unexplained Level 3 observation / private only |
| Investigation closure | Next test, expiry, correction rules and whether updates can occur automatically |

**Do not build a large generic rule-engine UI just to store these definitions.** Start with a handful of versioned configurations in the repository. Introduce a registry service only when duplicated logic or maintenance demonstrates a need.

### Causality policy

Observational F1 telemetry can support **descriptions, associations, predictive usefulness and competing mechanistic hypotheses**; it rarely identifies team intent, setup parameters or true causal effects on its own.

For example, a late-straight speed decline may be consistent with different deployment profiles, aerodynamics, corner-exit speeds, wind, regulation effects, setup or an unobserved limitation. The **measured speed loss and its time consequence** are the publishable facts. “Mercedes chose to save energy for another part of the lap” is not a confirmed fact without independent supporting evidence.

Causal claims need explicit assumptions and supporting evidence. When multiple mechanisms fit the observations, the system says **“mechanism unresolved.”** Feature importance or SHAP-style contributions are not mechanical explanations.

The creator may manually add a researched interpretation later. It must be distinguishable from automated evidence and anchored to specific observations/sources; it must not be necessary for routine updates. A system **may publish “cause not established”** without waiting for the creator. A speculative cause remains private even if an AI-generated brief sounds persuasive.

### External Evidence Layer: context without invented explanations

Telemetry describes visible behaviour, but does not always explain its mechanism. The system may enrich an investigation with **structured, attributable context** where sourcing, timing, comparability and usage rights permit it:

- Circuit and session characteristics, weather, track conditions, and confirmed tyre/compound information.
- Published technical or sporting regulations and circuit-specific rule changes, with effective dates.
- Officially declared upgrades or publicly documented equipment changes, as factual context only.
- Optional human-curated team/driver statements or independent technical reporting, clearly attributed and dated.

Each item needs a **source URL/identifier, source type, publication and availability time, event applicability, raw-versus-interpreted flag, retrieval/version provenance and rights status**. If the fact was published after a forecast cutoff, it may enrich later retrospective interpretation but **must not leak into the earlier forecast or its pre-session story**.

**Automation boundary:** approved structured feeds can update routine charts; ambiguous press claims, motives, setup intent and mechanistic conclusions require human review to appear as interpretations. No automated scraping-and-paraphrasing of unsupported articles, no LLM-generated causal explanations, no dependence on external context for an otherwise healthy pipeline. Missing context is displayed as missing, not inferred. Keep the data integrations few and resilient; source availability or licensing restrictions may prevent use.

A possible energy-management explanation remains a **hypothesis** unless independent evidence distinguishes it from competing mechanisms. External statements can document a team's claimed intent without verifying that the measured effect was caused by that intent.

---

## 7. Visual storytelling is part of the analytical method

**A Trackside Labs visual should allow the reader to *see* why a claim was made and where it could fail.** The chart is often the primary output, not an illustration beneath a paragraph.

### Default visual language

1. **Baseline before headline:** contrast expected and observed values whenever possible.
2. **Paired benefit and cost:** show time gained *and* time lost using aligned scales.
3. **Chronology and direction:** mark exactly when evidence arrived relative to forecasts.
4. **Distribution, not false precision:** show variation, ranges, sample sizes and model uncertainty.
5. **Observed ≠ modelled ≠ hypothesised:** use distinct, consistent visual treatments.
6. **Few carefully designed compositions:** avoid an endless gallery of widgets.
7. **Progressive disclosure:** the takeaway works at a glance; interactions expose details; methods enable verification.
8. **Mobile and accessible:** legible text, colour-blind-safe encoding, explicit units, no hover-only conclusions, reasonable loading speed.
9. **Reproducible design:** visual transformations, annotations and screenshots are tested like calculations.
10. **Shareable by default:** a static card can be rendered from the same saved dataset and chart specification as the interactive view, without manual design work.

### The signature visual stories

| Visual story | Purpose | Required elements |
|---|---|---|
| **Expectation Gap** | Show what differs from pre-session expectations | Baseline, observed value, uncertainty, effect size and timestamp |
| **Performance Trade-off Map** | Show where an advantage is won or paid for | Matched speed-over-distance and/or time-delta trace, gain/loss segments, net effect, comparability |
| **Evidence → Forecast Revision** | Show whether new data actually changed the view | Previous distribution, new distribution, delta, supporting observations, cutoff and later outcome |
| **Forecast Pressure Test** | Show what future observations could meaningfully change a prediction | Frozen baseline, conditional scenarios, ranges, sensitivity, evidence caveats and subsequent resolution |

The **same analytical episode** should link these views. Avoid dashboards made of impressive but disconnected charts. Treat the Pressure Test as a conditional layer of the forecast story—not a mandate to show an extra chart for every event.

**Signature longitudinal experience: the Signal Passport.** Each qualifying finding has a persistent, shareable visual record: *first detected → measurements and trade-off → investigation state → competing tests → subsequent evidence → forecast impact → supported / weakened / unresolved outcome*. For public Level 2/3 cases, the visual timeline must make **“what we observed”** and **“what we still do not know”** equally obvious. The record can span multiple weekends, accumulate contradictory evidence and close without being “confirmed.” A compact **Surprise Ledger** may eventually summarise how frequently qualified deviations were useful versus the baseline, including false alarms and inconclusive cases. Do not build a new database subsystem for this: start by linking the existing versioned insight and forecast records.

A public investigation should feel like **an open case, not a finished article**. Its primary visual shows the gap from expectation; secondary visual(s) show the measured gains/losses, validity limits and the next discriminating observation. The status indicator must never be the main evidence. Use F1-inspired wording with independent branding and an explicit separation from official steward investigations.

A useful public card has three reading depths:

- **Glance (10 seconds):** a meaningful title, one clear graphic, effect size, evidence label and an actionable **What to Watch** question or “not established.”
- **Explore (1–3 minutes):** drivers, laps, segments, comparison options, distributions and linked observations.
- **Verify:** source IDs, date/as-of time, sample selection, exclusions, methodology, uncertainty, versions and historic forecast snapshots.

### One signature interactive investigation: Performance Trade-off Explorer

Build **one** purposeful interaction rather than a generic telemetry playground. Let a visitor compare two drivers/teams or representative laps and explore **where equal or similar total pace comes from different gains and losses**:

- **Primary view:** distance-aligned cumulative time delta with annotated straights/corner groups and net lap effect. Optional synchronised speed trace adds context, not a separate conclusion.
- **Selection:** eligible team/driver, session and approved comparison pair; restrict comparisons to defensible lap/condition/coverage combinations.
- **Robustness:** show whether the pattern persists across eligible lap selections or uncertainty ranges; rejected/mismatched laps and unavailable comparisons are explained.
- **Interaction:** hover/tap or select a segment to highlight the corresponding gain, loss, uncertainty and evidence state in all linked views. Support mobile tap and keyboard focus.
- **What it must not imply:** equal final lap times do not establish a deliberate engineering trade-off. Gains and losses may arise from unrelated factors.

The frontend reads **precomputed, versioned and bounded comparison datasets**. Interactions must not trigger expensive live simulation, API fetching or hidden manual review. Limit available pairs rather than display invalid comparisons. A static, shareable version must preserve the main conclusion for non-interactive readers. If the interaction adds little clarity beyond the static chart, keep the static version.

### Visual Story Definitions

Pair each approved hypothesis with one **versioned visual template** containing: chart type, axes, permitted comparisons and controls, labels, uncertainty treatment, explanatory footnotes, responsive layout, a pre-renderable social image and suppression rules when inputs are missing.

**The computer selects among approved visual stories; it does not invent new chart narratives each weekend.** A missing or non-comparable quantity must remove/downgrade the corresponding chart, not silently create a misleading graphic.

### Design acceptance test

Before any public chart ships, answer:

- What exact question can the reader answer from this chart?
- Are the reference, units, sign/direction and cutoff unambiguous?
- Does the chart show the net effect rather than cherry-picking a positive segment?
- Are uncertainty, exclusions and missing data visible where relevant?
- Could a reader misread an association as causality?
- Does the 10-second view work without an article or a desktop hover interaction?
- Can a visitor identify the specific upcoming observation to watch, and later tell whether it was resolved?
- Does an interactive comparison clarify the evidence without letting visitors generate scientifically invalid comparisons?

---

## 8. Operational autonomy: an explicit contract

**Target: zero *required* race-weekend actions by the creator during ordinary supported events.** This is an engineering acceptance criterion, not a guarantee that external services will never break.

### State machine and safe defaults

| System state | Automatic behaviour | Visitor sees | Creator action |
|---|---|---|---|
| **Healthy** | Ingest → validate → compute → evaluate → publish an atomic snapshot | Current eligible views and as-of times | None |
| **Waiting** | Back off until the session/source data is eligible; retry within bounded limits | “Awaiting complete data” and last valid snapshot | None |
| **Degraded** | Isolate failed analyses; suppress affected claims; preserve unrelated outputs | Partial coverage, clear reason and last-good result | None by default |
| **Recovered** | Idempotently recompute and publish a new complete snapshot | Current data; old snapshots remain accessible | None |
| **Persistent failure** | Trip circuit breaker, retain last valid public state, log diagnostic and escalate only if operational threshold met | Explicit stale/unavailable label | Scheduled repair when practical |
| **Interesting anomaly** | Diagnose, deduplicate, classify knowledge level and apply public Level 2/3 eligibility rules; continue routine operations | Qualified **Under Investigation** observation or no public card | Optional, asynchronous investigation |

Research novelty must never be treated as operational failure. System failure must never become an unsourced F1 insight.

### Required operational behaviours

- **Schedule from actual event/session metadata**, not a hard-coded race-weekend calendar. Respect postponed, cancelled, sprint and changed-session formats.
- **Check data readiness**, not just elapsed time. Partial data cannot impersonate complete data.
- **Use idempotent jobs and stable keys** so retries, backfills and late source corrections do not duplicate insights or forecasts.
- **Cache raw inputs and derived outputs**; enforce rate limits, timeouts, bounded retries with backoff and configured cost ceilings.
- **Run calculation outside page rendering**; the frontend should read versioned, already-computed results.
- **Publish atomically** so no visitor sees half-updated charts paired with a different model version.
- **Preserve last known good results** with visible freshness and provenance when new data is unavailable.
- **Never overwrite a forecast after its information cutoff**, even when data arrive late or a bug is found; publish corrected versions as separate, explicitly marked records.
- **Never require human approval to complete a normal weekend or close a published research case.** Known methods and bounded observations publish or abstain automatically; new causal interpretations stay private until reviewed.
- **Publish a versioned investigation state transition**, not a silent change to a headline. Retain the earlier observation even if the candidate is later weakened, corrected or closed unresolved.
- **Use a small set of meaningful operational checks**, with aggregated alerts only for persistent, consequential failures.
- **Allow safe manual re-run and pause/rollback**, without making manual operation the normal path.

### How autonomy is tested

Replay at least **three historical weekend types** through the real orchestration and UI output:

1. A normal weekend with all sessions available.
2. A sprint weekend with different session ordering.
3. An incomplete, delayed or corrected-data weekend.

Simulate duplicate jobs and an unavailable upstream API. Check that the result is either **correct** or **clearly withheld/stale**, never silently misleading.

Then run the full pipeline **unattended for two eligible consecutive weekends**. Review afterwards, not between sessions. These are *acceptance tests*, not a commitment to interrupt those weekends for manual oversight.

### Maintenance budget

Avoid introducing components that require frequent handholding. Prefer a simple scheduled pipeline plus persistent snapshots to a microservice architecture, live-streaming infrastructure or elaborate orchestration tool.

Monitor **human interventions per weekend**, freshness/coverage, false alarms, duplicate case suppression, unresolved-case ageing, bad publications prevented, failed/recovered runs and operating cost. If a feature demands regular weekend intervention, simplify, downgrade or remove it before adding further features.

---

## 9. Forecasting and scientific integrity

### Non-negotiable forecasting rules

- Keep qualifying, sprint and race predictions from the current project where they work; avoid gratuitous rebuilding.
- Define the prediction **target** precisely (e.g., qualifying classification versus official starting grid; race finishing position versus classified result).
- Record **what was knowable at each information cutoff**, including which sessions were available and source snapshots used.
- Establish a transparent **recent-form or other simple baseline** before claiming an advanced feature helps.
- Measure predictive value at the **event level**, using chronological/rolling backtests, never random telemetry-level splits that leak a weekend into training and evaluation.
- Avoid treating thousands of telemetry samples as thousands of independent race-weekend observations.
- Evaluate uncertainty and calibration with appropriate metrics; compare models on the same targets and cutoffs.
- Retain original forecast distributions, versions and timestamps; report misses and inconclusive results as faithfully as hits.
- Treat 2026 as a changed technical regime. Use older data conservatively for relatively stable circuit attributes, not as interchangeable observations of current-generation car performance.
- Require meaningful out-of-sample support before an experimental feature enters **production**, and document decisions when evidence is limited.

### Promote experimental features deliberately

**Explore → shadow forecast → prospective/rolling evaluation → human review → versioned production approval → monitor and retire if needed.**

No novel signal changes published production predictions merely because it looks surprising. Conversely, the exploratory track should remain available even when there is not yet enough evidence to promote it.

### Confounding and data limitations

Public feeds provide incomplete information about energy deployment, active aerodynamics, setup, fuel, tyres and engineering decisions. Some relationships are fundamentally non-identifiable from the available telemetry. Matching laps across drivers or sessions is hard because of track evolution, traffic, run programmes and conditions.

FastF1 and OpenF1 are useful sources but have availability, resolution, licensing and access constraints; data timing and usage rights must be checked for each source and revisited if the project becomes commercial. Do not promise real-time predictions when dependable historical/post-session feeds are sufficient.

Separate **observed physical measurements**, **model-derived estimates**, **predictive associations** and **hypothesised mechanisms** throughout storage, code and visuals.

---

## 10. The research inbox: human attention only when it is worth something

A rare research candidate must arrive with a **prepared case**, not a request to check a random outlier. A published **Under Investigation** observation is not itself a request for human action. Public status resolution follows approved checks; the inbox is for creator curiosity and potentially novel explanations:

- The expected-vs-observed gap and the measurements underlying it.
- The location and size of the performance trade-off, if supported.
- A quality and persistence assessment, with known confounders.
- Whether it resembles an existing signal or is genuinely different.
- Why it could matter for a subsequent session or circuit.
- A proposed next test that can support, weaken or distinguish hypotheses, including when/why the case should close **unresolved**.
- A **private** proposed explanation of why the deviation occurred and what it might have changed, with each numerical consequence linked to an approved model or explicitly labelled scenario and with unverified claims clearly marked. The private brief is a research starting point, not publishing-ready copy.
- A reproducible saved visual/story with sources and versions.

**Deduplicate across sessions and weekends.** Repeated examples of the same pattern should update one case, not create a new notification.

The creator can choose to **investigate, refine, dismiss, defer or retire** the case. Opting out cannot leave a public status stuck awaiting a promised manual verdict. A choice affects research direction and analytical methods, not the routine operation of the live website.

The inbox does not impose a deadline. Without human action, production forecasting and routine publishing continue normally using approved methods. Novel candidates stay experimental or descriptive.

### Research Workbench: curiosity without data-preparation chores

The research inbox must offer **a prepared, reproducible launch point** when the creator decides to explore a case. Start with a lightweight generated experiment bundle (Python notebook/script + configuration + data references), **not** a notebook hosting product or a general-purpose hypothesis GUI.

A case launch should provide:

1. **Question and competing hypotheses:** measurable claim, alternatives, likely confounders and a falsifiable next observation.
2. **Frozen evidence bundle:** event/session identifiers, selected and excluded laps, accessible cached data references, comparison baseline, transformations, model/version IDs, source cutoffs and provenance.
3. **Ready visual starting point:** the same chart/comparison that triggered the case, reproducible in code.
4. **Experiment scaffold:** edit a feature or comparison rule, run against saved historical checkpoints, compare with baseline using valid chronological evaluation and export a shadow result.
5. **Outcomes:** record supported / weakened / inconclusive, with notes and a possible follow-up—without requiring publication or production promotion.

Prefer **links to cached/versioned datasets** over copying bulky telemetry into every bundle. Respect licences and access controls; do not package data for redistribution when rights do not allow it. The experiment must re-run without depending on the current live website state, and should expose leakage risks.

**Personal success criterion:** Going from “that looks odd” to a defensible first experiment takes little setup. Improving methods and exploring F1 patterns should consume more time than maintaining the pipeline.

An occasional original analysis or social-media post can come from this queue. Publishing it is optional and never on a fixed schedule.

---

## 11. Product, engagement and discoverability

### Homepage: “What should I watch for next?”

The website is for people who want to **notice something during the next session**, not just read a technical debrief. Lead with a concrete anticipation cue, while retaining the scientific foundation:

1. **What to Watch:** at most one or two clearly evidenced questions for the upcoming supported session. E.g. “Will the unusual straight-line profile persist on representative qualifying laps, and is time recovered elsewhere?” This is a question, **not** a claim about engineering intent.
2. **Weekend outlook:** event/session context, available data, fresh production forecasts and as-of time.
3. **What could change the prediction?** A Forecast Pressure Test using supported model sensitivities; label scenario assumptions and uncertainty.
4. **What the evidence shows:** the biggest *qualified* expectation gap and a trade-off visual **if** evidence supports them, including a carefully labelled **Under Investigation** case when eligible. The question can remain open without a speculative answer.
5. **Resolution:** the original question and frozen forecast beside subsequent evidence, actual changes and outcome; unresolved stays unresolved.

The default view adapts to **before**, **during** and **after** an event; a persistent Signal Passport connects all phases. A fan should be able to follow an idea across sessions without searching through separate pages or remembering previous predictions.

**Quiet-weekend behaviour:** Never fill a slot with an arbitrary anomaly. When no qualified signal/Pressure Test exists, show the current forecast, normal performance comparison, data coverage and a clear “No meaningful evidence to revise expectations” state. No dramatic headline is required.

Each card links to its detailed **Signal Passport**, the official forecast snapshot, the experimental alternative if eligible, the corresponding visual evidence and eventual scoring. A new **Under Investigation** card should be compelling because the measured puzzle and upcoming falsifiable test are visible, not because of a dramatic badge. Avoid a news feed, generic “AI takeaways” and a dashboard catalogue.

### Explore: a single compelling interaction

The first exploratory destination is the **Performance Trade-off Explorer** defined in §7. The experience should let a fan form and test one limited comparison, not ask them to become a race engineer. The result should answer: **where did the cars gain/lose time, was there a net advantage, and which parts of the comparison remain uncertain?**

A readable, useful static rendering is the fallback on slow devices, noninteractive previews and search/social link cards. Add more interactions only when user observation shows that the existing one creates value.

### Automated discoverability, without an editorial schedule

The website should be **findable and shareable as a technical default**, not depend on the creator posting every race:

- Stable, human-readable **public URLs** for events, frozen forecast snapshots and qualified Signal Passports, including open investigations with a persistent dated history. Preserve permanent IDs across subsequent corrections and status changes.
- Indexable page summaries with useful titles, descriptions, canonical URLs, robots directives and structured metadata *where appropriate*. Never index private/draft research cases, duplicate snapshots or suppressed evidence as though they were published insights.
- Automatically maintained sitemap and internal links from event → insight → forecast → outcome. Respect late publication, retirement and corrected data.
- Automatically rendered **social preview images** from the same approved visual dataset/template; readable without hover and labelled with the cutoff/units.
- A lightweight public static/prerendered representation when the current JavaScript-heavy frontend cannot expose meaningful crawlable content. Check actual indexing before considering a full frontend rewrite; **SEO does not justify abandoning a working research stack**.
- No requirement for social accounts, outreach, weekly articles or analytics-growth targets to keep the site operating.

Discoverability is a **one-time capability with automated upkeep**, not a content marketing project. Periodically verify that representative public pages render and are indexable; do not confuse an indexed URL with a validated acquisition channel.

### Test whether actual F1 fans understand it

Before major UX expansion, test a **real automatically generated** episode—not a hand-written demo—with **3–5 F1 fans unfamiliar with the project**. Let them explore independently, preferably including one mobile session. Ask them to show, rather than merely tell:

1. What did Trackside Labs expect, and *as of when*?
2. What evidence would change that view, and what should they watch for next?
3. Which differences are measured and which are hypotheses? Can they tell that **Under Investigation** means independent Trackside research, not FIA steward activity?
4. Did the later evidence support, contradict or fail to resolve the idea? Does an unresolved case look like an honest outcome rather than forgotten content?
5. Which visual did they want to investigate further, and what confused them?

**Pass condition:** Participants can locate the core evidence and next observation without an article or creator explanation. Record confusion and improve the visual design; do not optimise for flattering feedback or invented numeric conversion targets. User testing informs **clarity and relevance**, not the direction of scientific research.

### Presentation standard

**Differentiation test:** Compare the homepage and one real Signal Passport with leading F1 prediction/analytics sites. Within roughly 15 seconds, an interested fan should recognise a different question being answered: **“What should I watch for, what could change the prediction, and what did the evidence eventually show?”** If they see only generic telemetry and a top-10 forecast, the experience has not earned its positioning.

The site should feel like a **small, carefully edited analytical instrument**, with a strong visual hierarchy, not a news portal, a grid of charts or a replica of another project. Preserve the existing identity and working Python analytics, but **replace the public Streamlit UI**. Build visual clarity around verified data, not the reverse.

---

## 12. Platform rebuild, existing features, and prediction correctness

This section is a deliberate engineering decision, not a suggestion to replace the working forecasting codebase.

### Protected existing product features

These survive the migration **without functional regression**:

| Capability | Role on the new site | Preservation contract |
|---|---|---|
| Qualifying, sprint qualifying, sprint race and Grand Prix race predictions | Secondary dedicated **Predictions** area; surfaced as compact supporting evidence from the primary intelligence pages | Preserve targets, cutoffs, probability distributions, position estimates, scenario/weather context and model-version attribution. Never recompute forecasts in the browser. |
| Constructor comparison | Secondary **Compare** area, with stronger visual design and explainable, correctly normalised performance comparisons | Preserve available metrics, comparisons, historical data, source and freshness. Replace chart presentation, not underlying definitions without an explicit migration. |
| Accuracy tracker | Secondary **Track Record** area; prominently reachable wherever a model claim is made | Preserve original issued forecasts, scoring cohorts, timestamps, frozen model versions, uncertainty calibration and honest baselines. Do not substitute hindsight replays for published performance. |
| Optional diagnostics | Personal/advanced research area, not the default homepage | Keep auditable evidence and debugging tools; access control for operational and privileged actions. |

**Secondary priority means less prominent in the main story, not less reliable or slated for removal.** Visitors who mainly want predictions or constructor comparisons should find them in one interaction. Existing bookmarked routes should redirect to their new equivalent where feasible.

### Architectural direction

- **Python remains the analytical backend:** existing predictors, simulations, FastF1/OpenF1 acquisition, scheduled warmup jobs, feature extraction, scoring and Supabase `ArtifactStore` are retained initially. Refactor boundaries rather than rewriting mathematical logic during a UI migration.
- **API:** adopt **FastAPI + Pydantic** as a narrow, versioned, read-only public API over *published and validated* snapshots. It must not instantiate a predictor, download FastF1 data, or update learning state for an HTTP page request. Privileged research/admin operations are separate and authenticated; they are not exposed through public GET endpoints.
- **Frontend:** prefer **Next.js + React + TypeScript**. Use server-rendered/indexable pages for events and investigations, a mobile-first responsive design, reusable theming tokens (CSS variables/Tailwind), accessible components, and a consistent interaction vocabulary. Avoid a one-off hard-coded dashboard aesthetic. Prove this stack with an end-to-end vertical slice before treating it as permanent.
- **Visual engine:** choose a charting foundation suited to linked views, distance-aligned telemetry, uncertainty bands, brushing, hover comparisons and exportable static/share images (for example ECharts with limited bespoke D3 where necessary). Chart implementations must respect units, sample coverage and chart-validity rules defined by Python analysis; a frontend visual can never silently reinterpret the data.
- **Deployment:** deploy frontend and API separately, retaining the scheduled Python workers. Publish immutable, schema-versioned JSON payloads or database snapshots. The frontend consumes the **last complete, validated** publication if workers fail. A static or server-rendered event page must not depend on an active Python model run. No service-role credentials in browser bundles.
- **Data contracts first:** typed API schemas, provenance IDs, as-of cutoffs, publication eligibility, generated-at/freshness, model version, forecast identity, units, uncertainty and missing-data states. Version contracts and provide contract/fixture tests on both sides.
- **Visual design standard:** design tokens for typography, spacing, colours, chart encodings and light/dark themes, dark mode by preference rather than necessity, WCAG AA contrast, keyboard navigation, reduced motion, touch-friendly charts and static fallbacks. The new website should feel bespoke, not like a default UI starter.

Proposed information architecture:

| Route | Primary experience |
|---|---|
| `/` | **What to Watch** and the most meaningful verified evidence; quiet-weekend mode works too |
| `/events/[event]` | Event lifecycle: expectations → evidence → revisions → outcome |
| `/investigations/[id]` | Signal Passport, Under Investigation lifecycle, interactive visual evidence |
| `/explore` | One excellent, constrained Performance Trade-off Explorer |
| `/predictions` | All existing predictions, including qualifying, sprint and race |
| `/constructors` | All existing constructor comparisons |
| `/accuracy` | Existing accuracy tracker, baselines and frozen history |
| `/research` | Optional personal workbench and diagnostics (private or explicitly separated) |

These are conceptual paths, not a requirement to create every route in the first pull request. Validate URLs and indexing in the first vertical slice.

### Prediction defects and calibration debt: repair, do not silently retune

The repository already records known problems in `LIMITATIONS.md`, `docs/MODEL_LEDGER.md`, `docs/FORECAST_LINEAGE_AUDIT.md`, `docs/SCORING_PARITY.md` and the latest evaluation reports. Treat those as the initial triage backlog, not as proof that every item is still present at the current HEAD. Specific audit targets:

1. **Prospective-vs-replay parity:** distinguish frozen forecasts actually issued from retrospective simulations; identify which model-version/state/cutoff differences drive apparent quality gaps. Never silently apply evaluation-only adjustments to production or headline accuracy.
2. **Prediction lineage and safe serving:** verify model identity, artifact hashes, checkpoint boundaries, actual/predicted grid routing and fallback compatibility; disallow unverified cross-model results being passed off as current.
3. **Retirement risk:** the legacy per-lap `p / N` hazard under-delivers race-level risk. A correct probability conversion is already available behind a feature flag, but reportedly did not pass replay evaluation. **Do not flip it without re-estimating/calibrating the underlying DNF model and passing the existing promotion gate.** Review incident classification and per-driver Brier skill against the base rate.
4. **Prediction intervals:** investigate why replay widening and saved production intervals differ. Keep nominal coverage and observed coverage together; no cosmetic narrowing or hidden post-processing.
5. **Leakage and independence:** refit or isolate any 2026-wide constants that can leak later information into earlier event replays. Evaluate by event and checkpoint, never by random individual telemetry samples.
6. **Safety Car, sprint and tyre mechanics:** add invariant/property tests for probabilities, event timing and risk scaling; compare simulated event rates to configured race-level probabilities, not only ranking MAE.
7. **UI/scoring semantics:** correct mismatched captions (median-rank vs expected position), explicit targets and sample counts. A visualization must not upgrade an approximation into a stronger statistical claim.
8. **CI and fixture isolation:** reproduce any failing golden tests and remove side effects on committed prediction artifacts. No silently regenerated goldens merely to make a new model appear correct.

**Bug-fix acceptance:** Reproduce the specific defect; add the smallest deterministic regression test; fix the cause with no unexplained change to unrelated paths; run focused tests, type/lint and historical checkpoint replay; compare probabilistic calibration, forecast quality and simulation realism to the frozen baseline; record the outcome in the model ledger. Use shadow flags or draft-only deployment where historical evidence is insufficient. An improvement to mathematical internal consistency does not by itself justify promoting a recalibrated production model.

### Migration strategy and rollback

1. Freeze a fixture set of issued predictions, constructor data and accuracy snapshots with **as-of timestamps**. Record current API/data semantics and the exact version of the Python code.
2. Extract one stable public read model from existing persisted artifacts. Implement a typed FastAPI endpoint and contract test that returns the same published forecast as the legacy UI, without invoking expensive predictors.
3. Build **one complete frontend vertical slice**: event overview + primary What-to-Watch visual + a link to the preserved prediction and accuracy data. Validate on normal, sprint and missing-data fixtures.
4. Build the design system and routes iteratively. Migrate existing predictions, constructors and accuracy views before disabling Streamlit. Run old and new UIs in parallel against the *same* immutable artifacts; compare content as well as visuals.
5. Introduce signature investigation visuals, publication lifecycles and shadow forecasts only after parity and the autonomy contract hold. Do not couple experimental signal deployments to critical forecast jobs.
6. Shift the public domain behind a reversible routing change, observe telemetry/error budgets, preserve bookmarked deep links, and keep a tested rollback path until at least three uneventful supported event cycles. Decommission Streamlit only after parity and stability checks.

**Out of scope for the first migration:** replacing Supabase, rewriting Monte Carlo models, switching data vendors, introducing live LLM explanation publishing, building microservices, adding accounts, or promising real-time telemetry. Modern design is a reason to change the frontend—not the reason to discard working analysis.

---

## 13. Execution plan: prove the loop, then extend it

**Principle:** Build the smallest *complete* automated experience rather than a large collection of partially automated capabilities. The engagement/research additions are **sequenced**, not parallel feature programmes: first a useful watch question and one strong static visual, then reproducible research and a bounded Pressure Test, then a single interaction and richer contextual evidence. Basic shareability arrives early; more discoverability work follows evidence of usefulness.

### Stage 0 — Audit and protect what exists

Inventory the existing data sources, FastF1/OpenF1 integration, forecasting models, Streamlit UI, job scheduling, database/storage and failure points. Identify what can be reused unchanged. Record baseline forecast quality **and frozen published prediction identities**, not just replay scores. Audit the prediction defects listed in Section 12, rank by correctness impact and risk, and isolate at least one regression test before changing production behaviour.

**Exit:** One documented current pipeline, a frozen set of published-data fixtures and a prioritized correctness/quality register. The frontend replacement follows the migration plan in Section 12; model rewrites are not part of it.

### Stage 1 — Autonomous weekend backbone

Establish event-aware orchestration, readiness/quality gates, immutable forecast cutoffs, versioned outputs, atomic publication, last-good fallbacks and visible data freshness. Inventory current approved **Level 1 mechanisms**, with applicability and abstention rules. Begin with those actually supported by existing data and models; do not assume tyres, temperature or Safety Car counterfactuals are already calibrated. Keep existing predictions and a basic visual summary running without manual refresh.

**Exit:** The three historical replay tests pass; routine production forecasts and existing views are unattended. **This reliability foundation precedes a new discovery engine.** Preserve as-of snapshots needed later for Pressure Tests and resolution, without implementing a general scenario engine yet.

### Stage 2 — Modern frontend vertical slice and one signature investigation

**Question:** After practice, can we detect a meaningful *expected-versus-observed* difference in relative qualifying performance, decompose the supported gain/loss profile, and state an honest next test?

- Expose the relevant **published snapshot through a versioned, read-only Python API** and render the episode in the new Next.js frontend, using shared design tokens, responsive layouts and accessible static fallbacks.
- Keep the **predictions, constructor comparison and accuracy tracker** reachable and ensure their data contracts stay compatible; full page migration follows the staged plan.
- Start with **one** approved hypothesis and a conservative recent-form reference.
- Use defensible clean-lap selection and conditions; handle insufficient sample/data explicitly.
- Make the **Expectation Gap + Performance Trade-off Map** the hero visuals.
- Show measurement, uncertainty, alternative explanations and any limitations.
- Present one grounded **What to Watch** question before the next session and attach its later resolution; if no qualified question exists, use the honest quiet-weekend state.
- Render an indexable/shareable static version of the episode with a stable ID and graphic through the new frontend; this should be the first fully integrated migration slice, not a separate manually published page.
- Automatically publish eligible observations and suppress unsupported conclusions.
- Track the episode across relevant sessions; apply stable IDs and deduplicate repeats.
- Demonstrate **one real Level 1 routine output and one Level 2 Under Investigation case** on saved data, both published or withheld using explicit evidence rules. The Level 2 case must update or close without a human-written follow-up.

**Exit:** One useful **anticipation → observation → resolution** episode is automatically created from data and remains defensible under normal/sprint/incomplete-data replays; the Level 2 public lifecycle works without manual approval. A reader can understand it without a manually written article. It does not need a novel causal explanation.

### Stage 3 — A forward-looking shadow experiment

Introduce **one candidate feature family** and a shadow forecast. Show what it *would* change, what evidence supports it, and whether prospective outcomes validate the change. Add a **bounded Pressure Test** for that feature where the sensitivity is meaningful, without implying the scenario will occur. Keep production forecasts untouched until a deliberate promotion decision.

At this stage, expose a **minimal Research Workbench** by exporting a saved case as a reproducible Python experiment/configuration that can regenerate the triggering visual and run a baseline comparison. No hosted notebook app.

**Exit:** A reproducible evaluation, a clear visual forecast/Pressure-Test-to-resolution story, a one-step research starting point, and an accept/reject/inconclusive outcome. Null results count as completed research. No race-weekend human action is required.

### Stage 4 — Broaden only when justified

First evaluate whether a **single interactive Performance Trade-off Explorer** clarifies an actual episode beyond the static chart, and whether a small number of structured external evidence sources genuinely improve interpretation. Add the interactive view only with controlled, cached comparisons. Test a real episode with 3–5 F1 fans before adding more dashboards.

Only after the published Level 2 experience is demonstrably useful, consider **verified-but-unexplained Level 3 public observations** behind strict gates and replay tests. Keep novel causal hypotheses private unless manually reviewed. Other research candidates: circuit transfer, tyre warm-up signatures, changes in team performance, better anomaly ranking, improved uncertainty models. Extend only when a method improves understanding, validated prediction, autonomy or visual communication. **These are options, not a committed roadmap.**

### Explicitly not part of the initial build

- General-purpose AI analyst or automated explanatory articles.
- A full pit-wall strategy system or live race simulator.
- Dozens of detectors or a generic hypothesis-authoring interface.
- A custom microservice platform or rewriting the Python analytics engine as part of the **necessary frontend rebuild**.
- Manual publication of every session, social-media cadence or SEO campaign.
- Live LLM-driven headline generators, unsourced contextual scraping, or a generic automatic explanation engine.
- Interactive analysis that fetches raw session data or reruns costly models on each user action.
- A large public Research Workbench GUI, broad technical news feeds, user accounts or an unlimited scenario simulator.
- A rule that every race must produce an extraordinary insight.
- A duty to manually approve Level 2 cards, resolve investigations by writing posts, or respond to a novelty alert during a race weekend.
- Automatic public AI-generated reasons for unexplained performance or accusations based on inferred team intent.

---

## 14. Decision rights and trade-off rules

| Decision | Default owner |
|---|---|
| Pick the next research question or novel signal family | **Creator** |
| Define causal hypotheses, confounders, thresholds and approved interpretation | **Creator** |
| Design/approve a new visual story, valid interactions and analysis method | **Creator** |
| Choose which external qualitative claims or causal interpretations merit attachment | **Creator** |
| Collect, validate, calculate and apply already approved analyses | **System** |
| Produce routine forecasts, defensible What-to-Watch cues/Pressure Tests and qualified outputs | **System** |
| Retry, contain ordinary failures, suppress incomplete claims and retain last-good data | **System** |
| Rank/deduplicate candidates and populate optional research inbox | **System** |
| Publish approved Level 1 explanations and eligible Level 2 **Under Investigation** cards | **System**, within reviewed method rules |
| Publish a verified but unexplained Level 3 observation without a causal claim (after feature enablement) | **System**, only under the stricter publication contract |
| Approve a novel causal interpretation or publicly attach a private model-suggested explanation | **Creator**, optional and never blocking routine operations |
| Transition/expire public investigations according to predefined tests and cutoffs | **System** |
| Promote/retire a signal in production forecasting | **Creator, informed by evaluation** |
| Investigate rare findings and write optional public commentary | **Creator, when interested** |
| Decide whether audience growth deserves time | **Creator, independently of race-weekend operations** |

### Trade-off hierarchy when constraints conflict

1. **Truthfulness and correct information cutoffs** over a compelling headline.
2. **Safe unattended operation** over extra features or timeliness.
3. **Reader anticipation, clear visual evidence and honest resolution** over more dashboards.
4. **A testable next implication** over retrospective storytelling alone.
5. **Reusable research methods** over hand-crafted race summaries.
6. **Enjoyment and curiosity** over audience-growth obligations.

A candidate feature must answer: **What will we learn? What would disprove it? What can a fan watch for? How will it look and resolve? Can I investigate it from prepared data? Does it add to the next expectation? Can it run without me?**

If it fails the evidence or autonomy test, reduce scope or keep it as a private experiment.

---

## 15. What counts as a successful Trackside Labs release

These are **acceptance conditions**, not vanity metrics:

- An ordinary event flows from baseline to post-session evidence to evaluation with **no required manual weekend action**.
- The site reliably handles missing data, delayed/corrected feeds, sprint formats and a week with no notable finding.
- Readers can see which claims are observed, which are hypotheses and which have demonstrated predictive value; **knowledge level, measurement confidence and publication state are never conflated**.
- Approved mechanism models can quantify conditional effects (or abstain), not just classify anomalies; unfamiliar behaviour does not disable those routine outputs.
- A Level 2 **Under Investigation** episode publishes verified measurements automatically, shows alternative explanations only within approved bounds, and transitions to supported/weakened/unresolved without human action.
- A private Level 3 brief is useful without auto-publishing a causal story; if public Level 3 is enabled later, it publishes **only** an independently checked unexplained observation with no unapproved mechanism claim.
- Readers can distinguish the independent research label from any FIA/stewards investigation; no public finding implies team wrongdoing.
- At least one analysis conveys a nontrivial gain/loss trade-off visually **without a written article**.
- A public **What to Watch** question is framed prospectively, connects to a later resolution and does not claim an unmeasured mechanism.
- Where a Pressure Test is published, the scenario, forecast delta, uncertainty, information cutoff and actual subsequent resolution are explicit. Unsupported sensitivity is suppressed.
- Every featured insight has a recorded comparison, effect size, limitations, evidence state and optional future test.
- Predictions have immutable cutoffs, reproducible versions, a transparent baseline and evaluation against outcomes.
- A novel case can be investigated **later**, without blocking the site or filling the creator's weekend with alerts.
- A selected research candidate opens with reproducible data references, an initial chart, a testable question and a lightweight Python experiment scaffold, without rebuilding the data preparation by hand.
- The creator still wants to explore a new question, even if nobody visits.
- At least one longitudinal **Signal Passport** shows how an idea developed across checkpoints, including the possibility that it was weakened or rejected—not merely a one-off chart.
- The public evidence trail makes it possible to assess false alarms, forecast improvements and inconclusive hypotheses, not only showcase successful predictions.
- New and legacy public views demonstrate **data parity** for predictions, constructors and accuracy on frozen issued-event fixtures before a public cutover.
- The frontend supports modern, re-themeable, responsive visual storytelling without binding to Streamlit, and the public API reads validated artifacts without computing predictions.
- The first signature comparison has a shareable static rendering; if an interactive view is added, invalid lap/session comparisons cannot be selected or silently passed off as valid.
- Public event/insight routes have stable, crawlable summaries, working internal links, an automated sitemap and correct preview metadata; no weekly manual SEO workflow.
- A small real-user test shows whether F1 fans can trace expectation → what to watch → evidence → resolution without creator narration.

**There is no requirement to discover something revolutionary every race.** A well-calibrated forecast, an honest “no meaningful change,” or an experiment that rejects an appealing hypothesis is useful progress.

---

## 16. Decisions preserved and questions left open

### Decisions

| Area | Decision |
|---|---|
| Positioning | Autonomous, visually driven **F1 performance intelligence** built around expectation gaps, trade-offs and forward tests |
| Intellectual emphasis | Roughly 60% performance understanding / 40% predictions; **not a quota** |
| Predictions | Keep qualifying, sprint and race outputs for personal use and public evaluation |
| Differentiation | The **anticipation → public investigation → evidence-based resolution** experience and accumulating approved mechanism knowledge; not the badge, automation or telemetry alone |
| Forecast Pressure Test | Display only grounded model sensitivities/labelled conditions, frozen ahead of the observation and checked afterward |
| Interactive design | One bounded **Performance Trade-off Explorer** with comparable data and a useful static fallback before adding further interactions |
| External context | Source- and cutoff-aware structured evidence; human-curated ambiguous statements and mechanistic interpretations |
| Research Workbench | Prepared reproducible experiment bundle, not a hosted notebook platform |
| Discoverability | Automated indexable pages, shareable deep links and chart previews, without an editorial calendar |
| UX validation | Test real automated investigations with a small number of F1 fans before broadening the interface |
| Routine operations | Unattended, data-readiness-driven, gracefully degrading, versioned and reproducible |
| Human contribution | Hypotheses, causal reasoning, new methods, visual design and optional rare-case investigation |
| Publishing | Level 1 runs autonomously; verified Level 2 may publish **Under Investigation** automatically; narrowly checked Level 3 observations may later publish with **cause unknown**, while new causal stories remain private pending creator approval. No automated public editorial/causal prose |
| Research vs production | Shadow-test new signals; manual promotion; production keeps running independently |
| Public investigation lifecycle | Frozen observation, dated updates, discriminating test, then **supported / weakened / unresolved**; no creator follow-up deadline |
| Discovery loop | Creator explores novel mechanisms and designs visual methods; approved models become autonomous knowledge only when validated and maintainable |
| Homepage | Start with **What to Watch** when defensible; useful on a quiet weekend; no forced surprise or artificial card quotas |
| Technology | **Replace Streamlit public UI** with a maintainable Next.js/TypeScript frontend and a read-only FastAPI/Pydantic boundary; keep Python models, workers and Supabase initially; migrate by parity-tested vertical slices |
| Preserved features | **Qualifying/sprint/race predictions, constructor comparison and accuracy tracker all stay**. Second priority in the product hierarchy, full correctness and availability requirements |
| Prediction defects | Fix reproduced correctness bugs with regression tests; calibrate and gate model changes; do not confuse retrospective scores with issued-forecast accuracy |
| Competition | Learn from F1 Strategy's discipline and other tools; do not copy their strategy product or compete on dashboard count |
| Social | Optional, occasional and outside the site's operating requirements |

### Open research questions (not feature commitments)

- Can a track-specific car-performance *signature* predict future relative pace beyond a recent-form baseline?
- Which straight-line and corner-level differences are stable enough to interpret, given public telemetry and unknown operating conditions?
- When do apparent gains in one part of the lap coincide with measurable losses elsewhere, and are they genuinely coupled or merely correlated?
- Which FP measurements are informative enough to change a pre-qualifying forecast, and which should be ignored?
- Can the detector distinguish a genuine change in competitiveness from an artefact, tyre preparation or ordinary session evolution?
- How often does an experimental signal alter a forecast *usefully*, instead of just making it more dramatic?
- Can each approved finding be presented in one exceptional visual without requiring manual writing?
- Can model sensitivity highlight a future observation that genuinely changes expectations, rather than a merely dramatic what-if?
- Which external facts actually narrow competing technical explanations without leaking information from after the prediction cutoff?
- Which interactive comparison helps a fan understand a trade-off that static visualisation does not?
- Can an approved model estimate conditional Safety Car, tyre-degradation or temperature effects reliably enough for Level 1 publishing, and when must it abstain?
- Can visitors follow a qualified **Under Investigation** case and distinguish a measured deviation from the cause somebody merely suspects?
- Does the public investigation lifecycle improve meaningful return visits and reader comprehension, or merely produce more colourful anomalies?

### External references to revisit, not imitate

- [F1 Strategy](https://f1-strategy.com/) — explicit assumptions, measured consequences, strategy context.
- [F1 Facts](https://f1facts.app/about) — telemetry/model ecosystem and prediction accountability.
- [OpenF1 documentation](https://openf1.org/docs/) — API fields, data availability and data-quality limits.
- [OpenF1 access and usage](https://openf1.org/) — access tiers, rate limits and rights/usage guidance.
- [FastF1 documentation](https://docs.fastf1.dev/) — timing and telemetry data semantics, caching and API behaviour.

**These examples and source terms change. Check them when making implementation or licensing decisions.**

---

## Working principle

**Discover what you find interesting. Teach the system the methods that survive testing. Let it explain established effects, openly investigate credible surprises, and close the evidence loop without needing you every race weekend.**
