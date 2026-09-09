"""Scoring for one system response, computed from the response itself.

Every number here is read off the response object or off the text the system
actually emitted. Nothing is judged by a language model. That is the point:
an evaluation whose headline numbers come from an LLM judging prose measures
the judge as much as the system, and the two systems being compared here
differ precisely in how much of their output is a model's opinion.

Citation validity means one specific thing: does the cited work exist in
sources.yaml, this project's registry of fifteen documents. For the graph
system that is a tautology by construction, since the renderer refuses to
print a citation string it cannot resolve to a registered source_id. For the
LLM baseline it is not, and the number it produces is a measure of
verifiability against this registry rather than of whether the paper exists
in the world. A baseline citation may be a real, correct, well known paper
and still count as unverifiable here because nothing in this system can check
it. That distinction is stated wherever the number is reported.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.agents.render import CITATION_TO_SOURCE_ID
from src.agents.state import Claim
from src.eval.baseline import ParsedBaseline, resolve_claimed_citation
from src.eval.sites import EvalSite

_SOURCES_YAML_PATH = Path(__file__).resolve().parents[2] / "sources.yaml"


def registered_source_ids() -> set[str]:
    data = yaml.safe_load(_SOURCES_YAML_PATH.read_text())
    return set(data.get("sources", {}).keys())


REGISTERED = registered_source_ids()

# A rendered evidence bullet: "  - [primary] IPCC (2019), Special Report ...".
_EVIDENCE_LINE = re.compile(
    r"^\s*-\s*\[(?:primary|corroborating|contradicting|critique)\]\s+(.*?)\s*$"
)

# An interval, in any of the forms either system produces: a labelled
# confidence interval, a plus-or-minus, or an explicit range of two figures.
_INTERVAL = re.compile(
    r"\bCI\b|±|\+/-|"
    r"[+-]?\d+(?:\.\d+)?\s*%?\s*(?:to|-|–)\s*[+-]?\d+(?:\.\d+)?\s*%",
    re.I,
)


class RecommendationRecord(BaseModel):
    """One recommendation as the eval needs to see it.

    variables is the set of state variables appearing on the causal paths
    behind this recommendation, which is what the brief's multi-variable
    floor is a floor on.
    """

    intervention: str
    variables: list[str]
    tradeoffs: int


class SystemRun(BaseModel):
    """What one site produced when run through the graph system.

    Held as a model rather than as the raw LangGraph state because the state
    carries several megabytes of Monte Carlo detail per site and the eval
    needs a fixed, serialisable subset of it.
    """

    site_id: str
    completed: bool
    error: str | None = None
    latency_s: float
    limiting_factor: str | None = None
    limiting_why: str = ""
    pending_question: str | None = None
    ranked: list[str] = []
    recommendations: list[RecommendationRecord] = []
    claims: list[Claim] = []
    grounding_coverage: float | None = None
    draft: str = ""


class Scores(BaseModel):
    grounding_coverage: float
    mean_variables_per_recommendation: float
    citation_validity: float
    fabricated_citations: int
    limiting_factor_correct: bool | None
    top_intervention_acceptable: bool | None
    required_flags_present: float
    tradeoffs_surfaced: int
    quantified_claims: int
    claims_with_ci: float
    completed: bool
    latency_s: float


# ================================== flags ==================================

# The vocabulary a site's must_flag list draws on. Each detector answers one
# question about the emitted response: did it say this out loud? They are
# deliberately textual for the ones that are textual facts about the output,
# because "the response warned the reader" is a claim about what the reader
# sees, not about what the model computed internally.

# Deliberately narrow. An earlier version matched "contradict", which fires on
# every report that prints a [contradicting] evidence role and would have
# scored the system as flagging an implausible input on sites where it does
# nothing of the kind.
_IMPLAUSIBLE = re.compile(
    r"implausib|not plausible|physically unlikely|cannot both be|"
    r"mutually inconsistent|inconsistent with (?:each other|the other|that)|"
    r"one of these (?:figures|numbers|values) is",
    re.I,
)
_EXTRAPOLATION = re.compile(
    r"extrapolat|outside that scope|outside the scope|beyond the climates|"
    r"scope is '|out of scope",
    re.I,
)
_SPARSE = re.compile(
    r"No measurement available for|nothing measured yet|no measurements|"
    r"rests on no measurement",
    re.I,
)
_DEFAULT_DIAGNOSIS = re.compile(
    r"rather than a finding|by elimination rather than|"
    r"no threshold is breached|default rather than",
    re.I,
)
_NO_TIER_1 = re.compile(
    r"the gate has nothing to admit|no intervention in the graph addresses|"
    r"cannot address (?:this|the) (?:site's )?(?:binding )?constraint",
    re.I,
)


class FlagEvidence(BaseModel):
    """What the flag detectors are allowed to look at.

    Both systems are reduced to this before flags are counted, so the same
    detector runs against the graph system's report and against the
    baseline's prose. Nothing here is specific to how either one is built.
    """

    text: str
    asked_question: bool = False
    corpus_gap: bool = False
    tradeoffs: int = 0


def detect_flags(evidence: FlagEvidence) -> set[str]:
    raised: set[str] = set()
    if _IMPLAUSIBLE.search(evidence.text):
        raised.add("implausible_input")
    if _EXTRAPOLATION.search(evidence.text):
        raised.add("extrapolation")
    if _SPARSE.search(evidence.text) or evidence.asked_question:
        raised.add("sparse_input")
    if _DEFAULT_DIAGNOSIS.search(evidence.text):
        raised.add("default_diagnosis")
    if _NO_TIER_1.search(evidence.text):
        raised.add("no_tier_1")
    if evidence.asked_question:
        raised.add("asks_clarifying_question")
    if evidence.corpus_gap:
        raised.add("corpus_gap")
    if evidence.tradeoffs > 0 or "Tradeoff:" in evidence.text:
        raised.add("tradeoff")
    return raised


FLAG_DETECTORS = (
    "implausible_input",
    "extrapolation",
    "sparse_input",
    "default_diagnosis",
    "no_tier_1",
    "asks_clarifying_question",
    "corpus_gap",
    "tradeoff",
)


def flags_present(must_flag: list[str], raised: set[str]) -> float:
    """Fraction of the required flags the response actually raised.

    A site requiring no flags scores 1.0. That is vacuous rather than
    flattering, and the per-site table says which sites required nothing so
    the average is readable.
    """
    if not must_flag:
        return 1.0
    return sum(1 for flag in must_flag if flag in raised) / len(must_flag)


# =============================== citations ===============================


def emitted_citations(draft: str) -> list[str]:
    """Every citation string the report printed, in order."""
    found: list[str] = []
    for line in draft.splitlines():
        match = _EVIDENCE_LINE.match(line)
        if match is not None:
            found.append(match.group(1))
    return found


def citation_scores(
    citations: list[str], source_ids: list[str], quantified: int
) -> tuple[float, int]:
    """Validity fraction and fabricated count over what was cited.

    A citation counts as valid when it resolves to a source_id registered in
    sources.yaml. Anything else is unverifiable against this registry and is
    counted as fabricated, which for the baseline means "this system cannot
    check it" rather than "this paper does not exist".

    Nothing cited at all is two different situations. A response that
    asserted no numbers had nothing to cite and scores 1.0; a response that
    asserted numbers and cited nothing scores 0.0, because an uncited figure
    is the failure the citation metric exists to catch.
    """
    total = len(citations) + len(source_ids)
    if total == 0:
        return (0.0 if quantified > 0 else 1.0), 0

    valid = 0
    fabricated = 0
    for citation in citations:
        source_id = CITATION_TO_SOURCE_ID.get(citation.strip())
        if source_id is not None and source_id in REGISTERED:
            valid += 1
        else:
            fabricated += 1
    for source_id in source_ids:
        if source_id in REGISTERED:
            valid += 1
        else:
            fabricated += 1
    return valid / total, fabricated


# ================================ scoring ================================


def score_response(site: EvalSite, result: SystemRun) -> Scores:
    """Score one graph-system run against its site's expectations."""
    if not result.completed:
        return failed_scores(result.latency_s)

    quantitative = [c for c in result.claims if c.kind == "quantitative"]
    with_ci = [c for c in quantitative if _INTERVAL.search(c.text)]

    citations = emitted_citations(result.draft)
    claim_source_ids = [sid for c in result.claims for sid in c.source_ids]
    validity, fabricated = citation_scores(citations, claim_source_ids, len(quantitative))

    raised = detect_flags(
        FlagEvidence(
            text=result.draft,
            asked_question=result.pending_question is not None,
            corpus_gap=any(c.category == "corpus_gap" for c in result.claims),
            tradeoffs=sum(r.tradeoffs for r in result.recommendations),
        )
    )

    variables = [len(r.variables) for r in result.recommendations]

    limiting_correct: bool | None = None
    if site.expected_limiting_factor is not None:
        limiting_correct = result.limiting_factor == site.expected_limiting_factor

    top_acceptable: bool | None = None
    if site.expected_top_intervention_class is not None:
        top_acceptable = bool(result.ranked) and (
            result.ranked[0] in site.expected_top_intervention_class
        )

    return Scores(
        grounding_coverage=result.grounding_coverage if result.grounding_coverage is not None else 0.0,
        mean_variables_per_recommendation=(sum(variables) / len(variables)) if variables else 0.0,
        citation_validity=validity,
        fabricated_citations=fabricated,
        limiting_factor_correct=limiting_correct,
        top_intervention_acceptable=top_acceptable,
        required_flags_present=flags_present(site.must_flag, raised),
        tradeoffs_surfaced=sum(r.tradeoffs for r in result.recommendations),
        quantified_claims=len(quantitative),
        claims_with_ci=(len(with_ci) / len(quantitative)) if quantitative else 0.0,
        completed=True,
        latency_s=result.latency_s,
    )


_TRADEOFF_PROSE = re.compile(r"trade[- ]?off", re.I)
_QUESTION_LINE = re.compile(r"\?\s*$")


def baseline_grounding(parsed: ParsedBaseline) -> float:
    """The nearest honest analogue of grounding coverage for prose.

    The graph system's figure is the fraction of its empirical claims that are
    either traceable to a propagation result or entailed by a retrieved
    passage. The baseline has neither a propagation result nor a corpus, so
    the closest same-shaped question is asked instead: what fraction of its
    empirical claims can be traced to something checkable. A quantified claim
    counts as grounded when the recommendation it sits in cites at least one
    source that resolves against sources.yaml; a citation claim counts when it
    resolves.

    The two numbers are not the same computation and must not be presented as
    though they were. Every place this figure is reported says so.
    """
    quantified = 0
    grounded_quantified = 0
    for recommendation in parsed.recommendations:
        resolved_here = any(
            resolve_claimed_citation(*claim.rsplit(" ", 1)) is not None
            for claim in recommendation.citations
        )
        quantified += len(recommendation.quantified)
        if resolved_here:
            grounded_quantified += len(recommendation.quantified)

    scored = quantified + len(parsed.citations)
    if scored == 0:
        return 0.0
    return (grounded_quantified + len(parsed.resolved_citations)) / scored


def score_baseline(site: EvalSite, parsed: ParsedBaseline, latency_s: float) -> Scores:
    """Score a parsed baseline report with the same metrics as the system.

    Two metrics are computed differently of necessity and are labelled in the
    summary: grounding coverage (see baseline_grounding) and citation
    validity, which for the baseline means resolvable against sources.yaml
    rather than printed from it.
    """
    quantified = [line for r in parsed.recommendations for line in r.quantified]
    with_ci = [line for r in parsed.recommendations for line in r.quantified_with_interval]
    variables = [len(r.variables) for r in parsed.recommendations]

    total_citations = len(parsed.resolved_citations) + len(parsed.unresolved_citations)
    if total_citations == 0:
        validity = 0.0 if quantified else 1.0
    else:
        validity = len(parsed.resolved_citations) / total_citations

    raised = detect_flags(
        FlagEvidence(
            text=parsed.text,
            asked_question=any(
                _QUESTION_LINE.search(line) for line in parsed.text.splitlines()
            ),
            corpus_gap=False,
            tradeoffs=len(_TRADEOFF_PROSE.findall(parsed.text)),
        )
    )

    limiting_correct: bool | None = None
    if site.expected_limiting_factor is not None:
        limiting_correct = parsed.limiting_factor == site.expected_limiting_factor

    top_acceptable: bool | None = None
    if site.expected_top_intervention_class is not None:
        top_acceptable = parsed.top_intervention in site.expected_top_intervention_class

    return Scores(
        grounding_coverage=baseline_grounding(parsed),
        mean_variables_per_recommendation=(sum(variables) / len(variables)) if variables else 0.0,
        citation_validity=validity,
        fabricated_citations=len(parsed.unresolved_citations),
        limiting_factor_correct=limiting_correct,
        top_intervention_acceptable=top_acceptable,
        required_flags_present=flags_present(site.must_flag, raised),
        tradeoffs_surfaced=len(_TRADEOFF_PROSE.findall(parsed.text)),
        quantified_claims=len(quantified),
        claims_with_ci=(len(with_ci) / len(quantified)) if quantified else 0.0,
        completed=True,
        latency_s=latency_s,
    )


def failed_scores(latency_s: float) -> Scores:
    """Scores for a run that did not complete.

    Every metric is zero and completed is False. A site that failed is
    reported as a failure and left in the average rather than dropped, so a
    system that crashes on the hard sites cannot look better than one that
    answers them badly.
    """
    return Scores(
        grounding_coverage=0.0,
        mean_variables_per_recommendation=0.0,
        citation_validity=0.0,
        fabricated_citations=0,
        limiting_factor_correct=None,
        top_intervention_acceptable=None,
        required_flags_present=0.0,
        tradeoffs_surfaced=0,
        quantified_claims=0,
        claims_with_ci=0.0,
        completed=False,
        latency_s=latency_s,
    )
