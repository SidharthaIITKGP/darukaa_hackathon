"""The LLM-only baseline: the thing the brief bans, represented fairly.

A strawman baseline would make this comparison worthless, so the prompt below
is a genuinely good one. It tells the model it is an expert environmental
scientist, hands it every field the SiteState carries including the units and
the provenance of each figure, and asks in as many words for quantified
effects with intervals, impacted metrics, time horizons, confidence levels
and citations to specific studies, in the same output structure the graph
system produces. Anything the baseline fails to do here it failed to do with
the question put to it as well as we know how to put it.

The baseline is scored with the same metrics as the graph system, with two
differences that are declared rather than hidden:

  citation validity   means "resolves to a source registered in
                      sources.yaml". A citation the baseline gives may be a
                      real and correct paper and still count as unverifiable,
                      because nothing in this system can check it. The number
                      measures verifiability against this registry, not
                      existence in the world.
  grounding coverage  has no critic behind it. The graph system's figure is
                      the fraction of its empirical claims that are either
                      traceable to a propagation result or entailed by a
                      retrieved passage. The baseline has no propagation
                      result, so the nearest honest analogue is used: the
                      fraction of its empirical claims that carry a citation
                      resolving to a registered source. The two numbers are
                      computed differently and the summary says so.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.graph.nodes import STATE_VARIABLES
from src.graph.schemas import Measurement, SiteState

_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _ROOT / "data" / "derived" / "baseline_cache"
_SOURCES_YAML_PATH = _ROOT / "sources.yaml"

# Groq deprecated llama-3.3-70b-versatile on 2026-06-17 and names
# gpt-oss-120b as the recommended replacement, so that is what the baseline
# runs on. Pinned rather than left to an environment variable, because the
# published comparison is only reproducible if the model behind it is named.
DEFAULT_BASELINE_MODEL = "groq/openai/gpt-oss-120b"

# The model the baseline would have used before 2026-06-17. Recorded so the
# summary can say what was replaced and when, rather than leaving a reader to
# wonder why an older write-up quotes a different model.
DEPRECATED_BASELINE_MODEL = "groq/llama-3.3-70b-versatile"
DEPRECATION_DATE = "2026-06-17"

# Wall-clock ceiling on one baseline call. Generous compared with the
# conversational budget in src.agents.nodes, because the baseline is asked
# for a whole report in one shot and cutting it off early would be scoring
# the timeout rather than the model.
BASELINE_TIMEOUT_S = 120.0
BASELINE_MAX_TOKENS = 4096


class BaselineUnavailable(RuntimeError):
    """No cached response and no way to produce one.

    Raised rather than returning empty text, so a missing API key shows up in
    the results as a baseline that did not run rather than as a baseline that
    scored zero on everything.
    """


_SYSTEM = (
    "You are an expert environmental scientist advising on land degradation and "
    "restoration. You have deep knowledge of soil science, agronomy, agroforestry, "
    "hydrology and agricultural biodiversity, and of the meta-analytic literature on "
    "management effects on soil and biodiversity outcomes."
)

_INSTRUCTIONS = """Diagnose this site and recommend interventions.

Requirements, all of which matter:

1. Name the single binding constraint on this site and say why it binds before
   anything else does.
2. Give three or four ranked interventions.
3. Justify each recommendation with at least three distinct environmental
   variables and the causal chain linking them. A recommendation resting on one
   variable is not acceptable.
4. For every recommendation give the quantified effect on each impacted metric
   as a percentage change WITH an uncertainty interval, not a point value.
5. Give a time horizon in years for each effect.
6. Give a confidence level for each recommendation and say what it rests on.
7. Cite specific published studies or assessment reports by author and year for
   every quantitative claim. Do not cite a study you are not confident exists.
8. State any tradeoffs, and any case where you are applying evidence outside the
   context it was gathered in.

Use exactly this structure, and repeat the RECOMMENDATION block per intervention:

DIAGNOSIS
Binding constraint: <variable>
Reasoning: <why>

RECOMMENDATION 1: <name of the practice>
What to do: <concrete action>
Why it works: <causal chain naming the environmental variables involved>
Impacted metrics:
  <metric>  <percent change>  (<interval>)  <time horizon>
Time horizon: <band and years>
Confidence: <level, and what it rests on>
Evidence:
  - <author, year, title or journal>
Caveats:
  - <tradeoffs, extrapolation, uncertainty>
"""


def _measurement_text(measurement: Measurement) -> str:
    if measurement.value is None:
        return f"{measurement.band} (qualitative band, no number given)"
    unit = measurement.unit or ""
    text = f"{measurement.value:g}{'' if unit in ('', 'pH') else ' ' + unit}"
    if unit == "%":
        text = f"{measurement.value:g}%"
    return f"{text} [{measurement.provenance.value}, {measurement.confidence.value} confidence]"


def site_description(site: SiteState) -> str:
    """Every field the SiteState carries, written out for the prompt.

    Deliberately complete, including provenance and confidence per field and
    an explicit list of what is unknown. The baseline is not being tested on
    its ability to guess what it was not told.
    """
    lines = [f"Site id: {site.site_id}"]
    if site.lat is not None and site.lon is not None:
        lines.append(f"Location: latitude {site.lat}, longitude {site.lon}")
    else:
        lines.append("Location: not given")
    if site.crop is not None:
        lines.append(f"Crop: {site.crop}")

    known = site.known()
    if known:
        lines.append("Measured:")
        for field in known:
            measurement: Measurement = getattr(site, field)
            lines.append(f"  {field}: {_measurement_text(measurement)}")
    else:
        lines.append("Measured: nothing has been measured at this site.")

    missing = site.missing()
    if missing:
        lines.append("Unknown (no value available): " + ", ".join(missing))
    return "\n".join(lines)


def build_prompt(site: SiteState) -> str:
    return f"{site_description(site)}\n\n{_INSTRUCTIONS}"


def _cache_path(site_id: str, model: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return CACHE_DIR / f"{site_id}__{slug}.json"


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class BaselineCall(BaseModel):
    """One baseline report and what it cost to produce.

    latency_s is the wall clock of the model call that produced the text,
    carried in the cache so a rerun reports what the baseline actually costs
    rather than what a dictionary lookup costs. Reporting the cache read as
    the baseline's latency would hand it a win it did not earn, on the one
    metric it genuinely does win.
    """

    text: str
    latency_s: float
    cached: bool


def baseline_call(site: SiteState, model: str = DEFAULT_BASELINE_MODEL) -> BaselineCall:
    """One baseline report for one site, cache first.

    Cached responses are keyed on site_id and model, so a rerun costs
    nothing and the published numbers are reproducible from the cache. The
    prompt hash is stored alongside and a mismatch is reported, because a
    cached answer to a prompt that has since been edited is an answer to a
    different question.
    """
    prompt = build_prompt(site)
    path = _cache_path(site.site_id, model)

    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("prompt_hash") != _prompt_hash(prompt):
            print(
                f"  warning: cached baseline for {site.site_id} was produced from a "
                f"different prompt; delete {path} to refresh it."
            )
        return BaselineCall(
            text=str(payload["response"]),
            latency_s=float(payload.get("latency_s", 0.0)),
            cached=True,
        )

    call = _call_model(prompt, model)
    if call is None:
        raise BaselineUnavailable(
            f"No cached baseline for {site.site_id} at model {model}, and the model could "
            f"not be called. Put the provider API key in .env (GROQ_API_KEY for a groq/ "
            f"model) and rerun, or run with --no-baseline."
        )

    text, elapsed = call
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "site_id": site.site_id,
                "model": model,
                "prompt_hash": _prompt_hash(prompt),
                "prompt": prompt,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "latency_s": elapsed,
                "response": text,
            },
            indent=2,
        )
    )
    return BaselineCall(text=text, latency_s=elapsed, cached=False)


def baseline_response(site: SiteState, model: str = DEFAULT_BASELINE_MODEL) -> str:
    """The baseline's report for one site as plain text."""
    return baseline_call(site, model).text


def _provider_key_present(model: str) -> bool:
    provider = model.split("/")[0] if "/" in model else ""
    keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "vertex_ai": "VERTEXAI_PROJECT",
        "bedrock": "AWS_ACCESS_KEY_ID",
    }
    # Presence only. The value is never read into a local, never logged and
    # never written to the cache or to either output artifact.
    if provider in keys:
        return bool(os.environ.get(keys[provider]))
    return True


# A whole eval run is twelve reports of a few thousand tokens each, which is
# several times a free-tier per-minute token allowance, so rate limiting is
# the normal case rather than an error. Retried rather than reported as a
# failed site: a site dropped for a quota is not a finding about the
# baseline, and leaving those sites out would quietly change which twelve the
# comparison is over.
RATE_LIMIT_RETRIES = 6
RATE_LIMIT_DEFAULT_WAIT_S = 30.0
# Added to whatever wait the provider suggests. Its hint is the time until
# the window has room for the last request, and the next one is not smaller.
RATE_LIMIT_WAIT_MARGIN_S = 5.0

_RETRY_AFTER = re.compile(r"try again in ([\d.]+)\s*s", re.I)


def _is_rate_limit(error: Exception) -> bool:
    return "ratelimit" in type(error).__name__.lower() or "rate_limit" in str(error).lower()


def _retry_wait(error: Exception) -> float:
    match = _RETRY_AFTER.search(str(error))
    if match is None:
        return RATE_LIMIT_DEFAULT_WAIT_S
    return float(match.group(1)) + RATE_LIMIT_WAIT_MARGIN_S


def _call_model(prompt: str, model: str) -> tuple[str, float] | None:
    """One completion, with its own wall clock, or None.

    The returned latency times the attempt that succeeded and excludes any
    time spent waiting out a rate limit. Charging a provider quota to the
    baseline's reported latency would misreport what the model costs to run.
    """
    if not _provider_key_present(model):
        return None

    import litellm

    for attempt in range(RATE_LIMIT_RETRIES + 1):
        started = time.perf_counter()
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                timeout=BASELINE_TIMEOUT_S,
                max_tokens=BASELINE_MAX_TOKENS,
                temperature=0.0,
            )
        except Exception as error:  # noqa: BLE001 - retried or reported, not swallowed
            if _is_rate_limit(error) and attempt < RATE_LIMIT_RETRIES:
                wait = _retry_wait(error)
                print(
                    f"  rate limited, waiting {wait:.0f}s "
                    f"(attempt {attempt + 1} of {RATE_LIMIT_RETRIES})"
                )
                time.sleep(wait)
                continue
            print(f"  baseline model call failed: {type(error).__name__}: {error}")
            return None
        return response.choices[0].message.content, time.perf_counter() - started
    return None


# ================================= parsing =================================

# A recommendation header, allowing for the markdown a model wraps it in.
# Deliberately does NOT match a bare numbered line: the models write their
# causal chains as numbered lists, and treating "1. Added organic matter
# increases SOC" as a new recommendation split one report into five and
# scattered its variables and citations across them. A numbered line counts
# as a header only when it is also a markdown heading.
_RECOMMENDATION_HEADER = re.compile(
    r"^\s*(?:#{1,4}\s*)?\**\s*(?:RECOMMENDATION\s+\d+|INTERVENTION\s+\d+)\b\s*[:.\-]?\s*(.*)$",
    re.I,
)
_HEADING_NUMBERED = re.compile(r"^\s*#{1,4}\s*\**\s*\d+[.)]\s*(.+)$")

# Markdown a title arrives wrapped in, stripped so the practice name can be
# matched against the intervention vocabulary.
_MARKDOWN_NOISE = re.compile(r"[*_`#]+")
_PERCENT = re.compile(r"[+-]?\d+(?:\.\d+)?\s*%")
_INTERVAL = re.compile(
    r"\bCI\b|±|\+/-|"
    r"[+-]?\d+(?:\.\d+)?\s*%?\s*(?:to|–|—|-)\s*[+-]?\d+(?:\.\d+)?\s*%",
    re.I,
)

# Claimed citations, in the forms a model actually writes them.
_CITATION_PATTERNS = (
    re.compile(r"\b([A-Z][A-Za-z'’-]+)\s+et\s+al\.?,?\s*\(?((?:19|20)\d{2})\)?"),
    re.compile(r"\b([A-Z][A-Za-z'’-]+)\s+(?:and|&)\s+[A-Z][A-Za-z'’-]+,?\s*\(?((?:19|20)\d{2})\)?"),
    re.compile(r"\b([A-Z]{2,10})\s*\(?((?:19|20)\d{2})\)?"),
    re.compile(r"\b([A-Z][A-Za-z'’-]+),\s*((?:19|20)\d{2})\b"),
)

# Words that look like a citation surname but are section headings or common
# prose. Without this an "Evidence:" line beginning a sentence would be read
# as an author.
_NOT_AUTHORS = {
    "Evidence",
    "Caveats",
    "Recommendation",
    "Diagnosis",
    "Reasoning",
    "Confidence",
    "Impacted",
    "Time",
    "Why",
    "What",
    "Tradeoff",
    "Tradeoffs",
    "Note",
    "Source",
    "Sources",
    "Meta",
    "See",
    "The",
    "This",
    "In",
    "For",
}

# Environmental variables the baseline might name, and the phrasings it uses
# for them. Built from the graph's own state variables so the two systems are
# counted against the same vocabulary, with synonyms added because the
# baseline writes prose rather than node names.
_VARIABLE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "soil_organic_carbon": ("soil organic carbon", "organic carbon", "soil carbon", "soc", "organic matter"),
    "soil_ph": ("ph",),
    "aggregate_stability": ("aggregate stability", "soil structure", "aggregation"),
    "infiltration_rate": ("infiltration",),
    "plant_available_water": ("plant available water", "available water", "soil moisture", "water holding", "water-holding"),
    "soil_moisture_retention": ("moisture retention", "water retention"),
    "microbial_biomass_carbon": ("microbial biomass", "microbial carbon"),
    "mycorrhizal_colonisation": ("mycorrhiz",),
    "soil_fauna_abundance": ("earthworm", "soil fauna", "macrofauna"),
    "nutrient_cycling_rate": ("nutrient cycling", "mineralisation", "mineralization"),
    "nitrogen_availability": ("nitrogen availability", "available nitrogen", "nitrogen supply"),
    "erosion_rate": ("erosion", "soil loss", "runoff"),
    "canopy_cover": ("canopy cover", "ground cover", "vegetation cover"),
    "structural_heterogeneity": ("structural heterogeneity", "habitat structure", "vegetation structure"),
    "habitat_connectivity": ("connectivity", "habitat connect"),
    "pollinator_abundance": ("pollinator",),
    "natural_enemy_abundance": ("natural enem", "predator", "parasitoid", "pest control"),
    "species_richness": ("species richness", "biodiversity", "species diversity"),
    "crop_yield": ("yield",),
    "groundwater_recharge": ("groundwater", "recharge"),
}

assert set(_VARIABLE_SYNONYMS) == set(STATE_VARIABLES), (
    "the baseline variable vocabulary must cover exactly the graph's state variables, "
    "or the two systems are being counted against different denominators"
)


class ParsedRecommendation(BaseModel):
    """One recommendation block.

    quantified holds the lines that assert a percentage, one line to one
    claim, which is the same unit the graph system's quantitative claims are
    counted in. quantified_with_interval is the subset of those lines that
    also stated an uncertainty interval.
    """

    title: str
    body: str
    variables: list[str]
    quantified: list[str]
    quantified_with_interval: list[str]
    citations: list[str]


class ParsedBaseline(BaseModel):
    """Structured view of one baseline report.

    Extraction is regex over the prose. It will miss things a careful reader
    would catch, and every miss counts against the baseline, so the parser is
    deliberately generous: a citation pattern that could be an author is
    counted as a citation, and a number followed by a percent sign is counted
    as a quantified claim even where the surrounding sentence is vague.
    """

    site_id: str
    model: str
    text: str
    recommendations: list[ParsedRecommendation]
    limiting_factor: str | None
    top_intervention: str | None
    citations: list[str]
    resolved_citations: list[str]
    unresolved_citations: list[str]


def _source_matchers() -> list[tuple[str, set[str], str]]:
    """(source_id, name tokens, year) for every registered source.

    The tokens come from the source_id and the citation string, so a
    baseline writing "Joshi et al. (2023)" resolves against
    Joshi_2023_covercrops_SOC without a hand-written alias table.
    """
    data = yaml.safe_load(_SOURCES_YAML_PATH.read_text())
    out: list[tuple[str, set[str], str]] = []
    for source_id, info in data.get("sources", {}).items():
        if not isinstance(info, dict):
            continue
        year_match = re.search(r"(19|20)\d{2}", source_id)
        year = year_match.group(0) if year_match else ""
        tokens = {part.lower() for part in re.split(r"[_\s]+", source_id) if len(part) > 2}
        citation = str(info.get("citation", ""))
        first_word = re.match(r"([A-Za-z]+)", citation)
        if first_word:
            tokens.add(first_word.group(1).lower())
        out.append((source_id, tokens, year))
    return out


_MATCHERS = _source_matchers()


def resolve_claimed_citation(name: str, year: str) -> str | None:
    """The registered source_id a claimed citation refers to, or None.

    A match requires both the name and the year, because an author surname
    alone would resolve half the literature onto whichever registered source
    happens to share it.
    """
    lowered = name.lower()
    for source_id, tokens, source_year in _MATCHERS:
        if source_year and source_year != year:
            continue
        if lowered in tokens:
            return source_id
    return None


def _claimed_citations(text: str) -> list[str]:
    found: list[str] = []
    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(text):
            name, year = match.group(1), match.group(2)
            if name in _NOT_AUTHORS:
                continue
            claim = f"{name} {year}"
            if claim not in found:
                found.append(claim)
    return found


def _variables_in(text: str) -> list[str]:
    lowered = text.lower()
    return [
        variable
        for variable, synonyms in _VARIABLE_SYNONYMS.items()
        if any(synonym in lowered for synonym in synonyms)
    ]


def _clean_title(title: str) -> str:
    return _MARKDOWN_NOISE.sub("", title).strip(" :.-")


def _header_title(line: str) -> str | None:
    """The recommendation title this line announces, or None.

    A title has to read as the name of a practice rather than as a sentence,
    so anything longer than a dozen words is treated as prose that happened
    to begin with a heading marker.
    """
    for pattern in (_RECOMMENDATION_HEADER, _HEADING_NUMBERED):
        match = pattern.match(line)
        if match is None:
            continue
        title = _clean_title(match.group(1))
        if title and len(title.split()) <= 14:
            return title
    return None


def _split_recommendations(text: str) -> list[tuple[str, str]]:
    """(title, body) per recommendation block."""
    blocks: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for line in text.splitlines():
        title = _header_title(line)
        if title is not None:
            if current is not None:
                blocks.append(current)
            current = (title, [])
        elif current is not None:
            current[1].append(line)
    if current is not None:
        blocks.append(current)
    return [(title, "\n".join(body)) for title, body in blocks]


_BINDING = re.compile(r"binding constraint\s*[:\-]?\s*(.+)", re.I)


def _limiting_factor(text: str) -> str | None:
    """The variable the baseline named as binding, mapped to a graph node.

    Mapped through the same synonym table the variable counting uses, so a
    baseline saying "soil moisture" and the graph saying
    plant_available_water are recognised as the same answer rather than
    scored as a miss on wording.
    """
    match = _BINDING.search(text)
    if match is None:
        return None
    stated = match.group(1).strip()
    variables = _variables_in(stated)
    return variables[0] if variables else None


# Practice names the baseline writes, mapped to intervention nodes, so its
# top pick can be compared against the same acceptable sets. Only the node
# names in the graph are targets; a practice the graph does not model (gypsum,
# liming) maps to nothing and is reported as unmapped rather than as a miss.
_INTERVENTION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "legume_cover_crop": ("legume cover", "leguminous cover", "cover crop with legume"),
    "non_legume_cover_crop": ("non-legume cover", "grass cover crop", "cover crop"),
    "crop_rotation_diversification": ("rotation", "diversif"),
    "intercropping": ("intercrop",),
    "alley_cropping": ("alley crop",),
    "boundary_tree_planting": ("boundary tree", "field boundary planting"),
    "hedgerow_planting": ("hedgerow", "hedge planting"),
    "reduced_tillage": ("reduced tillage", "minimum tillage", "conservation tillage"),
    "no_tillage": ("no-till", "no till", "zero till", "zero-till"),
    "residue_retention": ("residue retention", "crop residue", "stubble retention"),
    "mulching": ("mulch",),
    "farmyard_manure": ("farmyard manure", "fym", "manure"),
    "compost_application": ("compost",),
    "biochar_application": ("biochar",),
    "contour_bunding": ("contour bund", "bunding"),
    "contour_trenching": ("contour trench", "trenching"),
    "farm_pond": ("farm pond", "water harvesting pond"),
    "check_dam": ("check dam",),
    "vetiver_grass_strips": ("vetiver", "grass strip", "vegetative barrier"),
    "rotational_grazing": ("rotational grazing", "paddock rotation"),
    "grazing_exclosure": ("exclosure", "grazing exclusion", "destock"),
    "integrated_nutrient_management": ("integrated nutrient", "inm"),
    "agroforestry_silvopasture": ("silvopastur", "silvo-pastur"),
}


def _intervention_of(title: str) -> str | None:
    lowered = title.lower()
    for intervention, synonyms in _INTERVENTION_SYNONYMS.items():
        if any(synonym in lowered for synonym in synonyms):
            return intervention
    return None


def parse_baseline(text: str, site_id: str = "", model: str = "") -> ParsedBaseline:
    """Pull claimed numbers, citations, metrics and variables out of prose."""
    recommendations: list[ParsedRecommendation] = []
    for title, body in _split_recommendations(text):
        block = f"{title}\n{body}"
        # Counted per line, not per figure, so the denominator matches the
        # graph system's: one metric row there is one quantitative claim
        # however many numbers it prints.
        quantified = [line.strip() for line in block.splitlines() if _PERCENT.search(line)]
        with_interval = [line for line in quantified if _INTERVAL.search(line)]
        recommendations.append(
            ParsedRecommendation(
                title=title,
                body=body,
                variables=_variables_in(block),
                quantified=quantified,
                # Counted per line rather than per figure: an interval belongs
                # to the figure stated beside it, and a line carrying both is
                # a quantified claim that reported its uncertainty.
                quantified_with_interval=with_interval,
                citations=_claimed_citations(block),
            )
        )

    citations = _claimed_citations(text)
    resolved: list[str] = []
    unresolved: list[str] = []
    for claim in citations:
        name, year = claim.rsplit(" ", 1)
        source_id = resolve_claimed_citation(name, year)
        if source_id is None:
            unresolved.append(claim)
        else:
            resolved.append(claim)

    top = next(
        (
            intervention
            for intervention in (_intervention_of(r.title) for r in recommendations)
            if intervention is not None
        ),
        None,
    )

    return ParsedBaseline(
        site_id=site_id,
        model=model,
        text=text,
        recommendations=recommendations,
        limiting_factor=_limiting_factor(text),
        top_intervention=top,
        citations=citations,
        resolved_citations=resolved,
        unresolved_citations=unresolved,
    )
