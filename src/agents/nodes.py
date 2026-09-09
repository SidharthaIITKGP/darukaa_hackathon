"""The nodes of the agent state machine.

Division of labour, held to strictly: the graph and the source registry own
every number, ranking and citation. The language model is used in exactly
three places, none of which can invent a fact:

  intake parsing      turning free text into SiteState fields, and only for
                      fields the deterministic parser did not already find.
                      Every value it proposes is range-checked before it is
                      accepted.
  entailment          judging whether a retrieved passage supports a claim.
                      It can only downgrade a claim, never introduce one.
  prose phrasing      wording of a question or a summary.

Every one of those three has a deterministic fallback, because a missing API
key or a dead network must not stop the system. The fallbacks are not
degraded stubs: the intake parser is the primary path and the model only
tops it up, and entailment falls back to the retrieval floor, which is a
real signal rather than an assumption of support.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from src.agents import render
from src.agents.state import (
    MAX_CRITIC_PASSES,
    SUMMARY_EVERY,
    Claim,
    ConversationState,
    Verdict,
    save_site_profile,
)
from src.graph.edges import build_graph
from src.graph.propagate import (
    DELTA_REF,
    TRANSMISSION,
    RankedIntervention,
    limiting_factor,
    rank_interventions,
)
from src.graph.schemas import (
    CausalEdge,
    Confidence,
    Measurement,
    Provenance,
    SiteState,
    implausible_combinations,
)
from src.retrieval.bind import bind_evidence
from src.retrieval.search import RERANK_FLOOR, RetrievedChunk, has_support, search

_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _ROOT / "data" / "cache"

# Model is read from the environment so it is swappable without a code
# change. litellm resolves the provider from the prefix.
DEFAULT_MODEL = "anthropic/claude-sonnet-5"
# Wall-clock ceiling on one model call. A conversational turn that waits
# longer than this has already failed as a conversation.
LLM_TIMEOUT_S = 20.0

# How hard a rate-limited call is retried before the caller falls back. The
# budget is per call and deliberately small: a conversational turn that waits
# out three rate limits has already failed as a conversation, so this buys a
# verdict for a bounded delay rather than waiting indefinitely for one.
LLM_RATE_LIMIT_RETRIES = 3
LLM_RATE_LIMIT_DEFAULT_WAIT_S = 20.0
LLM_RATE_LIMIT_WAIT_MARGIN_S = 2.0
LLM_RATE_LIMIT_MAX_WAIT_S = 45.0

_LLM_RETRY_AFTER = re.compile(r"try again in ([\d.]+)\s*s", re.I)


def _rate_limited(error: Exception) -> bool:
    return "ratelimit" in type(error).__name__.lower() or "rate_limit" in str(error).lower()


def _rate_limit_wait(error: Exception) -> float:
    """How long the provider says to wait, bounded.

    The hint is the time until the window has room for the request that was
    refused, so it is honoured where given rather than replaced with a fixed
    backoff that would either overshoot or retry into the same wall.
    """
    match = _LLM_RETRY_AFTER.search(str(error))
    hinted = (
        float(match.group(1)) + LLM_RATE_LIMIT_WAIT_MARGIN_S
        if match
        else LLM_RATE_LIMIT_DEFAULT_WAIT_S
    )
    return min(hinted, LLM_RATE_LIMIT_MAX_WAIT_S)

# Per-API ceiling in acquire. Deliberately short: the acquire node exists to
# save the user typing, not to be the critical path, so an API that has not
# answered in five seconds is treated as absent and gap analysis asks
# instead. The committed cache under data/cache was fetched with a longer
# ceiling, because prefetching is not on anyone's critical path.
ACQUIRE_TIMEOUT_S = 5.0

# Radius for the GBIF occurrence query, and the facet ceiling for counting
# distinct species inside it. 15km is a compromise: small enough that the
# records plausibly describe the same landscape, large enough that a rural
# Indian site is not empty of records altogether.
GBIF_RADIUS_KM = 15
GBIF_FACET_LIMIT = 1200

# Churn below which a question is not worth asking, and the ceiling on
# questions per conversation. Both are modelling assumptions. 0.15 is just
# under a single swap in a top-3 set (1/3 churn on one sampled value out of
# three), so a field that cannot even flip one recommendation on one
# plausible value does not earn a turn of the user's attention.
CHURN_FLOOR = 0.15
MAX_QUESTIONS = 3

# Monte Carlo sample count for the value-of-information sweep. Lower than
# the 4000 used for a reported ranking, because this measures which top-3
# SET a value produces rather than reporting an effect size, and the set is
# far more stable than the figures inside it.
VOI_SAMPLES = 800

# How many recommendations get evidence bound to them. Binding runs a
# cross-encoder per candidate passage per edge, so this is the latency
# budget for a turn rather than a claim about how many recommendations
# matter.
BIND_TOP_N = 3
# Three recommendations, matching BIND_TOP_N so every rendered recommendation
# has passages bound to it. Three is enough to show the multi-variable
# reasoning and the tier gate, and the fourth cost a quarter of the critic's
# claim budget to say something the first three had already demonstrated.
RENDER_TOP_N = 3


# ============================== language model ==============================


def model_name() -> str:
    return os.environ.get("DARUKAA_MODEL", DEFAULT_MODEL)


def llm_available() -> bool:
    """Whether a model call is worth attempting.

    Deliberately cheap and deliberately not a network probe: it reports
    whether credentials for the configured provider are present at all.
    Every call site treats a None result as "no model", so a wrong answer
    here costs one failed call, not a broken turn.
    """
    name = model_name()
    provider = name.split("/")[0] if "/" in name else ""
    keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "vertex_ai": "VERTEXAI_PROJECT",
        "bedrock": "AWS_ACCESS_KEY_ID",
    }
    if provider in keys:
        return bool(os.environ.get(keys[provider]))
    # Unknown or local provider (ollama, a proxy). Let the call decide.
    return True


def _llm(system: str, user: str, json_only: bool) -> str | None:
    """One model call. Returns None on any failure whatsoever.

    Broad by intent rather than by laziness: this function has exactly one
    contract, which is that the conversation continues. A missing key, a
    rate limit, a timeout and a malformed response are all the same event to
    every caller, and each caller has a deterministic path for it.
    """
    if not llm_available():
        return None

    import litellm

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(LLM_RATE_LIMIT_RETRIES + 1):
        try:
            response = litellm.completion(
                model=model_name(),
                messages=messages,
                timeout=LLM_TIMEOUT_S,
                max_tokens=1024,
                temperature=0.0,
                **({"response_format": {"type": "json_object"}} if json_only else {}),
            )
        except Exception as error:  # noqa: BLE001 - every caller has a fallback
            # A rate limit is a queue, not a failure. Waiting it out matters
            # because the fallback is silent: a lost entailment call leaves
            # the critic counting a passage above the support floor as
            # support without judging whether it entails the claim, so the
            # grounding figure reports a check that never ran. Measured on a
            # batch eval against an 8000 tokens-per-minute allowance, 199 of
            # 220 entailment calls were being lost this way.
            if _rate_limited(error) and attempt < LLM_RATE_LIMIT_RETRIES:
                time.sleep(_rate_limit_wait(error))
                continue
            return None
        return response.choices[0].message.content
    return None


def llm_json(system: str, user: str) -> dict[str, Any] | None:
    """A model call expected to return one JSON object, or None."""
    raw = _llm(system, user, json_only=True)
    if raw is None:
        return None
    text = raw.strip()
    # Some providers wrap JSON in a fenced block even when asked not to.
    fence = re.search(r"\{.*\}", text, re.S)
    if fence is None:
        return None
    try:
        parsed = json.loads(fence.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def llm_text(system: str, user: str) -> str | None:
    raw = _llm(system, user, json_only=False)
    return raw.strip() if raw else None


# ================================== intake ==================================

# Deterministic extraction patterns. The gap between a field's keyword and
# its number may not contain a comma or a semicolon, which is what stops
# "rainfall is low, soil organic carbon 0.3%" from reading 0.3 as the
# rainfall. Ranges are plausibility bounds, not validation of the
# measurement: a value outside them is far more likely to be a mis-parse
# than a real reading, and a mis-parsed number is worse than a missing one.
_GAP = r"[^0-9\n,;]{0,24}?"

_NUMERIC_PATTERNS: list[tuple[str, str, str, tuple[float, float]]] = [
    (
        "soil_organic_carbon_pct",
        r"(?:soil\s+organic\s+(?:carbon|matter)|organic\s+carbon|\bsoc\b)",
        "%",
        (0.01, 20.0),
    ),
    ("annual_rainfall_mm", r"(?:annual\s+)?(?:rainfall|precipitation)", "mm", (10.0, 12000.0)),
    ("ph", r"(?:soil\s+)?\bp\.?h\b", "pH", (2.0, 11.0)),
    ("slope_pct", r"(?:slope|gradient)", "%", (0.0, 100.0)),
    ("clay_pct", r"clay(?:\s+content|\s+fraction)?", "%", (0.0, 100.0)),
    ("sand_pct", r"sand(?:\s+content|\s+fraction)?", "%", (0.0, 100.0)),
    ("mean_temperature_c", r"(?:mean\s+)?(?:temperature|temp)", "C", (-20.0, 55.0)),
    (
        "observed_species_richness",
        r"(?:species\s+(?:richness|count)|number\s+of\s+species)",
        "count",
        (0.0, 100000.0),
    ),
]

# Qualitative bands. Recorded as a band with LOW confidence, never coerced to
# a number: "rainfall is low" is a real statement about the site and a fake
# millimetre figure derived from it is not.
_BAND_WORDS = r"(?:very\s+low|very\s+high|low|high|moderate|medium|average)"
_BANDABLE = {
    "annual_rainfall_mm": r"(?:annual\s+)?(?:rainfall|precipitation)",
    "soil_organic_carbon_pct": r"(?:soil\s+organic\s+(?:carbon|matter)|organic\s+carbon|\bsoc\b)",
    "slope_pct": r"(?:slope|gradient)",
    "observed_species_richness": r"(?:species\s+richness|biodiversity)",
    "edge_density": r"(?:edge\s+density|fragmentation)",
}

_BAND_NORMAL = {"medium": "moderate", "average": "moderate"}

# Land use vocabulary is deliberately restricted to the bands the causal
# graph's edge preconditions actually use. Recognising "wasteland" and
# storing it would look like progress while satisfying no precondition, so
# an unrecognised description is reported as unmapped instead.
_LAND_USE_PATTERNS: list[tuple[str, str]] = [
    ("grazing_land", r"grazing|pasture|rangeland|grassland|silvopastur"),
    ("cropland", r"cropland|arable|monocultur|crop\b|crops\b|farmland|field"),
]

_CROPS = [
    "wheat",
    "rice",
    "paddy",
    "maize",
    "corn",
    "sorghum",
    "millet",
    "pearl millet",
    "finger millet",
    "cotton",
    "sugarcane",
    "soybean",
    "groundnut",
    "chickpea",
    "pigeonpea",
    "mustard",
    "barley",
    "sunflower",
]

_COORD_BARE = re.compile(r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*[,/]\s*(-?\d{1,3}(?:\.\d+)?)\s*$")
_COORD_INLINE = re.compile(
    r"(?:at|near|coords?|coordinates?|location|lat(?:itude)?)\D{0,14}"
    r"(-?\d{1,2}\.\d+)\s*[,/]\s*(-?\d{1,3}\.\d+)",
    re.I,
)

_CLIMATE_WORDS = re.compile(r"semi[-\s]?arid|arid|sub[-\s]?humid|humid|tropical|temperate", re.I)


class ParsedIntake:
    """What one user message yielded. Not a Pydantic model because it never
    crosses a module boundary: intake_node turns it straight into SiteState
    updates and notes.
    """

    def __init__(self) -> None:
        self.measurements: dict[str, Measurement] = {}
        self.lat: float | None = None
        self.lon: float | None = None
        self.crop: str | None = None
        self.notes: list[str] = []


def _band_note(text: str) -> str | None:
    match = _CLIMATE_WORDS.search(text)
    if match is None:
        return None
    return f"User described the setting as {match.group(0).lower()}."


def parse_free_text(text: str) -> ParsedIntake:
    """Deterministic extraction of SiteState fields from one message.

    This is the primary intake path, not a fallback. A regex that finds
    "0.3" after "soil organic carbon" is more trustworthy than a model
    asked to do the same job, because it cannot round, convert units, or
    fill in a plausible-looking value for something the user never said.
    """
    parsed = ParsedIntake()
    lowered = text.lower()

    as_json = _try_json(text)
    if as_json is not None:
        return as_json

    bare = _COORD_BARE.match(text)
    inline = _COORD_INLINE.search(text)
    pair = bare.groups() if bare else (inline.groups() if inline else None)
    if pair is not None:
        lat, lon = float(pair[0]), float(pair[1])
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            parsed.lat, parsed.lon = lat, lon
            parsed.notes.append(f"Read coordinates {lat}, {lon} from the message.")

    climate_note = _band_note(text)

    for field, keyword, unit, (low, high) in _NUMERIC_PATTERNS:
        value = _numeric_after(lowered, keyword, low, high)
        if value is None:
            continue
        note = climate_note if field == "annual_rainfall_mm" else None
        parsed.measurements[field] = Measurement(
            value=value,
            unit=unit,
            provenance=Provenance.USER_STATED,
            confidence=Confidence.MODERATE,
            note=note,
        )

    for field, keyword in _BANDABLE.items():
        if field in parsed.measurements:
            continue
        band = _band_after(lowered, keyword)
        if band is None:
            continue
        note = climate_note if field == "annual_rainfall_mm" else None
        parsed.measurements[field] = Measurement(
            band=band,
            provenance=Provenance.USER_STATED,
            confidence=Confidence.LOW,
            note=note,
        )

    for band, pattern in _LAND_USE_PATTERNS:
        if re.search(pattern, lowered):
            parsed.measurements["land_use"] = Measurement(
                band=band, provenance=Provenance.USER_STATED, confidence=Confidence.HIGH
            )
            break

    for crop in sorted(_CROPS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(crop)}\b", lowered):
            parsed.crop = crop
            # A named crop is a cropping system even where no land-use word
            # appeared, and every land-use precondition in the graph turns on
            # cropland versus grazing land.
            parsed.measurements.setdefault(
                "land_use",
                Measurement(
                    band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
                ),
            )
            break

    return parsed


def _try_json(text: str) -> ParsedIntake | None:
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    parsed = ParsedIntake()
    field_units = {field: unit for field, _, unit, _ in _NUMERIC_PATTERNS}
    ranges = {field: bounds for field, _, _, bounds in _NUMERIC_PATTERNS}
    for key, value in payload.items():
        if key == "lat" and isinstance(value, (int, float)):
            parsed.lat = float(value)
        elif key == "lon" and isinstance(value, (int, float)):
            parsed.lon = float(value)
        elif key == "crop" and isinstance(value, str):
            parsed.crop = value
        elif key in field_units and isinstance(value, (int, float)):
            low, high = ranges[key]
            if low <= float(value) <= high:
                parsed.measurements[key] = Measurement(
                    value=float(value),
                    unit=field_units[key],
                    provenance=Provenance.USER_STATED,
                    confidence=Confidence.HIGH,
                )
        elif key in ("land_use", "edge_density") and isinstance(value, str):
            parsed.measurements[key] = Measurement(
                band=value, provenance=Provenance.USER_STATED, confidence=Confidence.HIGH
            )
    parsed.notes.append("Read the message as JSON.")
    return parsed


def _numeric_after(lowered: str, keyword: str, low: float, high: float) -> float | None:
    """The number a keyword refers to, forwards then backwards."""
    forward = re.search(rf"{keyword}{_GAP}(-?\d+(?:\.\d+)?)", lowered)
    backward = re.search(rf"(-?\d+(?:\.\d+)?)\s*%?\s*(?:of\s+)?{keyword}", lowered)
    for match in (forward, backward):
        if match is None:
            continue
        value = float(match.group(1))
        if low <= value <= high:
            return value
    return None


def _band_after(lowered: str, keyword: str) -> str | None:
    forward = re.search(rf"{keyword}{_GAP}({_BAND_WORDS})\b", lowered)
    backward = re.search(rf"({_BAND_WORDS})\s+{keyword}", lowered)
    match = forward or backward
    if match is None:
        return None
    band = re.sub(r"\s+", " ", match.group(1).strip())
    return _BAND_NORMAL.get(band, band)


_LLM_INTAKE_SYSTEM = (
    "You extract soil and land measurements from a farmer's message for a land "
    "degradation model. Return one JSON object and nothing else. Allowed keys: "
    + ", ".join(field for field, _, _, _ in _NUMERIC_PATTERNS)
    + ", land_use, crop, lat, lon. Numeric keys take a number in the stated unit "
    "(soil_organic_carbon_pct percent, annual_rainfall_mm millimetres per year, "
    "ph pH units, slope_pct percent, clay_pct percent, sand_pct percent, "
    "mean_temperature_c Celsius, observed_species_richness a count). land_use must "
    "be exactly cropland or grazing_land. Omit any key the message does not state. "
    "Never estimate, never convert a qualitative word like 'low' into a number, and "
    "never infer a value from the region. Omission is correct when in doubt."
)


def llm_top_up(text: str, parsed: ParsedIntake, unknown: Iterable[str]) -> list[str]:
    """Ask the model only for fields the parser missed, and range-check it.

    The model never overrides the deterministic parser and never supplies a
    field the parser already found. Anything it returns outside the
    plausibility range is discarded, so the worst case is a field staying
    unknown and gap analysis asking about it.
    """
    wanted = [f for f in unknown if f not in parsed.measurements]
    if not wanted:
        return []
    payload = llm_json(_LLM_INTAKE_SYSTEM, text)
    if payload is None:
        return []

    ranges = {field: bounds for field, _, _, bounds in _NUMERIC_PATTERNS}
    units = {field: unit for field, _, unit, _ in _NUMERIC_PATTERNS}
    accepted: list[str] = []
    for field, value in payload.items():
        if field == "crop" and parsed.crop is None and isinstance(value, str):
            parsed.crop = value
            accepted.append("crop")
            continue
        if field == "land_use" and "land_use" not in parsed.measurements:
            if value in ("cropland", "grazing_land"):
                parsed.measurements["land_use"] = Measurement(
                    band=value, provenance=Provenance.USER_STATED, confidence=Confidence.LOW
                )
                accepted.append("land_use")
            continue
        if field not in wanted or field not in ranges:
            continue
        if not isinstance(value, (int, float)):
            continue
        low, high = ranges[field]
        if not low <= float(value) <= high:
            continue
        parsed.measurements[field] = Measurement(
            value=float(value),
            unit=units[field],
            provenance=Provenance.USER_STATED,
            # Lower than the parser's own confidence: the model read the
            # value out of prose that the parser could not resolve, so the
            # reading itself is less certain.
            confidence=Confidence.LOW,
            note="Read from free text by the language model, not by the deterministic parser.",
        )
        accepted.append(field)
    return accepted


def intake_node(state: ConversationState) -> dict[str, Any]:
    """Parse the newest user message into SiteState updates.

    Merges rather than replaces, and records the previous SiteState in
    site_history so belief revision can diff against it. A field that
    already had a value and now has a different one is a correction, and is
    reported as revised rather than absorbed silently.
    """
    site: SiteState = state["site"]
    messages = state.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == "user"]
    text = user_messages[-1]["content"] if user_messages else ""

    parsed = parse_free_text(text)
    if text:
        llm_top_up(text, parsed, site.missing())

    previous = site.model_copy(deep=True)
    updated = site.model_copy(deep=True)
    revised: list[str] = []
    filled: list[str] = []

    for field, measurement in parsed.measurements.items():
        existing: Measurement | None = getattr(updated, field)
        if existing is not None and _differs(existing, measurement):
            revised.append(field)
        elif existing is None:
            filled.append(field)
        setattr(updated, field, measurement)

    if parsed.lat is not None and parsed.lon is not None:
        if updated.lat != parsed.lat or updated.lon != parsed.lon:
            updated.lat, updated.lon = parsed.lat, parsed.lon
    if parsed.crop is not None and parsed.crop != updated.crop:
        if updated.crop is not None:
            revised.append("crop")
        updated.crop = parsed.crop

    notes = list(parsed.notes)
    if filled:
        notes.append("Recorded " + ", ".join(_label_fields(filled)) + " from your message.")
    if revised:
        notes.append(
            "Revised " + ", ".join(_label_fields(revised)) + ", which had a previous value."
        )
    if text and not parsed.measurements and parsed.lat is None and parsed.crop is None:
        notes.append("Nothing measurable was parsed from that message.")

    # Revisions accumulate until belief_revision reports them, and the
    # baseline is pinned to the state before the first unreported one. A
    # correction followed by a clarifying question must still produce its
    # diff once the answer comes back.
    pending_revisions = list(dict.fromkeys(list(state.get("revised_fields", [])) + revised))
    baseline = state.get("revision_baseline")
    if revised and baseline is None:
        baseline = previous

    # Internally inconsistent measurements are raised here rather than
    # downstream, because everything downstream reasons FROM them: a
    # diagnosis built on a soil carbon figure that the rainfall cannot
    # support is confidently wrong in a way no later stage can detect. The
    # note goes into the report whatever else happens, and the first
    # unasked one becomes the turn's question, ahead of any
    # value-of-information question, since checking a number is worth more
    # than filling a gap next to it.
    asked = set(state.get("asked_about", []))
    inconsistencies = [note for note in implausible_combinations(updated) if note not in asked]
    notes.extend(inconsistencies)

    update: dict[str, Any] = {
        "site": updated,
        "turn": state.get("turn", 0) + 1,
        "site_history": [previous],
        "revised_fields": pending_revisions,
        "revision_baseline": baseline,
        "notes": notes,
        # A new turn invalidates the previous turn's draft and its verdicts.
        "critic_passes": 0,
        "withdrawn": [],
        "belief_diff": None,
        "pending_question": None,
    }
    if inconsistencies:
        # Recorded in asked_about by its own text, so the same inconsistency
        # is put to the user once. An answer that leaves the figures as they
        # were is still an answer, and re-asking it every turn would be the
        # system arguing with the person who measured the field.
        update["pending_question"] = inconsistencies[0]
        update["asked_about"] = [inconsistencies[0]]
    return update


def _differs(a: Measurement, b: Measurement) -> bool:
    return (a.value, a.band) != (b.value, b.band)


def _label_fields(fields: Iterable[str]) -> list[str]:
    return [render.SITE_FIELD_LABELS.get(f, f.replace("_", " ")) for f in fields]


# ================================= acquire =================================


def _cache_path(api: str, lat: float, lon: float) -> Path:
    return CACHE_DIR / api / f"{lat:.3f}_{lon:.3f}.json"


def _cached(api: str, lat: float, lon: float) -> dict[str, Any] | None:
    path = _cache_path(api, lat, lon)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("response")


def _write_cache(api: str, lat: float, lon: float, url: str, response: dict[str, Any]) -> None:
    path = _cache_path(api, lat, lon)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "url": url,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": response,
    }
    path.write_text(json.dumps(envelope, indent=2))


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "darukaa-earth/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch(
    api: str, url: str, lat: float, lon: float, timeout: float, live: bool
) -> dict[str, Any] | None:
    """Cache first, then live, then nothing.

    Returning None is a first-class outcome, not an error: gap analysis will
    ask the user about whatever stayed unknown, which is a better answer than
    a blocked turn.
    """
    payload = _cached(api, lat, lon)
    if payload is not None:
        return payload
    if not live:
        return None
    try:
        payload = _get_json(url, timeout)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    _write_cache(api, lat, lon, url, payload)
    return payload


SOILGRIDS_PROPERTIES = ("soc", "phh2o", "clay", "sand", "nitrogen", "bdod")


def soilgrids_url(lat: float, lon: float) -> str:
    query = [("lon", f"{lon}"), ("lat", f"{lat}"), ("depth", "0-5cm")]
    query += [("property", p) for p in SOILGRIDS_PROPERTIES]
    query += [("value", "mean"), ("value", "uncertainty")]
    return "https://rest.isric.org/soilgrids/v2.0/properties/query?" + urllib.parse.urlencode(query)


def nasa_power_url(lat: float, lon: float) -> str:
    query = {
        "parameters": "PRECTOTCORR,T2M",
        "community": "AG",
        "longitude": f"{lon}",
        "latitude": f"{lat}",
        "format": "JSON",
    }
    return (
        "https://power.larc.nasa.gov/api/temporal/climatology/point?"
        + urllib.parse.urlencode(query)
    )


def gbif_url(lat: float, lon: float) -> str:
    query = {
        "geoDistance": f"{lat},{lon},{GBIF_RADIUS_KM}km",
        "hasCoordinate": "true",
        "facet": "speciesKey",
        "facetLimit": str(GBIF_FACET_LIMIT),
        "limit": "0",
    }
    return "https://api.gbif.org/v1/occurrence/search?" + urllib.parse.urlencode(query)


# SoilGrids returns integers scaled by d_factor, in the units named by
# mapped_units. Target units and the SiteState field are per property, so the
# conversion is tabulated rather than guessed from the response.
_SOILGRIDS_FIELDS: dict[str, tuple[str, float, str]] = {
    # property -> (SiteState field, multiplier applied after d_factor, unit)
    # soc arrives in g/kg once descaled; 10 g/kg is 1% of soil mass.
    "soc": ("soil_organic_carbon_pct", 0.1, "%"),
    "phh2o": ("ph", 1.0, "pH"),
    "clay": ("clay_pct", 1.0, "%"),
    "sand": ("sand_pct", 1.0, "%"),
    "nitrogen": ("nitrogen", 1.0, "g/kg"),
    "bdod": ("bulk_density", 1.0, "kg/dm3"),
}

# SoilGrids uncertainty above which a point estimate is reported as LOW
# rather than MODERATE confidence. The layer is the ratio of the 90%
# prediction interval width to the median, so 1.0 means the interval is as
# wide as the value itself. A modelling assumption about where a map value
# stops being worth treating as a measurement.
SOILGRIDS_UNCERTAINTY_LIMIT = 1.0


def parse_soilgrids(payload: dict[str, Any]) -> dict[str, Measurement]:
    out: dict[str, Measurement] = {}
    for layer in payload.get("properties", {}).get("layers", []):
        name = layer.get("name")
        if name not in _SOILGRIDS_FIELDS:
            continue
        field, multiplier, unit = _SOILGRIDS_FIELDS[name]
        d_factor = layer.get("unit_measure", {}).get("d_factor", 1) or 1
        for depth in layer.get("depths", []):
            if depth.get("label") != "0-5cm":
                continue
            values = depth.get("values", {})
            mean = values.get("mean")
            if mean is None:
                continue
            uncertainty = values.get("uncertainty")
            ratio = uncertainty / d_factor if uncertainty is not None else None
            out[field] = Measurement(
                value=round(mean / d_factor * multiplier, 3),
                unit=unit,
                provenance=Provenance.API_SOILGRIDS,
                confidence=(
                    Confidence.LOW
                    if ratio is not None and ratio > SOILGRIDS_UNCERTAINTY_LIMIT
                    else Confidence.MODERATE
                ),
                uncertainty=round(ratio, 3) if ratio is not None else None,
                note=(
                    "SoilGrids 250m prediction for 0-5cm, not a soil test. Uncertainty is "
                    "the ratio of the 90% prediction interval width to the median."
                ),
            )
    return out


def parse_nasa_power(payload: dict[str, Any]) -> dict[str, Measurement]:
    parameters = payload.get("properties", {}).get("parameter", {})
    out: dict[str, Measurement] = {}
    precipitation = parameters.get("PRECTOTCORR", {}).get("ANN")
    if precipitation is not None:
        out["annual_rainfall_mm"] = Measurement(
            value=round(precipitation * 365.0, 1),
            unit="mm",
            provenance=Provenance.API_NASA_POWER,
            confidence=Confidence.MODERATE,
            note=(
                "NASA POWER 2001-2020 climatology, annual mean daily precipitation "
                "multiplied by 365. A long-run average, not this season's rainfall."
            ),
        )
    temperature = parameters.get("T2M", {}).get("ANN")
    if temperature is not None:
        out["mean_temperature_c"] = Measurement(
            value=round(temperature, 1),
            unit="C",
            provenance=Provenance.API_NASA_POWER,
            confidence=Confidence.MODERATE,
            note="NASA POWER 2001-2020 climatology, annual mean 2m air temperature.",
        )
    return out


def parse_gbif(payload: dict[str, Any]) -> dict[str, Measurement]:
    facets = payload.get("facets", [])
    counts = next((f.get("counts", []) for f in facets if f.get("field") in ("SPECIES_KEY", "speciesKey")), [])
    if not counts:
        return {}
    return {
        "observed_species_richness": Measurement(
            value=float(len(counts)),
            unit="count",
            provenance=Provenance.API_GBIF,
            confidence=Confidence.LOW,
            note=(
                f"Distinct species with GBIF occurrence records within {GBIF_RADIUS_KM}km. "
                "This counts recording effort as much as richness, so it is a floor on "
                "what is present rather than a survey result."
            ),
        )
    }


def acquire_node(state: ConversationState) -> dict[str, Any]:
    """Fill unknown fields from SoilGrids, NASA POWER and GBIF.

    Only unknown fields are filled. Where the user has already stated a
    value, the API estimate is reported as a cross-check and the user's
    number is kept, because the user stood in the field and the model did
    not.
    """
    site: SiteState = state["site"]
    if site.lat is None or site.lon is None:
        return {}

    lat, lon = site.lat, site.lon
    live = os.environ.get("DARUKAA_OFFLINE", "").strip().lower() not in ("1", "true", "yes")
    # The five-second budget is the conversational one. Prefetching the
    # committed cache is not on anyone's critical path, so it can raise the
    # ceiling: SoilGrids takes about eight seconds for six properties at one
    # point, which is fine to wait for once and not fine to wait for in a
    # conversation.
    timeout = float(os.environ.get("DARUKAA_ACQUIRE_TIMEOUT", ACQUIRE_TIMEOUT_S))

    sources: list[tuple[str, str, Any]] = [
        ("soilgrids", soilgrids_url(lat, lon), parse_soilgrids),
        ("nasa_power", nasa_power_url(lat, lon), parse_nasa_power),
        ("gbif", gbif_url(lat, lon), parse_gbif),
    ]

    updated = site.model_copy(deep=True)
    notes: list[str] = []
    filled: list[str] = []

    # A cross-check is worth stating when it is news: the first time an API
    # is consulted for this site, or when the user has just changed the
    # figure the API disagrees with. Repeating the same disagreement on every
    # later turn would bury the answer under a standing objection.
    _api_provenances = {
        Provenance.API_SOILGRIDS,
        Provenance.API_NASA_POWER,
        Provenance.API_GBIF,
    }
    first_acquisition = not any(
        getattr(site, field).provenance in _api_provenances for field in site.known()
    )
    revised = set(state.get("revised_fields", []))

    for api, url, parser in sources:
        payload = _fetch(api, url, lat, lon, timeout, live)
        if payload is None:
            notes.append(
                f"{api} returned nothing within {timeout:g}s and had no cached "
                f"response for {lat}, {lon}, so nothing was recorded from it."
            )
            continue
        try:
            measurements = parser(payload)
        except (KeyError, TypeError, ValueError) as error:
            notes.append(f"{api} responded in an unexpected shape ({error}); nothing recorded.")
            continue

        for field, measurement in measurements.items():
            existing: Measurement | None = getattr(updated, field, None)
            if existing is None:
                setattr(updated, field, measurement)
                filled.append(field)
                notes.append(_acquired_check(api, field, measurement))
            elif (
                existing.provenance == Provenance.USER_STATED
                and _differs(existing, measurement)
                and (first_acquisition or field in revised)
            ):
                notes.append(_cross_check(api, field, existing, measurement))

    return {"site": updated, "notes": list(state.get("notes", [])) + notes} if filled or notes else {}


def _measurement_text(measurement: Measurement) -> str:
    if measurement.value is None:
        return str(measurement.band)
    unit = measurement.unit or ""
    if unit == "%":
        return f"{measurement.value:g}%"
    if unit in ("", "pH"):
        return f"{measurement.value:g}"
    return f"{measurement.value:g} {unit}"


def _acquired_check(api: str, field: str, measurement: Measurement) -> str:
    """Phrase an acquired value as a question, not as a finding.

    A 250m map cell is not a soil test, and presenting it as one would put a
    number the user never gave into the diagnosis unchallenged.
    """
    label = render.SITE_FIELD_LABELS.get(field, field.replace("_", " "))
    text = f"{api} estimates {label} at {_measurement_text(measurement)}"
    if measurement.uncertainty is not None:
        text += f", with uncertainty ratio {measurement.uncertainty:g}"
    text += f" ({measurement.confidence.value} confidence). Does that match what you see?"
    return text


def _cross_check(api: str, field: str, stated: Measurement, estimated: Measurement) -> str:
    label = render.SITE_FIELD_LABELS.get(field, field.replace("_", " "))
    return (
        f"You gave {label} as {_measurement_text(stated)} and {api} estimates "
        f"{_measurement_text(estimated)}. Your figure is the one being used; the "
        f"difference is worth knowing about."
    )


# =============================== gap analysis ===============================

# Plausible values sampled when measuring how much a field's answer would
# change the recommendation. Chosen to span the range Indian agricultural
# land actually occupies rather than to be centred on any one site: the
# point is to find out whether the answer could matter, so the samples have
# to reach the ends of the range where it would.
_VOI_VALUE_SAMPLES: dict[str, list[tuple[float, str]]] = {
    "soil_organic_carbon_pct": [(0.3, "%"), (0.8, "%"), (1.6, "%")],
    "annual_rainfall_mm": [(300.0, "mm"), (700.0, "mm"), (1500.0, "mm")],
    "ph": [(5.2, "pH"), (6.8, "pH"), (8.4, "pH")],
    "slope_pct": [(1.0, "%"), (5.0, "%"), (15.0, "%")],
    "clay_pct": [(10.0, "%"), (28.0, "%"), (48.0, "%")],
    "sand_pct": [(15.0, "%"), (45.0, "%"), (75.0, "%")],
    "mean_temperature_c": [(18.0, "C"), (27.0, "C")],
}

_VOI_BAND_SAMPLES: dict[str, list[str]] = {
    "land_use": ["cropland", "grazing_land"],
    "edge_density": ["low", "high"],
}

# Fields a person can answer. The landscape metrics are computed from
# geometry rather than reported, and a species count needs a survey, so
# asking about them would be asking for something the site cannot supply in
# a conversation.
ASKABLE_FIELDS = frozenset(
    {
        "soil_organic_carbon_pct",
        "annual_rainfall_mm",
        "ph",
        "slope_pct",
        "clay_pct",
        "sand_pct",
        "mean_temperature_c",
        "land_use",
    }
)

QUESTIONS: dict[str, str] = {
    "soil_organic_carbon_pct": (
        "What is the soil organic carbon or organic matter percentage on this land? A "
        "recent soil test figure is ideal; a rough number is still useful."
    ),
    "annual_rainfall_mm": (
        "Roughly how much rain does this land get in a year, in millimetres? An "
        "average across recent years is what matters, not this season."
    ),
    "ph": "What is the soil pH? A soil test reading, or the range if you have had several.",
    "slope_pct": (
        "How steep is the land, as a percentage or as a rough description (flat, "
        "gently sloping, steep)? This decides whether water-harvesting structures do "
        "anything here."
    ),
    "clay_pct": "Roughly what share of the soil is clay, or is it sandy, loamy or heavy clay?",
    "sand_pct": "Roughly what share of the soil is sand?",
    "mean_temperature_c": "What is the mean annual temperature here, in degrees Celsius?",
    "land_use": "Is this land cropped, or is it grazing land?",
}


def _fields_that_can_matter(graph: nx.MultiDiGraph) -> set[str]:
    """Site fields any edge precondition or the diagnosis actually reads.

    Derived from the graph rather than listed, so adding a precondition on a
    new field brings that field into value-of-information scoring without a
    second edit here. A field nothing reads cannot change a ranking, and
    sweeping it would spend Monte Carlo samples proving that.
    """
    condition_to_field = {
        "rainfall_mm": "annual_rainfall_mm",
        "soc_pct": "soil_organic_carbon_pct",
        "ph": "ph",
        "clay_pct": "clay_pct",
        "slope_pct": "slope_pct",
        "land_use": "land_use",
        "climate_zone": "annual_rainfall_mm",
    }
    fields: set[str] = set()
    for _, _, data in graph.edges(data=True):
        conditions = data["edge"].conditions
        for condition, field in condition_to_field.items():
            if getattr(conditions, condition) is not None:
                fields.add(field)
    # Read directly by limiting_factor, which is a gate ahead of every
    # ranking and so can change the answer on its own.
    fields |= {"annual_rainfall_mm", "soil_organic_carbon_pct", "ph", "slope_pct", "edge_density"}
    return fields


def _top_set(graph: nx.MultiDiGraph, site: SiteState, k: int = 3) -> set[str]:
    ranked = rank_interventions(graph, site, n=VOI_SAMPLES)
    return {item.intervention for item in ranked[:k]}


def _candidate_sites(site: SiteState, field: str) -> list[SiteState]:
    """The site as it would be under each plausible answer for one field."""
    out: list[SiteState] = []
    for value, unit in _VOI_VALUE_SAMPLES.get(field, []):
        candidate = site.model_copy(deep=True)
        setattr(
            candidate,
            field,
            Measurement(
                value=value, unit=unit, provenance=Provenance.USER_STATED, confidence=Confidence.LOW
            ),
        )
        out.append(candidate)
    for band in _VOI_BAND_SAMPLES.get(field, []):
        candidate = site.model_copy(deep=True)
        setattr(
            candidate,
            field,
            Measurement(band=band, provenance=Provenance.USER_STATED, confidence=Confidence.LOW),
        )
        out.append(candidate)
    return out


def value_of_information(
    graph: nx.MultiDiGraph, site: SiteState, fields: Iterable[str], k: int = 3
) -> dict[str, float]:
    """Churn in the top-k recommendation set per unknown field.

    For each field, the site is re-ranked once per plausible value and the
    top-k set is compared with the current one. Churn is
    1 - |intersection| / k, averaged over the sampled values. A field whose
    answer cannot move any recommendation into or out of the top k scores
    zero however interesting it is otherwise, which is the point: the
    question is worth a turn only if the answer would change the advice.
    """
    baseline = _top_set(graph, site, k)
    churn: dict[str, float] = {}
    for field in fields:
        candidates = _candidate_sites(site, field)
        if not candidates:
            continue
        scores = [1.0 - len(baseline & _top_set(graph, candidate, k)) / k for candidate in candidates]
        churn[field] = sum(scores) / len(scores)
    return churn


def gap_analysis_node(state: ConversationState) -> dict[str, Any]:
    """Decide whether to ask, and what.

    Not in field order: fields are ordered by how much their answer would
    change the top-3 recommendation set, so the first question is the one
    whose answer matters most. Stops when nothing left is worth a turn, when
    MAX_QUESTIONS have been spent, or when nothing askable is unknown.
    """
    site: SiteState = state["site"]
    asked = set(state.get("asked_about", []))
    graph = build_graph()

    # A question intake already raised stands. Intake asks only about an
    # internal inconsistency in what it was just told, and checking a figure
    # that may be wrong outranks filling in one that is merely missing:
    # value of information assumes the values it has are true.
    if state.get("pending_question"):
        return {}

    if len(asked) >= MAX_QUESTIONS:
        return {
            "pending_question": None,
            "voi": {},
            "notes": list(state.get("notes", []))
            + [
                f"{MAX_QUESTIONS} questions have been asked, which is the ceiling, so the "
                f"answer below is given on what is known."
            ],
        }

    unknown = [
        field
        for field in site.missing()
        if field in ASKABLE_FIELDS and field not in asked and field in _fields_that_can_matter(graph)
    ]
    if not unknown:
        return {"pending_question": None, "voi": {}}

    churn = value_of_information(graph, site, unknown)
    if not churn:
        return {"pending_question": None, "voi": {}}

    field, best = max(churn.items(), key=lambda item: (item[1], item[0]))
    if best < CHURN_FLOOR:
        return {
            "pending_question": None,
            "voi": churn,
            "notes": list(state.get("notes", []))
            + [
                f"No remaining unknown would change the top three recommendations by more "
                f"than {CHURN_FLOOR:.0%} (best was {render.SITE_FIELD_LABELS.get(field, field)} "
                f"at {best:.0%}), so nothing further is worth asking."
            ],
        }

    question = QUESTIONS[field]
    preface = (
        f"Before recommending anything: of everything still unknown about this site, "
        f"{render.SITE_FIELD_LABELS.get(field, field)} is the answer that would change the "
        f"recommendation most ({best:.0%} churn in the top three)."
    )
    return {
        "pending_question": f"{preface}\n{question}",
        "asked_about": [field],
        "voi": churn,
    }


# =========================== diagnose, plan, bind ===========================


def diagnose_node(state: ConversationState) -> dict[str, Any]:
    site: SiteState = state["site"]
    return {"diagnosis": limiting_factor(site)}


def plan_node(state: ConversationState) -> dict[str, Any]:
    site: SiteState = state["site"]
    graph = build_graph()
    ranked = rank_interventions(graph, site)
    # Persistent memory: the site is written out once it has been ranked, so
    # a later session recognises it without re-asking anything.
    save_site_profile(site)
    return {"ranked": ranked}


def _bindable_edges(
    graph: nx.MultiDiGraph, ranked: list[RankedIntervention]
) -> list[tuple[str, CausalEdge]]:
    """The edges behind the explanation each recommendation will print."""
    out: list[tuple[str, CausalEdge]] = []
    for item in ranked[:BIND_TOP_N]:
        items = render.reportable(item)
        path = render.explanatory_path(items)
        for edge in render.path_edges(graph, path):
            out.append((item.intervention, edge))
    return out


def bind_node(state: ConversationState) -> dict[str, Any]:
    """Attach corpus passages to the edges behind the top recommendations.

    Reranking is on, because a passage bound here becomes a citation.
    has_support separates evidence found from least-bad match returned, so a
    relationship the corpus does not actually discuss is reported as
    unsupported rather than cited against three irrelevant paragraphs.
    """
    site: SiteState = state["site"]
    ranked: list[RankedIntervention] = state["ranked"] or []
    graph = build_graph()

    unique_edges: dict[str, CausalEdge] = {}
    for _intervention, edge in _bindable_edges(graph, ranked):
        unique_edges.setdefault(f"{edge.source}->{edge.target}", edge)

    evidence: dict[str, list[RetrievedChunk]] = {}
    notes: list[str] = []
    for key, edge in unique_edges.items():
        chunks = bind_evidence(edge, site, k=3, precise=False)
        evidence[key] = chunks
        if not has_support(chunks):
            notes.append(
                f"Retrieval found no passage above the support floor for "
                f"{render.label(edge.source)} -> {render.label(edge.target)}. The edge keeps "
                f"its registered citation, but the corpus loaded here does not discuss it, so "
                f"no passage is quoted for it."
            )
    return {"evidence": evidence, "notes": list(state.get("notes", [])) + notes}


# ============================= belief revision =============================


def _rank_positions(ranked: list[RankedIntervention]) -> dict[str, int]:
    return {item.intervention: index + 1 for index, item in enumerate(ranked)}


def belief_revision_node(state: ConversationState) -> dict[str, Any]:
    """Report what changed, not just what is now true.

    Runs only when this turn overwrote a field that already had a value. The
    previous SiteState is re-ranked and the two orderings are diffed, so the
    user sees which recommendation moved and why, rather than a fresh answer
    that quietly contradicts the last one.
    """
    revised = state.get("revised_fields", [])
    history = state.get("site_history", [])
    ranked: list[RankedIntervention] = state["ranked"] or []
    if not revised or not ranked:
        return {"belief_diff": None}

    # The pinned baseline, falling back to the previous SiteState when there
    # is no pinned one, which is the case when the correction and the answer
    # arrived in the same turn.
    previous_site = state.get("revision_baseline") or (history[-1] if history else None)
    if previous_site is None:
        return {"belief_diff": None}

    graph = build_graph()
    previous_ranked = rank_interventions(graph, previous_site)

    before = _rank_positions(previous_ranked)
    after = _rank_positions(ranked)
    moved = [
        (name, before[name], after[name])
        for name in after
        if name in before and before[name] != after[name]
    ]
    moved.sort(key=lambda item: (item[2], -abs(item[1] - item[2])))

    site: SiteState = state["site"]
    old_var, _old_why = limiting_factor(previous_site)
    new_var, new_why = limiting_factor(site)

    lines: list[str] = ["BELIEF REVISION"]
    changes = []
    for field in revised:
        old = getattr(previous_site, field, None)
        new = getattr(site, field, None)
        old_text = _measurement_text(old) if isinstance(old, Measurement) else str(old)
        new_text = _measurement_text(new) if isinstance(new, Measurement) else str(new)
        label = render.SITE_FIELD_LABELS.get(field, field.replace("_", " "))
        changes.append(f"{label} from {old_text} to {new_text}")
    lines.append("You revised " + "; ".join(changes) + ".")

    if old_var != new_var:
        lines.append(
            f"The binding constraint moves from {render.label(old_var)} to "
            f"{render.label(new_var)}. {new_why}"
        )
    else:
        lines.append(
            f"The binding constraint stays {render.label(new_var)}, so the revision changes "
            f"magnitudes and ordering rather than the diagnosis."
        )

    if moved:
        for name, old_rank, new_rank in moved[:5]:
            title = render.ACTIONS.get(name, (name.replace("_", " "), ""))[0]
            direction = "up" if new_rank < old_rank else "down"
            lines.append(f"  - {title} moved {direction} from rank {old_rank} to rank {new_rank}.")
    else:
        lines.append(
            "  - No recommendation changed rank. The revision moved the numbers without "
            "reordering the advice, which is itself worth knowing."
        )
    lines.append("")
    # The diff has been reported, so the pending revision is cleared. Leaving
    # it set would re-report the same correction on every later turn.
    return {"belief_diff": "\n".join(lines), "revised_fields": [], "revision_baseline": None}


# ================================ synthesise ================================


def _evidence_lines(state: ConversationState) -> list[str]:
    evidence: dict[str, list[RetrievedChunk]] = state.get("evidence", {}) or {}
    if not evidence:
        return []
    lines = ["RETRIEVED EVIDENCE"]
    lines.append(
        "Passages retrieved for the edges behind the explanations above, reranked by a "
        "cross-encoder. A passage marked supporting only was found outside the sources "
        "the edge cites: it may inform, but it is not that edge's citation."
    )
    for key, chunks in evidence.items():
        source, target = key.split("->")
        supported = has_support(chunks)
        lines.append(f"  {render.label(source)} -> {render.label(target)}:")
        if not supported:
            lines.append(
                "    No passage cleared the support floor. Nothing is quoted here, and the "
                "relationship rests on its registered citation alone."
            )
            continue
        for chunk in chunks:
            if chunk.rerank_score is None or chunk.rerank_score < RERANK_FLOOR:
                continue
            body = _collapse_whitespace(chunk.body)
            excerpt = body[:280] + ("..." if len(body) > 280 else "")
            marks = []
            if chunk.supporting_only:
                marks.append("supporting only")
            if chunk.out_of_scope:
                marks.append("out of scope")
            mark = f" [{', '.join(marks)}]" if marks else ""
            lines.append(f"    - {chunk.citation} p{chunk.pages}{mark}")
            lines.append(f"      score {chunk.rerank_score:.2f}: {excerpt}")
            if chunk.scope_note:
                lines.append(f"      scope: {chunk.scope_note}")
    lines.append("")
    return lines


def _as_claims(claims: Any) -> list[Claim]:
    """Claims from state, revalidated if the checkpointer flattened them.

    A checkpointer serialises and restores state between nodes, and a model
    it has not been told about comes back as a plain dict. The allowlist in
    src.agents.graph names Claim so that does not happen; this revalidates
    anyway, because a state contract should not depend on a checkpointer's
    configuration being right, and a dict here would otherwise surface as an
    attribute error two nodes later.
    """
    if not claims:
        return []
    return [Claim.model_validate(c) if isinstance(c, dict) else c for c in claims]


def apply_withdrawals(
    draft: str, claims: list[Claim], ranked: list[RankedIntervention] | None = None
) -> str:
    """Remove or mark every claim the critic could not support.

    The two failures are not the same failure, so they are not handled the
    same way. A figure no propagation result produces came from nowhere and
    is dropped outright: softening a fabricated number would leave it on the
    page. A claim that is traceable to the graph but that retrieval could
    not back is kept and marked, because the relationship may well be real
    and deleting the model's own output over a thin corpus would overstate
    the corpus rather than the claim. Either way the withdrawal is listed at
    the end, so a reader sees what happened instead of reading a quietly
    shortened report.
    """
    unsupported = [c for c in claims if c.supported is False]
    if not unsupported:
        return draft

    traceable = _traceable_figures(ranked or [])
    drop_texts: set[str] = set()
    mark_texts: set[str] = set()
    for claim in unsupported:
        figures = _figures(claim.text)
        if claim.kind == "quantitative" and figures and not figures <= traceable:
            drop_texts.add(claim.text)
        else:
            mark_texts.add(claim.text)

    kept: list[str] = []
    for line in draft.splitlines():
        if any(_line_carries(line, text) for text in drop_texts):
            continue
        marks = [text for text in mark_texts if _line_carries(line, text)]
        if marks:
            kept.append(_mark_line(line, marks))
            continue
        kept.append(line)

    kept.append("WITHDRAWN CLAIMS")
    kept.append(
        "The critic could not support the following, so they were dropped from or marked in "
        "the report above rather than left standing."
    )
    for claim in unsupported:
        action = "dropped" if claim.text in drop_texts else "marked unverified"
        kept.append(f"  - [{claim.kind}, {action}] {claim.text}")
        kept.append(f"    {claim.support_note}")
    kept.append("")
    return "\n".join(kept)


_UNVERIFIED_MARK = "[unverified: no retrieved passage cleared the support floor]"


def _mark_line(line: str, claim_texts: list[str]) -> str:
    """Mark a line as unverified, as narrowly as the line allows.

    A "Why it works" paragraph is several mechanism sentences from several
    edges. When one of them has no corpus support and the others do, marking
    the whole paragraph would withdraw support from claims that have it, so
    the mark goes on the sentence when the sentence can be found and on the
    line only when it cannot.
    """
    marked = line
    inline = False
    for text in claim_texts:
        if text and text in marked:
            marked = marked.replace(text, f"{text} {_UNVERIFIED_MARK}", 1)
            inline = True
    return marked if inline else f"{line}  {_UNVERIFIED_MARK}"


_CITATION_CLAIM = re.compile(r" is supported by (.+?)\.?$")


def _line_carries(line: str, claim_text: str) -> bool:
    """Whether a rendered line is the one a claim was taken from.

    Matches on the figures, on the cited work, or on a distinctive prose
    fragment rather than on the whole claim text, because a claim sentence
    is assembled from a rendered line and is not identical to it.
    """
    if claim_text in line:
        return True
    # Equality rather than containment. A metric row's claim sentence keeps
    # every figure the row stated, so the sets match exactly; matching on a
    # subset instead let a single withdrawn figure mark every other line that
    # happened to mention the same number.
    figures = _figures(claim_text)
    if figures and figures == _figures(line):
        return True
    cited = _CITATION_CLAIM.search(claim_text)
    if cited is not None and cited.group(1) in line:
        return True
    fragment = claim_text.strip()[:60]
    return bool(fragment) and fragment in line


def synthesise_node(state: ConversationState) -> dict[str, Any]:
    """Render the answer, using the same code the standalone demo uses."""
    site: SiteState = state["site"]
    ranked: list[RankedIntervention] = state["ranked"] or []
    limiting_var, limiting_why = state["diagnosis"] or limiting_factor(site)
    graph = build_graph()

    sections: list[str] = []
    notes = state.get("notes", []) or []
    if notes:
        sections.append("WHAT I READ AND FETCHED")
        sections.extend(f"  - {note}" for note in notes)
        sections.append("")

    if state.get("belief_diff"):
        sections.append(state["belief_diff"])

    sections.append(
        render.render_report(
            graph, site, ranked, limiting_var, limiting_why, top_n=RENDER_TOP_N
        )
    )

    draft = "\n".join(sections)
    evidence_block = _evidence_lines(state)
    if evidence_block:
        draft = draft + "\n\n" + "\n".join(evidence_block)

    claims = _as_claims(state.get("claims"))
    if claims:
        draft = apply_withdrawals(draft, claims, ranked)

    # The grounding figure is not appended here. The critic is what computes
    # it, and it computes it from this draft, so it appends it to the draft
    # it has finished checking. Appending a previous pass's figure here would
    # report coverage of a draft that no longer exists.
    return {"draft": draft}


# The claim kinds that carry an empirical assertion and so count towards
# coverage. Qualitative prose is excluded because it asserts nothing that
# could be grounded, and counting it as supported would inflate the figure.
GROUNDED_KINDS = ("quantitative", "causal", "citation")


def grounding_coverage(claims: list[Claim]) -> float:
    """Fraction of empirical claims that are grounded.

    Grounded means either traceable to the propagation result that produced
    it or entailed by a retrieved passage. Both are real grounding and
    neither is stronger than the other: they answer different questions
    about different kinds of claim.
    """
    scored = [c for c in claims if c.kind in GROUNDED_KINDS]
    if not scored:
        return 1.0
    grounded = [c for c in scored if c.category in ("traceable", "entailed")]
    return len(grounded) / len(scored)


def coverage_report(claims: list[Claim], coverage: float) -> str:
    """The grounding figure broken into how each claim was settled.

    A bare percentage invites the wrong inference, because the residue is not
    one thing. A claim can fail because its number came from nowhere, which
    is the system's fault and is what the critic exists to catch, or because
    this corpus has no meta-analysis on the topic, which is a fact about the
    twelve documents loaded here. Contour bunding is the standing example:
    the FAO guidelines give the direction and no pooled effect size exists in
    the corpus, so has_support finds nothing above the floor. Reporting those
    two together as "unsupported" would read as weak rigour when the second
    is the opposite: it is the system declining to claim support it does not
    have.
    """
    scored = [c for c in claims if c.kind in GROUNDED_KINDS]
    counts = {
        "traceable": sum(1 for c in scored if c.category == "traceable"),
        "entailed": sum(1 for c in scored if c.category == "entailed"),
        "softened": sum(1 for c in scored if c.category == "softened"),
        "corpus_gap": sum(1 for c in scored if c.category == "corpus_gap"),
    }
    grounded = counts["traceable"] + counts["entailed"]
    lines = [
        f"GROUNDING: {coverage:.0%}  ({grounded} of {len(scored)} claims)",
        f"  traceable to propagation:      {counts['traceable']:>3}",
        f"  entailed by retrieved passage: {counts['entailed']:>3}",
        f"  unsupported, softened:         {counts['softened']:>3}",
        f"  no corpus evidence available:  {counts['corpus_gap']:>3}   <- corpus gap, not a "
        f"failed check",
    ]
    if counts["corpus_gap"]:
        lines.append(
            "  The last line counts claims whose cited source returned no passage above the "
            "cross-encoder support floor. The relationship may well hold; this corpus of "
            "twelve documents does not quantify it, so the claim is marked rather than "
            "asserted. Reading it as a weakness of the reasoning inverts what it records."
        )
    lines.append(
        "  Qualitative framing and instruction prose is excluded from the denominator: it "
        "asserts nothing that could be grounded."
    )
    return "\n".join(lines)


# ================================== critic ==================================

_PERCENT = re.compile(r"[+-]?\d+(?:\.\d+)?%")

# The confidence level in "(90% CI +3.3% to +15.4%)" is a label on the
# interval, not an effect size, so it is removed before figures are
# extracted. Stripping it here rather than requiring a sign on every figure
# keeps an unsigned fabricated number detectable.
_CI_LABEL = re.compile(r"\(\d{2}% CI\b")


def _collapse_whitespace(text: str) -> str:
    """PDF-extracted prose flattened onto one line, for a prompt or excerpt."""
    return re.sub(r"\s+", " ", text).strip()


def _figures(text: str) -> set[str]:
    """Every percentage in a piece of text that asserts an effect size."""
    return set(_PERCENT.findall(_CI_LABEL.sub("(CI", text)))


# Line prefixes the renderer produces. Recognised so a claim can be
# classified by what the renderer meant by the line, and so an unrecognised
# line carrying a number is treated as a claim of unknown provenance rather
# than assumed to be one of ours.
_STRUCTURAL_PREFIXES = (
    "RECOMMENDATION ",
    "What to do:",
    "Why it works:",
    "Impacted metrics:",
    "Time horizon:",
    "Confidence:",
    "Variables in the justification:",
    "Tier 2:",
    "Evidence:",
    "Caveats:",
    "Sequencing:",
)


def _traceable_figures(ranked: list[RankedIntervention]) -> set[str]:
    """Every percentage the propagation engine actually produced.

    A rendered figure is checkable against this exactly, which is the only
    honest test available for a Monte Carlo output: no passage in the corpus
    contains a propagated number, so entailment against a passage cannot
    confirm one. What entailment can do, and does below, is confirm that the
    relationship the figure describes is one the cited sources support.
    """
    figures: set[str] = set()
    for item in ranked:
        if item.constraint_movement is not None:
            figures.add(render.pct(item.constraint_movement))
        for result in item.effects.values():
            figures.add(render.pct(result.p50))
            figures.add(render.pct(result.ci90[0]))
            figures.add(render.pct(result.ci90[1]))
        for tradeoff in item.tradeoffs:
            figures.add(render.pct(tradeoff.positive_effect))
            figures.add(render.pct(tradeoff.negative_effect))
            figures.add(render.pct(abs(tradeoff.negative_effect)))
    return figures


def _block_source_ids(block: list[str]) -> list[str]:
    """The source_ids a recommendation block cites, read off its own output.

    Resolved through the reverse citation map, so a citation string that is
    not registered in sources.yaml resolves to nothing and every claim in
    the block is left uncited rather than attributed to a guess.
    """
    ids: list[str] = []
    for line in block:
        stripped = line.strip()
        if not stripped.startswith("- ["):
            continue
        match = re.match(r"- \[(?:primary|corroborating|contradicting|critique)\]\s+(.*)", stripped)
        if match is None:
            continue
        source_id = render.CITATION_TO_SOURCE_ID.get(match.group(1).strip())
        if source_id and source_id not in ids:
            ids.append(source_id)
    return ids


def _split_blocks(draft: str) -> list[list[str]]:
    """The draft split into recommendation blocks.

    Only recommendation blocks are decomposed into claims. The diagnosis and
    methodology sections state modelling constants and thresholds, which are
    declared assumptions rather than empirical assertions, and running them
    through an entailment check would demand a citation for the system's own
    parameters.
    """
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in draft.splitlines():
        if line.startswith("RECOMMENDATION "):
            if current is not None:
                blocks.append(current)
            current = [line]
        elif current is not None:
            if line.startswith(("SEQUENCING", "METHODOLOGY", "RETRIEVED EVIDENCE", "GROUNDING:")):
                blocks.append(current)
                current = None
            else:
                current.append(line)
    if current is not None:
        blocks.append(current)
    return blocks


def decompose(draft: str) -> list[Claim]:
    """Break a rendered draft into atomic, individually checkable claims.

    A draft on a revision pass may already carry unverified marks from the
    pass before, so those are stripped before decomposition: a mark is the
    critic's own annotation, and letting it into a claim's text would make
    the claim its own commentary and change what gets retrieved for it.
    """
    draft = draft.replace(_UNVERIFIED_MARK, "")
    claims: list[Claim] = []
    for block in _split_blocks(draft):
        source_ids = _block_source_ids(block)
        title = block[0].split(":", 1)[1].strip() if ":" in block[0] else block[0]
        in_metrics = False

        for line in block[1:]:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("Impacted metrics:"):
                in_metrics = True
                continue

            if stripped.startswith("Why it works:"):
                in_metrics = False
                for sentence in _mechanism_sentences(stripped):
                    claims.append(
                        Claim(text=sentence, kind="causal", source_ids=list(source_ids))
                    )
                continue

            if stripped.startswith("- ["):
                in_metrics = False
                match = re.match(
                    r"- \[(primary|corroborating|contradicting|critique)\]\s+(.*)", stripped
                )
                if match is not None:
                    citation = match.group(2).strip()
                    source_id = render.CITATION_TO_SOURCE_ID.get(citation)
                    claims.append(
                        Claim(
                            text=f"{title} is supported by {citation}.",
                            kind="citation",
                            source_ids=[source_id] if source_id else [],
                        )
                    )
                continue

            if in_metrics and _PERCENT.search(stripped):
                claims.append(
                    Claim(
                        text=_metric_sentence(title, stripped),
                        kind="quantitative",
                        source_ids=list(source_ids),
                    )
                )
                continue

            if stripped.startswith("Tier 1:") and _PERCENT.search(stripped):
                claims.append(
                    Claim(text=stripped, kind="quantitative", source_ids=list(source_ids))
                )
                continue

            if stripped.startswith(_STRUCTURAL_PREFIXES) or stripped.startswith("- "):
                in_metrics = False
                claims.append(Claim(text=stripped, kind="qualitative", source_ids=[]))
                continue

            # An unrecognised line carrying a figure did not come from the
            # renderer's own tables. It is treated as a quantitative claim of
            # unknown provenance, which is exactly what has to be checked
            # hardest rather than waved through.
            if _PERCENT.search(stripped):
                claims.append(
                    Claim(text=stripped, kind="quantitative", source_ids=list(source_ids))
                )
            else:
                claims.append(Claim(text=stripped, kind="qualitative", source_ids=[]))
    return claims


def _mechanism_sentences(line: str) -> list[str]:
    body = line.split("Why it works:", 1)[1].strip()
    body = body.split("Causal chain to", 1)[0]
    sentences = [s.strip() for s in re.split(r"(?<=[.])\s+", body) if len(s.strip()) > 30]
    return sentences


def _metric_sentence(title: str, row: str) -> str:
    """One metrics row as a sentence, keeping every figure it stated.

    A sentence retrieves better than a padded table row and reads as a claim
    rather than as a fragment, and keeping the figures verbatim is what makes
    the traceability check possible.
    """
    row = row.replace("<- binding constraint, orders tier 1", "")
    row = row.replace("<- binding constraint, below the reporting floor", "")
    # The renderer pads its columns, so runs of two or more spaces are the
    # column separators: label, median, interval, lag.
    parts = [part for part in re.split(r"\s{2,}", row.strip()) if part]
    label = parts[0] if parts else row.strip()
    median = parts[1] if len(parts) > 1 else ""
    interval = next((part for part in parts[2:] if part.startswith("(")), "")
    lag = parts[-1] if len(parts) > 3 else ""

    sentence = f"{title} changes {label} by {median}".rstrip()
    if interval:
        sentence += f" {interval}"
    if lag:
        sentence += f" over {lag}"
    return sentence + " at this site."


_ENTAILMENT_SYSTEM = (
    "You judge whether retrieved scientific passages support a claim made by a land "
    "degradation model. Answer with one JSON object: "
    '{"verdict": "supported" | "partially" | "unsupported", "reason": "<one line>"}. '
    "The claim's numeric magnitude comes from a Monte Carlo model and will not appear "
    "in the passages; judge whether the passages support the RELATIONSHIP and its "
    "DIRECTION. Answer unsupported when the passages are about a different "
    "relationship, and partially when they support the direction in a different "
    "system, climate or crop than the claim asserts."
)


def check_entailment(claim: Claim, chunks: list[RetrievedChunk]) -> Verdict:
    """Whether the retrieved passages support one claim.

    The retrieval floor is checked first and can settle the question without
    a model call: retrieval always returns k results, so a claim whose best
    passage is below the floor has no support to assess. Where no model is
    configured the floor is the whole verdict, which is a real signal rather
    than an assumption of support, and the note says so.
    """
    if not chunks:
        return Verdict(verdict="unsupported", reason="Retrieval returned nothing for this claim.")
    if not has_support(chunks):
        return Verdict(
            verdict="unsupported",
            reason=(
                "No retrieved passage cleared the cross-encoder support floor, so the corpus "
                "loaded here does not discuss this relationship."
            ),
        )

    passages = "\n\n".join(
        f"[{chunk.citation}] {_collapse_whitespace(chunk.body)[:1200]}" for chunk in chunks[:3]
    )
    payload = llm_json(_ENTAILMENT_SYSTEM, f"CLAIM: {claim.text}\n\nPASSAGES:\n{passages}")
    if payload is None:
        best = max(c.rerank_score for c in chunks if c.rerank_score is not None)
        return Verdict(
            verdict="supported",
            reason=(
                f"Best retrieved passage scores {best:.2f}, above the support floor. No "
                f"entailment model is configured ({model_name()} unavailable), so this rests "
                f"on retrieval alone and is not a judgement that the passage entails the claim."
            ),
        )
    verdict = payload.get("verdict")
    if verdict not in ("supported", "partially", "unsupported"):
        return Verdict(
            verdict="unsupported",
            reason="The entailment model returned an unrecognised verdict, so the claim is not counted as supported.",
        )
    return Verdict(verdict=verdict, reason=str(payload.get("reason", "")).strip() or "No reason given.")


def verify_claim(
    claim: Claim,
    ranked: list[RankedIntervention],
    site: SiteState,
    bound: dict[str, list[RetrievedChunk]] | None = None,
) -> Claim:
    """Verify one claim and return it with a verdict attached.

    Convenience wrapper over the two stages verify_claims runs as a batch.
    Identical in result, and the path a single ad-hoc check takes.
    """
    resolved = resolve_by_traceability(claim, ranked)
    if resolved is not None:
        return resolved
    chunks = _reuse_bound(claim, bound) or search(
        claim.text, k=3, source_ids=claim.source_ids, rerank=True
    )
    return _apply_retrieval_verdict(claim, chunks)


def resolve_by_traceability(claim: Claim, ranked: list[RankedIntervention]) -> Claim | None:
    """Settle a claim without touching the corpus, or return None.

    Three kinds of claim are decided here, and none of them is decidable by
    retrieval:

    A figure the propagation engine did not produce came from nowhere. No
    passage could rescue it, so it is withdrawn on the spot.

    A figure the engine did produce is already verified, because its
    provenance is the graph and not a document: the corpus contains published
    effect sizes, never a Monte Carlo output composed along a path at this
    site. Sending it to retrieval asked a question no passage can answer and
    then counted the silence against it. What the corpus is asked to back is
    the RELATIONSHIP, and that is checked as the mechanism and citation claims
    of the same recommendation, which do go to retrieval. The number carries
    the methodology note instead of a citation.

    A tier statement reports where this system's own gate put an
    intervention. Its figure is checked as above and no paper discusses this
    system's tiering.

    Claims that reach the end of this function are the ones a passage can
    actually bear on, and only those pay for retrieval.
    """
    if claim.kind == "qualitative":
        claim.supported = None
        claim.category = None
        claim.support_note = (
            "Framing or instruction prose, so there is no empirical claim to check. Where such "
            "a line carries a figure it is quoted from a curated mechanism or contested note, "
            "or it is a declared modelling constant, neither of which is a model output."
        )
        return claim

    if claim.kind == "quantitative":
        stray = sorted(_figures(claim.text) - _traceable_figures(ranked))
        if stray:
            claim.supported = False
            claim.category = "softened"
            claim.support_note = (
                f"No propagation result for this site produces {', '.join(stray)}. The figure "
                f"is not traceable to the causal graph, so it is withdrawn rather than cited."
            )
            return claim

        claim.supported = True
        claim.category = "traceable"
        claim.support_note = (
            "Figure traceable to the propagation result for this site. Composed "
            f"multiplicatively along the causal chain with TRANSMISSION={TRANSMISSION:g} and "
            f"DELTA_REF={DELTA_REF:g}; the interval is a Monte Carlo interval, not a published "
            "one. The relationship behind it is checked separately as this recommendation's "
            "mechanism and citation claims."
        )
        return claim

    if not claim.source_ids:
        claim.supported = False
        claim.category = "softened"
        claim.support_note = (
            "The claim cites no source registered in sources.yaml, so there is nothing to "
            "verify it against and it must not be emitted."
        )
        return claim

    return None


def _apply_retrieval_verdict(claim: Claim, chunks: list[RetrievedChunk]) -> Claim:
    """Turn retrieved passages into a verdict on one claim."""
    if not has_support(chunks):
        claim.supported = False
        claim.category = "corpus_gap"
        claim.support_note = (
            "Retrieval restricted to the cited sources found no passage above the support "
            "floor. That is a gap in the corpus loaded here rather than a fault in the claim: "
            "the edge keeps its registered citation and the claim is marked, not asserted."
        )
        return claim

    verdict = check_entailment(claim, chunks)
    claim.supported = verdict.verdict == "supported"
    claim.category = "entailed" if claim.supported else "softened"
    claim.support_note = f"{verdict.verdict}: {verdict.reason}"
    return claim


def verify_claims(
    claims: list[Claim],
    ranked: list[RankedIntervention],
    site: SiteState,
    bound: dict[str, list[RetrievedChunk]] | None = None,
) -> list[Claim]:
    """Verify a whole draft's claims, cheapest resolution first.

    Stage one settles everything traceability can settle: every figure the
    engine produced, every figure it did not, and all prose carrying no
    empirical assertion. None of that needs the corpus, and this is where the
    latency went. On the Deccan site it resolves roughly a third of the
    empirical claims without a single cross-encoder pass.

    Stage two answers what is left, the mechanism and citation claims a
    passage can actually bear on, from the passages bind already retrieved
    where it can and from a fresh reranked search where it cannot. bound is
    bind's output: a mechanism claim is a verbatim edge mechanism and
    bind_evidence already ran a reranked search restricted to that edge's own
    sources. Reuse is not a shortcut past verification, since the support
    floor and the entailment check still run on the passages the edge cites.

    One search per remaining claim, deliberately. Collecting them into a
    single batched cross-encoder call was implemented and measured at 0.93x,
    slightly slower: see the note in src.retrieval.search.
    """
    pending: list[Claim] = []
    for claim in claims:
        if resolve_by_traceability(claim, ranked) is None:
            pending.append(claim)

    to_retrieve: list[Claim] = []
    for claim in pending:
        reused = _reuse_bound(claim, bound)
        if reused is not None:
            _apply_retrieval_verdict(claim, reused)
        else:
            to_retrieve.append(claim)

    for claim in to_retrieve:
        _apply_retrieval_verdict(
            claim, search(claim.text, k=3, source_ids=list(claim.source_ids), rerank=True)
        )

    return claims


def _reuse_bound(
    claim: Claim, bound: dict[str, list[RetrievedChunk]] | None
) -> list[RetrievedChunk] | None:
    """Passages already bound for the edge a mechanism claim came from.

    Matched on the claim sentence appearing in the bound edge's mechanism
    text, which is exact rather than fuzzy: decompose lifted the sentence out
    of that mechanism field verbatim.
    """
    if not bound or claim.kind != "causal":
        return None
    graph = build_graph()
    for key, chunks in bound.items():
        source, target = key.split("->")
        if source not in graph or target not in graph[source]:
            continue
        for _, data in graph[source][target].items():
            if claim.text in data["edge"].mechanism:
                return chunks
            break
    return None


def critic_node(state: ConversationState) -> dict[str, Any]:
    """Decompose the draft, verify every claim, and decide whether to revise.

    grounding_coverage counts quantitative and citation claims only.
    Mechanism prose and instructions are recorded but not scored, because a
    coverage figure that counts unfalsifiable sentences as supported would
    flatter the system rather than measure it.

    Identical claim texts are verified once. The same mechanism sentence and
    the same citation line appear under several recommendations that share an
    edge, and verifying each occurrence separately would spend a
    cross-encoder pass to reach a verdict already reached, then report the
    same claim several times in the coverage figure.
    """
    draft: str = state.get("draft") or ""
    ranked: list[RankedIntervention] = state["ranked"] or []
    site: SiteState = state["site"]
    bound = state.get("evidence", {}) or {}

    seen: set[tuple[str, str]] = set()
    unique: list[Claim] = []
    for claim in decompose(draft):
        key = (claim.kind, claim.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(claim)

    claims = verify_claims(unique, ranked, site, bound)
    coverage = grounding_coverage(claims)

    unsupported = [c for c in claims if c.supported is False]
    passes = state.get("critic_passes", 0)
    # One decision, recorded, rather than the same predicate evaluated here
    # and again in the router. Evaluating it twice was an off-by-one: this
    # node saw the count before its own increment and concluded it would
    # revise, the router saw it after and ended the run, and the grounding
    # figure went with the pass that never happened.
    revising = bool(unsupported) and passes < MAX_CRITIC_PASSES
    update: dict[str, Any] = {
        "claims": claims,
        "grounding_coverage": coverage,
        "withdrawn": [c.text for c in unsupported],
        "revision_pending": revising,
    }
    if revising:
        update["critic_passes"] = passes + 1
    else:
        # Last pass over this draft, so the grounding figure is reported on
        # the draft it was measured against. On a revising pass it is left
        # off, because synthesise is about to replace the draft and a figure
        # describing the old one would be reporting a document nobody reads.
        update["draft"] = draft + "\n" + coverage_report(claims, coverage)

    turn = state.get("turn", 0)
    if turn - state.get("summary_turn", 0) >= SUMMARY_EVERY:
        update["summary"] = rebuild_summary(state)
        update["summary_turn"] = turn
    return update


def needs_revision(state: ConversationState) -> bool:
    """Whether the critic asked for another pass.

    Reads the flag the critic recorded rather than recomputing the decision,
    so the router and the node cannot disagree about whether a revision is
    happening. Falls back to the claim-level check for a state assembled by
    hand without having run the critic.
    """
    if "revision_pending" in state:
        return bool(state["revision_pending"])
    return any(c.supported is False for c in _as_claims(state.get("claims")))


# ============================= episodic memory =============================


def rebuild_summary(state: ConversationState) -> str:
    """A bounded precis of the conversation so far.

    Rebuilt from state rather than from the transcript, so it cannot drift
    from what the system actually believes. The model is asked only to
    tighten the wording; if it is unavailable the deterministic version is
    the summary, which is a list of established facts and is arguably the
    better artifact anyway.
    """
    site: SiteState = state["site"]
    diagnosis = state.get("diagnosis")
    ranked = state.get("ranked") or []
    lines = [
        f"Site {site.site_id} after {state.get('turn', 0)} turns.",
        render.observed_state_line(site),
        f"Asked about: {', '.join(state.get('asked_about', [])) or 'nothing yet'}.",
    ]
    if diagnosis:
        lines.append(f"Binding constraint: {diagnosis[0]}.")
    if ranked:
        top = ", ".join(item.intervention for item in ranked[:3])
        lines.append(f"Top three: {top}.")
    if state.get("grounding_coverage") is not None:
        lines.append(f"Grounding coverage last turn: {state['grounding_coverage']:.0%}.")
    deterministic = " ".join(lines)

    polished = llm_text(
        "Tighten this conversation summary into at most four plain sentences. Change no "
        "number, no site value and no intervention name. Add nothing that is not stated.",
        deterministic,
    )
    return polished or deterministic
