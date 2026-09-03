from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .curation import concentration_to_micromolar, normalize_concentration_unit

NUMBER = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
UNIT = (
    r"(?:[uµμ]g\s*/\s*m[lL]|mg\s*/\s*[lL]|ng\s*/\s*m[lL]|mg\s*/\s*m[lL]"
    r"|[uµμ]mol\s*/\s*[lL]|pmol\s*/\s*m[lL]|microM|[uµμ]M|mM|nM)"
)
VALUE_PATTERN = (
    rf"(?P<operator><=|>=|<|>|≤|≥)?\s*(?P<low>{NUMBER})"
    rf"(?:\s*(?:-|–|—|~|to)\s*(?P<high>{NUMBER}))?"
    rf"(?:\s*±\s*{NUMBER})?\s*(?P<unit>{UNIT})"
)
MIC_PATTERN = re.compile(
    rf"\(\s*MIC\s*(?:=|:)?\s*{VALUE_PATTERN}[^)]*\)",
    re.IGNORECASE,
)
HC50_PATTERN = re.compile(
    rf"\bHC\s*50\s*(?:=|:)?\s*{VALUE_PATTERN}",
    re.IGNORECASE,
)
MIC_MENTION = re.compile(r"\bMIC\b", re.IGNORECASE)
HC50_MENTION = re.compile(r"\bHC\s*50\b", re.IGNORECASE)

RELATIONS = {None: "eq", "<": "lt", "<=": "le", "≤": "le", ">": "gt", ">=": "ge", "≥": "ge"}

NON_BACTERIAL = re.compile(
    r"fung|yeast|candida|asperg|fusarium|cryptococcus|saccharomyces|neurospora|"
    r"alternaria|botrytis|virus|\bhiv\b|protozo|leishmania|tumou?r|cancer|"
    r"cell\s+line|insect|plasmodium|"
    r"\bC\.?\s*(?:albicans|parapsilosis|neoformans|tropicalis|glabrata|krusei)\b|"
    r"\bA\.?\s*(?:fumigatus|flavus|niger|brassicicola)\b|"
    r"\bF\.?\s*(?:oxysporum|culmorum|solani)\b|"
    r"\bS\.?\s*cerevisiae\b|\bB\.?\s*cinerea\b|\bN\.?\s*crassa\b",
    re.IGNORECASE,
)

BACTERIAL_GENERA = frozenset(
    {
        "Acholeplasma",
        "Achromobacter",
        "Acinetobacter",
        "Aerococcus",
        "Aeromonas",
        "Agrobacterium",
        "Alcaligenes",
        "Arthrobacter",
        "Bacillus",
        "Bifidobacterium",
        "Bordetella",
        "Burkholderia",
        "Campylobacter",
        "Citrobacter",
        "Clavibacter",
        "Clostridium",
        "Corynebacterium",
        "Edwardsiella",
        "Enterobacter",
        "Enterococcus",
        "Erwinia",
        "Escherichia",
        "Haemophilus",
        "Helicobacter",
        "Klebsiella",
        "Lactobacillus",
        "Lactococcus",
        "Leuconostoc",
        "Listeria",
        "Micrococcus",
        "Morganella",
        "Moraxella",
        "Mycobacterium",
        "Mycoplasma",
        "Neisseria",
        "Nocardia",
        "Paenibacillus",
        "Pasteurella",
        "Pediococcus",
        "Porphyromonas",
        "Propionibacterium",
        "Proteus",
        "Pseudomonas",
        "Psychrobacter",
        "Rhizobium",
        "Rhodococcus",
        "Ruminococcus",
        "Salmonella",
        "Sarcina",
        "Serratia",
        "Shewanella",
        "Shigella",
        "Spiroplasma",
        "Staphylococcus",
        "Stenotrophomonas",
        "Streptococcus",
        "Ureaplasma",
        "Vibrio",
        "Xanthomonas",
        "Xenorhabdus",
        "Yersinia",
        "Zymomonas",
    }
)

SPECIES_ALIASES = (
    (re.compile(r"\b(?:MRSA|VISA|VRSA|PRSA)\b", re.I), "Staphylococcus aureus"),
    (re.compile(r"\bMDRPA\b", re.I), "Pseudomonas aeruginosa"),
    (re.compile(r"\bVREF\b", re.I), "Enterococcus faecium"),
    (re.compile(r"\bE\.?\s*coli\b", re.I), "Escherichia coli"),
    (re.compile(r"\bS\.?\s*aureus\b", re.I), "Staphylococcus aureus"),
    (re.compile(r"\bP\.?\s*aeruginosa\b", re.I), "Pseudomonas aeruginosa"),
    (re.compile(r"\bK\.?\s*pneumoniae?\b", re.I), "Klebsiella pneumoniae"),
    (re.compile(r"\bA\.?\s*baumannii\b", re.I), "Acinetobacter baumannii"),
    (re.compile(r"\bE\.?\s*faecalis\b", re.I), "Enterococcus faecalis"),
    (re.compile(r"\bE\.?\s*faecium\b", re.I), "Enterococcus faecium"),
    (re.compile(r"\bS\.?\s*epidermidis\b", re.I), "Staphylococcus epidermidis"),
    (re.compile(r"\bM\.?\s*luteus\b", re.I), "Micrococcus luteus"),
    (re.compile(r"\bB\.?\s*subtilis\b", re.I), "Bacillus subtilis"),
    (re.compile(r"\bB\.?\s*cereus\b", re.I), "Bacillus cereus"),
    (re.compile(r"\bL\.?\s*monocytogenes\b", re.I), "Listeria monocytogenes"),
    (re.compile(r"\bS\.?\s*typhimurium\b", re.I), "Salmonella typhimurium"),
    (re.compile(r"\bE\.?\s*cloacae\b", re.I), "Enterobacter cloacae"),
    (re.compile(r"\bS\.?\s*maltophilia\b", re.I), "Stenotrophomonas maltophilia"),
    (re.compile(r"\bP\.?\s*mirabilis\b", re.I), "Proteus mirabilis"),
    (re.compile(r"\bP\.?\s*vulgaris\b", re.I), "Proteus vulgaris"),
    (re.compile(r"\bV\.?\s*cholerae\b", re.I), "Vibrio cholerae"),
    (re.compile(r"\bH\.?\s*pylori\b", re.I), "Helicobacter pylori"),
)

GRAM_POSITIVE_GENERA = frozenset(
    {
        "Bacillus",
        "Clostridium",
        "Corynebacterium",
        "Enterococcus",
        "Lactobacillus",
        "Lactococcus",
        "Leuconostoc",
        "Listeria",
        "Micrococcus",
        "Mycobacterium",
        "Pediococcus",
        "Propionibacterium",
        "Staphylococcus",
        "Streptococcus",
    }
)
GRAM_NEGATIVE_GENERA = frozenset(BACTERIAL_GENERA - GRAM_POSITIVE_GENERA)


@dataclass(frozen=True, slots=True)
class ConcentrationInterval:
    relation: str
    value_original_low: float
    value_original_high: float | None
    unit_original: str
    unit_normalized: str
    lower_um: float | None
    upper_um: float | None

    @property
    def lower_log2_um(self) -> float | None:
        return math.log2(self.lower_um) if self.lower_um is not None else None

    @property
    def upper_log2_um(self) -> float | None:
        return math.log2(self.upper_um) if self.upper_um is not None else None


@dataclass(frozen=True, slots=True)
class Organism:
    name: str
    strain: str | None
    raw: str
    gram: str
    taxonomy_confidence: str
    is_bacterial: bool
    resistance_profile: str | None


@dataclass(frozen=True, slots=True)
class ParsedMIC:
    interval: ConcentrationInterval
    organism: Organism
    span_start: int
    span_end: int


@dataclass(frozen=True, slots=True)
class ParsedHC50:
    interval: ConcentrationInterval
    cell_source: str | None
    span_start: int
    span_end: int


def _interval_from_match(
    match: re.Match[str],
    sequence: str,
    *,
    n_terminal: str,
    c_terminal: str,
) -> ConcentrationInterval:
    operator = match.group("operator")
    low = float(match.group("low"))
    high_raw = match.group("high")
    high = float(high_raw) if high_raw is not None else None
    if high is not None and operator is not None:
        raise ValueError("a concentration range cannot also have a censoring operator")
    if high is not None and high < low:
        raise ValueError("concentration range has descending bounds")
    unit = match.group("unit")
    normalized = normalize_concentration_unit(unit)

    def convert(value: float) -> float:
        return concentration_to_micromolar(
            value,
            unit,
            sequence,
            n_terminal=n_terminal,
            c_terminal=c_terminal,
        )

    if high is not None:
        relation = "interval"
        lower_um, upper_um = convert(low), convert(high)
    else:
        relation = RELATIONS[operator]
        converted = convert(low)
        if relation in {"lt", "le"}:
            lower_um, upper_um = None, converted
        elif relation in {"gt", "ge"}:
            lower_um, upper_um = converted, None
        else:
            lower_um = upper_um = converted
    return ConcentrationInterval(
        relation=relation,
        value_original_low=low,
        value_original_high=high,
        unit_original=unit.strip(),
        unit_normalized=normalized,
        lower_um=lower_um,
        upper_um=upper_um,
    )


def _preceding_clause(text: str, start: int) -> str:
    depth = 0
    index = start - 1
    while index >= 0:
        character = text[index]
        if character == ")":
            depth += 1
        elif character == "(":
            if depth:
                depth -= 1
        elif depth == 0 and character in ",;#":
            break
        index -= 1
    return text[index + 1 : start].strip()


def _clean_organism_label(raw: str) -> str:
    cleaned = re.sub(r"\[\s*Ref[.:]?\s*[^\]]+\]", " ", raw, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;#")
    prefix = re.compile(
        r"^(?:Gram[- ](?:positive|negative)\s+bacter(?:ia|ium)|Human\s+pathogens?|"
        r"Pathogens?|Bacteria|Clinical\s+isolates?|Food[- ]borne\s+pathogens?)\s*:\s*",
        re.I,
    )
    return prefix.sub("", cleaned).strip()


def normalize_organism(raw: str, *, activity: str = "") -> Organism:
    cleaned = _clean_organism_label(raw)
    lower_raw = raw.lower()
    explicit_non_bacterial = bool(NON_BACTERIAL.search(raw))
    species: str | None = None
    match_span: tuple[int, int] | None = None
    confidence = "low"
    for pattern, canonical in SPECIES_ALIASES:
        if match := pattern.search(cleaned):
            species = canonical
            match_span = match.span()
            confidence = "high"
            break
    if species is None:
        corrected = (
            cleaned.replace("Staphylococcuss aureus", "Staphylococcus aureus")
            .replace("Staphylococcus epidermis", "Staphylococcus epidermidis")
            .replace("Psecdomonas aeruginosa", "Pseudomonas aeruginosa")
            .replace("Klebsiella pneumonia ", "Klebsiella pneumoniae ")
        )
        full = re.search(r"\b([A-Z][a-z]{2,})\s+([a-z][a-z-]{2,})\b", corrected)
        if full:
            species = f"{full.group(1)} {full.group(2)}"
            match_span = full.span()
            cleaned = corrected
            confidence = "medium"
    name = species or cleaned or "unknown"
    genus = name.split()[0] if " " in name else ""
    is_bacterial = not explicit_non_bacterial and genus in BACTERIAL_GENERA
    if "gram-negative" in lower_raw or genus in GRAM_NEGATIVE_GENERA:
        gram = "negative"
    elif "gram-positive" in lower_raw or genus in GRAM_POSITIVE_GENERA:
        gram = "positive"
    else:
        gram = "unknown"
    strain: str | None = None
    if match_span is not None:
        tail = cleaned[match_span[1] :].strip(" .,:;()")
        tail = re.sub(r"^(?:strain\s*:?)\s*", "", tail, flags=re.I)
        strain = tail or None
    resistance_terms = re.findall(
        r"\b(?:MDR|MRSA|VISA|VRSA|VRE|Van\s*[AB]|methicillin[- ]resistant|"
        r"carbapenem[- ]resistant)\b",
        raw,
        flags=re.I,
    )
    resistance = ";".join(dict.fromkeys(term.upper() for term in resistance_terms)) or None
    return Organism(
        name=name,
        strain=strain,
        raw=raw.strip(),
        gram=gram,
        taxonomy_confidence=confidence,
        is_bacterial=is_bacterial,
        resistance_profile=resistance,
    )


def parse_mic_entries(
    text: str,
    sequence: str,
    *,
    n_terminal: str,
    c_terminal: str,
    activity: str = "",
) -> tuple[list[ParsedMIC], list[str]]:
    parsed: list[ParsedMIC] = []
    errors: list[str] = []
    for match in MIC_PATTERN.finditer(text):
        try:
            interval = _interval_from_match(
                match,
                sequence,
                n_terminal=n_terminal,
                c_terminal=c_terminal,
            )
            organism_raw = _preceding_clause(text, match.start())
            if not organism_raw:
                raise ValueError("missing organism text before MIC")
            parsed.append(
                ParsedMIC(
                    interval=interval,
                    organism=normalize_organism(organism_raw, activity=activity),
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )
        except ValueError as error:
            errors.append(f"{match.group(0)[:160]}: {error}")
    return parsed, errors


def _cell_source(text: str) -> str | None:
    lowered = text.lower()
    for species in ("human", "rabbit", "rat", "mouse", "sheep", "goat", "horse", "chicken"):
        if species in lowered:
            return f"{species} red blood cells"
    if "red blood" in lowered or "erythrocyte" in lowered or "rbc" in lowered:
        return "red blood cells (species unspecified)"
    return None


def parse_hc50_entries(
    text: str,
    sequence: str,
    *,
    n_terminal: str,
    c_terminal: str,
) -> tuple[list[ParsedHC50], list[str]]:
    parsed: list[ParsedHC50] = []
    errors: list[str] = []
    for match in HC50_PATTERN.finditer(text):
        try:
            parsed.append(
                ParsedHC50(
                    interval=_interval_from_match(
                        match,
                        sequence,
                        n_terminal=n_terminal,
                        c_terminal=c_terminal,
                    ),
                    cell_source=_cell_source(text),
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )
        except ValueError as error:
            errors.append(f"{match.group(0)[:160]}: {error}")
    return parsed, errors


def threshold_label(
    interval: ConcentrationInterval, threshold_um: float, *, higher_is_one: bool = False
) -> str:
    if higher_is_one:
        if interval.lower_um is not None and interval.lower_um >= threshold_um:
            return "1"
        if interval.upper_um is not None and (
            interval.upper_um < threshold_um
            or (interval.upper_um == threshold_um and interval.relation == "lt")
        ):
            return "0"
    else:
        if interval.upper_um is not None and interval.upper_um <= threshold_um:
            return "1"
        if interval.lower_um is not None and (
            interval.lower_um > threshold_um
            or (interval.lower_um == threshold_um and interval.relation == "gt")
        ):
            return "0"
    return ""
