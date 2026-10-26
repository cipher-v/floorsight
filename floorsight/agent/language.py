"""Plain-language rules: a deterministic, offline parser.

    LocalRuleParser(site).parse("tell me when Exit B is blocked for 30 seconds")

Returns a ParseResult with an unconfirmed Rule (or a clear message when the
text can't be turned into a rule). Zone names are matched fuzzily against
both the zone's display name and its id.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from floorsight.contracts import Rule, RuleType, SitePlan, TrackClass, Zone, ZoneKind

PERSON_WORDS = {"anyone", "anybody", "someone", "somebody", "person", "people", "worker", "workers", "pedestrian", "pedestrians", "staff", "human", "humans", "man", "woman", "operator", "picker", "visitor", "everyone"}
FORKLIFT_WORDS = {"forklift", "forklifts", "truck", "trucks", "reach", "pallet-truck", "vehicle", "vehicles"}
OBJECT_WORDS = {"object", "objects", "pallet", "pallets", "box", "boxes", "cart", "trolley", "stuff", "something", "anything"}
CLASS_WORDS = PERSON_WORDS | FORKLIFT_WORDS | OBJECT_WORDS
STOP_WORDS = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is", "be", "for", "me", "if", "when", "my", "our", "we", "it", "zone", "area"}

UNSEEN_PHRASES = ("can't see", "cannot see", "can not see", "can't be seen", "cannot be seen", "don't see", "do not see", "not visible", "out of sight", "unseen", "no camera", "blind spot", "invisible", "lose sight", "lost sight", "lose track", "not seen")
BLOCKED_WORDS = ("blocked", "blocking", "blockage", "obstructed", "obstruction", "obstruct", "block ")
PROXIMITY_WORDS = ("within", "closer than", "close to", "near ", "nearer than", "gets near", "comes near", "approach", "next to")
CLOSING_WORDS = ("closing", "approach", "heading toward", "heading towards", "moving toward", "moving towards", "driving toward")

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50,
    "sixty": 60, "ninety": 90, "half a": 0.5, "half": 0.5,
}
_WORDS = "|".join(sorted((re.escape(w) for w in NUMBER_WORDS), key=len, reverse=True))
# digits may take short units ("5s", "1.5 m"); number words need a space and a full unit ("five seconds")
_DURATION = re.compile(
    r"\b(?:(\d+(?:\.\d+)?)\s*-?\s*(seconds|second|secs|sec|s|minutes|minute|mins|min)"
    r"|(" + _WORDS + r")\s+(seconds|second|minutes|minute))\b"
)
_DISTANCE = re.compile(
    r"\b(?:(\d+(?:\.\d+)?)\s*-?\s*(metres|meters|metre|meter|m|centimetres|centimeters|cm|feet|foot|ft)"
    r"|(" + _WORDS + r")\s+(metres|meters|metre|meter|feet|foot))\b"
)

DEFAULT_SECONDS = {RuleType.blocked: 20.0, RuleType.unseen: 10.0, RuleType.dwell: 0.0}


@dataclass
class ParseResult:
    rule: Rule | None
    message: str
    zone: Zone | None = None
    alternatives: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.rule is not None


def normalise(text: str) -> str:
    s = text.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"\bfork[\s-]+lift", "forklift", s)
    s = re.sub(r"\bcoldroom\b", "cold room", s)
    s = re.sub(r"\bexit[-_]?([a-z0-9])\b", r"exit \1", s)
    s = re.sub(r"[_/]", " ", s)
    s = re.sub(r"[^\w\s'.-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _num(word: str) -> float:
    return float(word) if word[0].isdigit() else float(NUMBER_WORDS[word])


def parse_seconds(text: str) -> float | None:
    m = _DURATION.search(text)
    if not m:
        return None
    value, unit = (_num(m.group(1)), m.group(2)) if m.group(1) else (_num(m.group(3)), m.group(4))
    return value * 60.0 if unit.startswith("m") else value


def parse_metres(text: str) -> float | None:
    for m in _DISTANCE.finditer(text):
        value, unit = (_num(m.group(1)), m.group(2)) if m.group(1) else (_num(m.group(3)), m.group(4))
        if unit in ("cm", "centimetres", "centimeters"):
            return value / 100.0
        if unit in ("feet", "foot", "ft"):
            return round(value * 0.3048, 3)
        return value
    return None


def _singular(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", s)]


def match_zones(site: SitePlan, text: str) -> list[tuple[int, Zone]]:
    """Zones mentioned in the text, best first. Fuzzy on names and ids; class words alone don't count."""
    norm = normalise(text)
    words = [_singular(w) for w in _tokens(norm)]
    joined = " " + " ".join(words) + " "
    scored = []
    for order, zone in enumerate(site.zones):
        matched: set[str] = set()
        for seq in (_tokens(normalise(zone.name)), _tokens(normalise(zone.zone_id))):
            seq = [_singular(w) for w in seq]
            for i, zt in enumerate(seq):
                if zt in STOP_WORDS:
                    continue
                if len(zt) <= 1 or zt.isdigit():
                    # short tokens ("b", "4") only count next to their neighbour ("exit b", "aisle 4")
                    prev = seq[i - 1] if i > 0 else None
                    nxt = seq[i + 1] if i + 1 < len(seq) else None
                    if (prev and f" {prev} {zt} " in joined) or (nxt and f" {zt} {nxt} " in joined):
                        matched.add(zt)
                elif zt in words or (len(zt) > 4 and any(len(w) > 4 and difflib.SequenceMatcher(None, zt, w).ratio() >= 0.85 for w in words)):
                    matched.add(zt)
        if matched - CLASS_WORDS:
            scored.append((len(matched), -order, zone))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [(s[0], s[2]) for s in scored]


def zone_question(zone: Zone | None) -> str:
    if zone is None:
        return "Apply this rule everywhere on the floor?"
    noun = {ZoneKind.lane: "lane", ZoneKind.exit: "exit", ZoneKind.restricted: "room", ZoneKind.area: "area"}[zone.kind]
    return f"Is this the {noun} you mean? ({zone.name})"


def describe(rule: Rule, site: SitePlan) -> str:
    """Normalised one-line description of a parsed rule (shown next to the user's text)."""
    zone = next((z for z in site.zones if z.zone_id == rule.zone_id), None)
    where = zone.name if zone else "anywhere"
    cls = ", ".join(c.value for c in rule.classes) or "anything"
    if rule.type == RuleType.blocked:
        return f"Blocked: static object in {where} for more than {rule.seconds:g} s"
    if rule.type == RuleType.dwell:
        return f"Dwell: {cls} in {where} for more than {rule.seconds:g} s"
    if rule.type == RuleType.unseen:
        return f"Unseen: {cls} inside {where} with no camera view for more than {rule.seconds:g} s"
    closing = f", closing faster than {rule.closing_mps:g} m/s" if rule.closing_mps is not None else ""
    return f"Proximity: {rule.classes[0].value} within {rule.distance_m:g} m of a {rule.classes[1].value} ({where}){closing}"


class LocalRuleParser:
    """Turns plain English into a Rule without any network call."""

    mode = "local"

    def __init__(self, site: SitePlan):
        self.site = site

    def parse(self, text: str) -> ParseResult:
        norm = normalise(text)
        if not norm:
            return ParseResult(None, "The rule is empty. Try e.g. 'tell me when Exit B is blocked for 30 seconds'.")
        words = set(_tokens(norm))
        padded = f" {norm} "
        seconds = parse_seconds(norm)
        metres = parse_metres(norm)
        has_person = bool(words & PERSON_WORDS)
        has_forklift = bool(words & FORKLIFT_WORDS)
        zones = match_zones(self.site, norm)
        zone = zones[0][1] if zones else None
        alternatives = [z.zone_id for s, z in zones[1:] if s == zones[0][0]] if zones else []
        assumptions: list[str] = []

        if any(p in padded for p in UNSEEN_PHRASES):
            rtype = RuleType.unseen
        elif any(w in padded for w in BLOCKED_WORDS):
            rtype = RuleType.blocked
        elif has_forklift and (metres is not None or any(w in padded for w in PROXIMITY_WORDS)) and (has_person or not zone):
            rtype = RuleType.proximity
        elif zone is not None and (has_person or seconds is not None or words & {"stand", "stands", "standing", "stay", "stays", "linger", "loiter", "wait", "waits", "enter", "enters", "inside"}):
            rtype = RuleType.dwell
        else:
            return ParseResult(
                None,
                "I couldn't turn that into a rule. I understand blocked zones ('tell me when Exit B is blocked for 30 seconds'), "
                "people staying in a zone ('alert me if anyone stands in the forklift lane for more than 5 seconds'), "
                "people near forklifts ('warn when a person gets within 1 metre of a forklift') and people we can't see "
                "('alert if someone is in the cold room and we can't see them').",
            )

        if rtype in (RuleType.blocked, RuleType.dwell, RuleType.unseen) and zone is None:
            names = ", ".join(z.name for z in self.site.zones)
            return ParseResult(None, f"Which zone do you mean? I know: {names}.")

        if rtype == RuleType.proximity:
            classes = [TrackClass.person, TrackClass.forklift]
            if metres is None:
                metres = 1.0
                assumptions.append("no distance given, using 1 m")
            closing = 0.2 if any(w in padded for w in CLOSING_WORDS) else None
            rule = Rule(
                rule_id="",
                text=text.strip(),
                type=rtype,
                zone_id=zone.zone_id if zone else None,
                classes=classes,
                distance_m=metres,
                closing_mps=closing,
                source="language",
                confirmed=False,
                enabled=False,
            )
        else:
            if seconds is None:
                seconds = DEFAULT_SECONDS[rtype]
                assumptions.append(f"no duration given, using {seconds:g} s")
            if rtype == RuleType.blocked:
                classes = [TrackClass.object, TrackClass.unknown]
            elif rtype == RuleType.unseen:
                classes = [TrackClass.person, TrackClass.unknown]
            else:
                classes = [TrackClass.forklift] if has_forklift and not has_person and zone.kind != ZoneKind.lane else [TrackClass.person]
            rule = Rule(
                rule_id="",
                text=text.strip(),
                type=rtype,
                zone_id=zone.zone_id,
                classes=classes,
                seconds=seconds,
                source="language",
                confirmed=False,
                enabled=False,
            )
        msg = describe(rule, self.site)
        if assumptions:
            msg += " (" + "; ".join(assumptions) + ")"
        return ParseResult(rule, msg, zone=zone, alternatives=alternatives, assumptions=assumptions)
