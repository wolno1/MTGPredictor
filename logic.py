from __future__ import annotations

import difflib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
META_DECKS_DIR = ROOT_DIR / "meta_decks"
REGISTERED_DECKS_DIR = ROOT_DIR / "registered_decks"
LEARNING_FILE = ROOT_DIR / "learning.json"

SECTION_HEADERS = {
    "deck": "main",
    "main": "main",
    "maindeck": "main",
    "sideboard": "sideboard",
    "companion": "companion",
    "commander": "commander",
}

BASIC_LAND_COLORS = {
    "plains": ["W"],
    "island": ["U"],
    "swamp": ["B"],
    "mountain": ["R"],
    "forest": ["G"],
}

CARD_LINE_RE = re.compile(r"^\s*(\d+)\s*x?\s+(.+?)\s*$", re.IGNORECASE)
SET_CODE_SUFFIX_RE = re.compile(r"\s+\([A-Za-z0-9]{2,8}\)\s+\d+[A-Za-z]?\s*$")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_card_name(raw_name: str) -> str:
    name = raw_name.strip()
    while True:
        cleaned = SET_CODE_SUFFIX_RE.sub("", name).strip()
        if cleaned == name:
            break
        name = cleaned
    return re.sub(r"\s+", " ", name)


def normalize_lookup_key(name: str) -> str:
    return normalize_card_name(name).casefold()


def _safe_stat_value(raw_value: Any, fallback: float = 0.0) -> float:
    if raw_value is None:
        return fallback
    text = str(raw_value).strip()
    if not text:
        return fallback
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return float(text)
    return fallback


@dataclass(slots=True)
class CardInfo:
    name: str
    mana_value: float = 0
    colors: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    text: str = ""
    power: float = 0
    toughness: float = 0


@dataclass(slots=True)
class DeckEntry:
    quantity: int
    name: str
    raw_line: str
    section: str = "main"
    card: Optional[CardInfo] = None
    is_known: bool = False


@dataclass
class Deck:
    name: str
    entries: list[DeckEntry] = field(default_factory=list)
    sideboard: list[DeckEntry] = field(default_factory=list)
    companion: list[DeckEntry] = field(default_factory=list)
    source_path: Optional[Path] = None

    @property
    def total_cards(self) -> int:
        return sum(entry.quantity for entry in self.entries)

    @property
    def all_entries(self) -> list[DeckEntry]:
        return [*self.entries, *self.sideboard, *self.companion]

    @property
    def unknown_cards(self) -> list[str]:
        return sorted({entry.name for entry in self.all_entries if not entry.is_known})


class CardDatabase:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DATA_DIR / "AtomicCards.json"
        self.cards_by_key: dict[str, CardInfo] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        for printed_name, versions in payload.get("data", {}).items():
            if not versions:
                continue

            primary = versions[0]
            mana_value = float(primary.get("manaValue") or primary.get("convertedManaCost") or 0)
            info = CardInfo(
                name=primary.get("name", printed_name),
                mana_value=mana_value,
                colors=list(primary.get("colors") or primary.get("colorIdentity") or []),
                types=list(primary.get("types") or []),
                text=primary.get("text", ""),
                power=_safe_stat_value(primary.get("power"), fallback=max(0.0, mana_value - 1)),
                toughness=_safe_stat_value(primary.get("toughness"), fallback=max(1.0, mana_value)),
            )
            self.cards_by_key[normalize_lookup_key(info.name)] = info

        self._loaded = True

    def lookup(self, name: str) -> Optional[CardInfo]:
        self.load()
        return self.cards_by_key.get(normalize_lookup_key(name))

    def exists(self, name: str) -> bool:
        return self.lookup(name) is not None

    def suggest_names(self, name: str, limit: int = 3) -> list[str]:
        self.load()
        matches = difflib.get_close_matches(normalize_lookup_key(name), self.cards_by_key.keys(), n=limit, cutoff=0.75)
        return [self.cards_by_key[key].name for key in matches]


def _entry_colors(entry: DeckEntry) -> list[str]:
    if entry.card and entry.card.colors:
        return entry.card.colors
    return BASIC_LAND_COLORS.get(entry.name.casefold(), [])


def _parse_cards_as_deck(cards: str | list[str], deck_name: str, card_db: CardDatabase) -> Deck:
    if isinstance(cards, str):
        return parse_decklist(cards, deck_name=deck_name, card_db=card_db)

    normalized = "\n".join(f"1 {normalize_card_name(card)}" for card in cards if normalize_card_name(card))
    return parse_decklist(normalized, deck_name=deck_name, card_db=card_db)


def _entry_has_combat_body(entry: DeckEntry) -> bool:
    if not entry.card:
        return True

    text = (entry.card.text or "").casefold()
    return (
        "Creature" in entry.card.types
        or entry.card.power > 0
        or entry.card.toughness > 0
        or "station" in text
        or "crew" in text
    )


def _estimate_creature_power(entry: DeckEntry) -> float:
    if not entry.card:
        return 1.0 * entry.quantity
    if not _entry_has_combat_body(entry):
        return 0.0

    base_power = entry.card.power if entry.card.power > 0 else max(1.0, entry.card.mana_value - 1)
    text = entry.card.text.casefold()
    if "flying" in text:
        base_power += 0.5
    if "double strike" in text:
        base_power += 1.5
    return base_power * entry.quantity


def _estimate_creature_toughness(entry: DeckEntry) -> float:
    if not entry.card:
        return 1.0 * entry.quantity
    if not _entry_has_combat_body(entry):
        return 0.0

    base_toughness = entry.card.toughness if entry.card.toughness > 0 else max(1.0, entry.card.mana_value)
    text = entry.card.text.casefold()
    if "vigilance" in text:
        base_toughness += 0.25
    if "lifelink" in text:
        base_toughness += 0.25
    return base_toughness * entry.quantity


def analyze_battlefield(
    my_battlefield: str | list[str],
    opponent_battlefield: str | list[str],
    my_life: int,
    opponent_life: int,
    card_db: CardDatabase,
) -> dict[str, Any]:
    my_board = _parse_cards_as_deck(my_battlefield, deck_name="MyBattlefield", card_db=card_db)
    opponent_board = _parse_cards_as_deck(opponent_battlefield, deck_name="OpponentBattlefield", card_db=card_db)

    my_attack = round(sum(_estimate_creature_power(entry) for entry in my_board.entries), 2)
    opponent_attack = round(sum(_estimate_creature_power(entry) for entry in opponent_board.entries), 2)
    my_defense = round(sum(_estimate_creature_toughness(entry) for entry in my_board.entries), 2)
    opponent_defense = round(sum(_estimate_creature_toughness(entry) for entry in opponent_board.entries), 2)

    lethal_available = my_attack >= max(1, opponent_life)
    crackback_risk = opponent_attack >= max(1, my_life)

    if lethal_available:
        combat_recommendation = "Ataque agresivo: tienes letal estimado si el rival no tiene respuesta inmediata."
        danger_level = "medium" if crackback_risk else "low"
    elif crackback_risk:
        combat_recommendation = "Línea defensiva: evita exponerte y prioriza removal o bloqueos porque hay riesgo de crackback letal."
        danger_level = "high"
    elif my_attack > opponent_attack + 1:
        combat_recommendation = "Postura ofensiva favorable: puedes presionar con ataques rentables este turno."
        danger_level = "low"
    elif opponent_attack > my_attack + 1:
        combat_recommendation = "Postura defensiva recomendada: la mesa rival presiona más que la tuya."
        danger_level = "high" if my_life <= 6 else "medium"
    else:
        combat_recommendation = "Mesa equilibrada: busca desarrollar ventaja antes de comprometer un ataque total."
        danger_level = "medium"

    return {
        "my_attack": my_attack,
        "opponent_attack": opponent_attack,
        "my_defense": my_defense,
        "opponent_defense": opponent_defense,
        "lethal_available": lethal_available,
        "crackback_risk": crackback_risk,
        "danger_level": danger_level,
        "combat_recommendation": combat_recommendation,
    }


def recommend_combat_line(
    my_battlefield: str | list[str],
    opponent_battlefield: str | list[str],
    my_life: int,
    opponent_life: int,
    card_db: CardDatabase,
) -> dict[str, Any]:
    summary = analyze_battlefield(
        my_battlefield=my_battlefield,
        opponent_battlefield=opponent_battlefield,
        my_life=my_life,
        opponent_life=opponent_life,
        card_db=card_db,
    )

    if summary["lethal_available"]:
        return {
            "plan": "attack",
            "summary": "Ataca este turno: tienes presión de letal estimada y debes capitalizar la ventaja.",
            "reason": summary["combat_recommendation"],
        }

    if summary["crackback_risk"] or summary["danger_level"] == "high":
        return {
            "plan": "defend",
            "summary": "Bloquea y conserva recursos: el crackback rival puede castigarte si te giras de más.",
            "reason": summary["combat_recommendation"],
        }

    if summary["my_attack"] > summary["opponent_defense"]:
        return {
            "plan": "attack",
            "summary": "Ataca con presión medida: tu mesa supera la defensa rival en este intercambio estimado.",
            "reason": summary["combat_recommendation"],
        }

    return {
        "plan": "hold",
        "summary": "Mantén una postura prudente: desarrolla mesa o guarda interacción antes de atacar abierto.",
        "reason": summary["combat_recommendation"],
    }


def recommend_blocks(
    my_battlefield: str | list[str],
    opponent_attackers: str | list[str],
    my_life: int,
    card_db: CardDatabase,
) -> dict[str, Any]:
    my_board = _parse_cards_as_deck(my_battlefield, deck_name="MyBattlefield", card_db=card_db)
    opponent_board = _parse_cards_as_deck(opponent_attackers, deck_name="OpponentAttackers", card_db=card_db)

    blocker_names = [entry.name for entry in my_board.entries if _entry_has_combat_body(entry)]
    attacker_names = [entry.name for entry in opponent_board.entries if _entry_has_combat_body(entry)]

    if not attacker_names:
        return {
            "plan": "hold",
            "assignments": [],
            "summary": "No se detectaron atacantes rivales para calcular bloqueos.",
        }

    if not blocker_names:
        incoming = round(sum(_estimate_creature_power(entry) for entry in opponent_board.entries), 2)
        plan = "race" if incoming < my_life else "take"
        return {
            "plan": plan,
            "assignments": [],
            "summary": f"No tienes criaturas para bloquear; daño potencial estimado: {incoming}.",
        }

    assignments = choose_block_assignments(
        attackers=attacker_names,
        blockers=blocker_names,
        card_db=card_db,
        defender_life=my_life,
    )

    if not assignments:
        incoming = round(sum(_estimate_creature_power(entry) for entry in opponent_board.entries), 2)
        return {
            "plan": "hold",
            "assignments": [],
            "summary": f"No se encontró un bloqueo rentable claro; daño estimado sin bloquear: {incoming}.",
        }

    lines = [f"{pair['blocker_name']} bloquea a {pair['attacker_name']}" for pair in assignments]
    blocked_attackers = Counter(pair["attacker_name"] for pair in assignments)
    unblocked_damage = 0.0
    for entry in opponent_board.entries:
        blocked_copies = min(entry.quantity, blocked_attackers.get(entry.name, 0))
        unblocked_copies = max(0, entry.quantity - blocked_copies)
        if unblocked_copies:
            unblocked_damage += (_estimate_creature_power(entry) / max(1, entry.quantity)) * unblocked_copies

    double_block = any(count > 1 for count in blocked_attackers.values())
    summary_prefix = "Doble bloqueo sugerido" if double_block else "Bloquea así"

    return {
        "plan": "block",
        "assignments": lines,
        "summary": f"{summary_prefix} para reducir presión; daño estimado tras bloques: {round(unblocked_damage, 2)}.",
    }


def parse_decklist(deck_text: str, deck_name: str = "ImportedDeck", card_db: Optional[CardDatabase] = None) -> Deck:
    deck = Deck(name=deck_name)
    current_section = "main"

    for raw_line in deck_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        header_key = line.rstrip(":").casefold()
        if header_key in SECTION_HEADERS:
            current_section = SECTION_HEADERS[header_key]
            continue

        match = CARD_LINE_RE.match(line)
        if match:
            quantity = int(match.group(1))
            card_name = normalize_card_name(match.group(2))
        else:
            quantity = 1
            card_name = normalize_card_name(line)

        entry = DeckEntry(
            quantity=quantity,
            name=card_name,
            raw_line=raw_line,
            section=current_section,
        )

        if card_db is not None:
            entry.card = card_db.lookup(entry.name)
            entry.is_known = entry.card is not None

        if current_section == "sideboard":
            deck.sideboard.append(entry)
        elif current_section == "companion":
            deck.companion.append(entry)
        else:
            deck.entries.append(entry)

    return deck


def load_deck_file(path: Path | str, card_db: Optional[CardDatabase] = None) -> Deck:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    deck = parse_decklist(text, deck_name=file_path.stem, card_db=card_db)
    deck.source_path = file_path
    return deck


def load_decks_from_folder(folder: Path | str, card_db: Optional[CardDatabase] = None) -> list[Deck]:
    folder_path = Path(folder)
    if not folder_path.exists():
        return []
    return [load_deck_file(path, card_db=card_db) for path in sorted(folder_path.glob("*.txt"))]


def summarize_deck(deck: Deck) -> dict[str, Any]:
    colors = Counter()
    types = Counter()
    known_nonland_cards = 0
    total_mana_value = 0.0

    for entry in deck.entries:
        for color in _entry_colors(entry):
            colors[color] += entry.quantity

        if entry.card:
            for card_type in entry.card.types:
                types[card_type] += entry.quantity
            if "Land" not in entry.card.types:
                known_nonland_cards += entry.quantity
                total_mana_value += entry.card.mana_value * entry.quantity

    average_mana_value = round(total_mana_value / known_nonland_cards, 2) if known_nonland_cards else 0.0

    return {
        "total_cards": deck.total_cards,
        "unique_cards": len(deck.entries),
        "sideboard_cards": sum(entry.quantity for entry in deck.sideboard),
        "colors": sorted(colors.keys()),
        "primary_types": types.most_common(5),
        "average_mana_value": average_mana_value,
        "unknown_cards": deck.unknown_cards,
    }


def format_deck(deck: Deck) -> str:
    sections: list[str] = []

    if deck.entries:
        sections.extend(f"{entry.quantity} {entry.name}" for entry in deck.entries)

    if deck.sideboard:
        sections.append("")
        sections.append("Sideboard")
        sections.extend(f"{entry.quantity} {entry.name}" for entry in deck.sideboard)

    if deck.companion:
        sections.append("")
        sections.append("Companion")
        sections.extend(f"{entry.quantity} {entry.name}" for entry in deck.companion)

    return "\n".join(sections).strip() + "\n"


def save_imported_deck(
    deck_text: str,
    target_name: str,
    folder: Path | str,
    card_db: Optional[CardDatabase] = None,
) -> tuple[Path, Deck]:
    deck = parse_decklist(deck_text, deck_name=target_name, card_db=card_db)
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    safe_stem = SAFE_FILENAME_RE.sub("_", target_name).strip("._") or "ImportedDeck"
    file_path = folder_path / f"{safe_stem}.txt"
    file_path.write_text(format_deck(deck), encoding="utf-8")
    return file_path, deck


def rank_matching_decks(observed_cards: list[str], candidate_decks: list[Deck], top_n: int = 3) -> list[dict[str, Any]]:
    observed = [normalize_card_name(name) for name in observed_cards if normalize_card_name(name)]
    observed_keys = {normalize_lookup_key(name) for name in observed}
    matched_results: list[dict[str, Any]] = []
    fallback_results: list[dict[str, Any]] = []

    for deck in candidate_decks:
        deck_map = {normalize_lookup_key(entry.name): entry for entry in deck.entries}
        hits = [deck_map[key].name for key in observed_keys if key in deck_map]
        score = len(hits) * 10

        if hits:
            score += round(len(hits) / max(1, len(observed_keys)), 2)

        result = {
            "deck_name": deck.name,
            "score": score,
            "match_count": len(hits),
            "hits": hits,
        }

        if hits:
            matched_results.append(result)
        else:
            fallback_results.append(result)

    if matched_results:
        return sorted(matched_results, key=lambda item: (-item["score"], -item["match_count"], item["deck_name"]))[:top_n]

    return sorted(fallback_results, key=lambda item: item["deck_name"])[:top_n]


def _is_land_entry(entry: DeckEntry) -> bool:
    if entry.card and "Land" in entry.card.types:
        return True
    return entry.name.casefold() in BASIC_LAND_COLORS


def _expand_deck_to_card_pool(deck: Deck) -> list[DeckEntry]:
    pool: list[DeckEntry] = []
    for entry in deck.entries:
        for _ in range(entry.quantity):
            pool.append(
                DeckEntry(
                    quantity=1,
                    name=entry.name,
                    raw_line=entry.raw_line,
                    section=entry.section,
                    card=entry.card,
                    is_known=entry.is_known,
                )
            )
    return pool


def _make_permanent(entry: DeckEntry, turn_played: int) -> dict[str, Any]:
    card = entry.card
    text = (card.text if card else "").casefold()
    is_creature = bool(card and "Creature" in card.types)
    return {
        "name": entry.name,
        "is_land": _is_land_entry(entry),
        "is_creature": is_creature,
        "power": _estimate_creature_power(entry) if is_creature else 0.0,
        "toughness": _estimate_creature_toughness(entry) if is_creature else 0.0,
        "tapped": False,
        "turn_played": turn_played,
        "vigilance": "vigilance" in text,
        "lifelink": "lifelink" in text,
        "flying": "flying" in text,
        "reach": "reach" in text,
        "trample": "trample" in text,
        "deathtouch": "deathtouch" in text,
        "first_strike": "first strike" in text or "double strike" in text,
        "double_strike": "double strike" in text,
    }


def _build_player_state(deck: Deck, rng: random.Random) -> dict[str, Any]:
    card_pool = _expand_deck_to_card_pool(deck)
    rng.shuffle(card_pool)
    opening_hand_size = min(7, len(card_pool))
    return {
        "name": deck.name,
        "hand": card_pool[:opening_hand_size],
        "library": card_pool[opening_hand_size:],
        "battlefield": [],
        "graveyard": [],
        "life": 20,
    }


def _battlefield_names(state: dict[str, Any]) -> list[str]:
    return [permanent["name"] for permanent in state["battlefield"]]


def _untap_all(state: dict[str, Any]) -> None:
    for permanent in state["battlefield"]:
        permanent["tapped"] = False


def _count_untapped_lands(state: dict[str, Any]) -> int:
    return sum(1 for permanent in state["battlefield"] if permanent["is_land"] and not permanent["tapped"])


def _tap_lands_for_cost(state: dict[str, Any], amount: int) -> int:
    spent = 0
    for permanent in state["battlefield"]:
        if permanent["is_land"] and not permanent["tapped"] and spent < amount:
            permanent["tapped"] = True
            spent += 1
    return spent


def _play_land_from_hand(state: dict[str, Any], current_turn: int) -> Optional[str]:
    for entry in list(state["hand"]):
        if _is_land_entry(entry):
            state["hand"].remove(entry)
            state["battlefield"].append(_make_permanent(entry, turn_played=current_turn))
            return entry.name
    return None


def _remove_permanent(state: dict[str, Any], permanent: dict[str, Any]) -> None:
    if permanent in state["battlefield"]:
        state["battlefield"].remove(permanent)
        state["graveyard"].append(permanent["name"])


def _extract_protection_colors(text: str) -> set[str]:
    protections: set[str] = set()
    for color in ("white", "blue", "black", "red", "green"):
        if f"protection from {color}" in text:
            protections.add(color)
    return protections


def _card_color_names(card: Optional[CardInfo]) -> set[str]:
    if not card:
        return set()

    color_map = {
        "W": "white",
        "U": "blue",
        "B": "black",
        "R": "red",
        "G": "green",
    }
    normalized: set[str] = set()
    for color in getattr(card, "colors", []):
        color_text = str(color).casefold()
        normalized.add(color_map.get(str(color), color_text).casefold())
    return normalized


def choose_best_target(
    candidates: list[str],
    card_db: CardDatabase,
    source_name: Optional[str] = None,
) -> Optional[str]:
    if not candidates:
        return None

    source_card = card_db.lookup(source_name) if source_name else None
    source_colors = _card_color_names(source_card)

    scored_candidates: list[tuple[float, str]] = []
    for name in candidates:
        card = card_db.lookup(name)
        if not card:
            scored_candidates.append((1.0, name))
            continue

        text = (card.text or "").casefold()
        if "hexproof" in text or "shroud" in text or "protection from everything" in text:
            continue

        protection_colors = _extract_protection_colors(text)
        if source_colors and protection_colors.intersection(source_colors):
            continue

        score = card.mana_value + card.power + card.toughness
        if "flying" in text:
            score += 1.0
        if "lifelink" in text:
            score += 1.25
        if "draw" in text or "whenever" in text:
            score += 1.5
        if "angel" in text or "cleric" in text:
            score += 0.75
        if "ward" in text:
            score -= 7.0 if source_name else 2.75
        scored_candidates.append((score, name))

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    return scored_candidates[0][1]


def _build_combat_creature_view(creature: Any, card_db: CardDatabase) -> dict[str, Any]:
    if isinstance(creature, dict):
        name = creature.get("name", "Unknown")
        power = float(creature.get("power", 0.0) or 0.0)
        toughness = float(creature.get("toughness", 0.0) or 0.0)
        card = card_db.lookup(name)
        base_text = ((card.text if card else "") or "").casefold()
        custom_text = str(creature.get("text", "") or "").casefold()
        text = f"{base_text} {custom_text}".strip()
        if card:
            if power <= 0:
                power = card.power if card.power > 0 else max(1.0, card.mana_value - 1)
            if toughness <= 0:
                toughness = card.toughness if card.toughness > 0 else max(1.0, card.mana_value)
            mana_value = float(card.mana_value)
        else:
            mana_value = max(1.0, power, toughness)
        text = f"{text} {'lifelink' if creature.get('lifelink') else ''} {'vigilance' if creature.get('vigilance') else ''} {'flying' if creature.get('flying') else ''} {'reach' if creature.get('reach') else ''} {'trample' if creature.get('trample') else ''} {'deathtouch' if creature.get('deathtouch') else ''} {'first strike' if creature.get('first_strike') else ''} {'double strike' if creature.get('double_strike') else ''}".strip()
        return {
            "ref": creature,
            "name": name,
            "power": power,
            "toughness": toughness,
            "mana_value": mana_value,
            "text": text,
            "lifelink": bool(creature.get("lifelink")) or "lifelink" in text,
            "flying": bool(creature.get("flying")) or "flying" in text,
            "reach": bool(creature.get("reach")) or "reach" in text,
            "trample": bool(creature.get("trample")) or "trample" in text,
            "deathtouch": bool(creature.get("deathtouch")) or "deathtouch" in text,
            "first_strike": bool(creature.get("first_strike")) or "first strike" in text or "double strike" in text,
            "double_strike": bool(creature.get("double_strike")) or "double strike" in text,
        }

    name = str(creature)
    card = card_db.lookup(name)
    text = ((card.text if card else "") or "").casefold()
    if card:
        power = card.power if card.power > 0 else max(1.0, card.mana_value - 1)
        toughness = card.toughness if card.toughness > 0 else max(1.0, card.mana_value)
        mana_value = float(card.mana_value)
    else:
        power = 1.0
        toughness = 1.0
        mana_value = 1.0

    return {
        "ref": creature,
        "name": name,
        "power": float(power),
        "toughness": float(toughness),
        "mana_value": mana_value,
        "text": text,
        "lifelink": "lifelink" in text,
        "flying": "flying" in text,
        "reach": "reach" in text,
        "trample": "trample" in text,
        "deathtouch": "deathtouch" in text,
        "first_strike": "first strike" in text or "double strike" in text,
        "double_strike": "double strike" in text,
    }


def _score_combat_threat(creature: dict[str, Any]) -> float:
    text = creature.get("text", "")
    score = creature["power"] * 1.8 + creature["toughness"] * 1.1 + creature["mana_value"] * 0.35
    if creature.get("flying"):
        score += 1.25
    if creature.get("lifelink"):
        score += 1.0
    if creature.get("deathtouch"):
        score += 1.2
    if creature.get("trample"):
        score += 0.9
    if creature.get("first_strike"):
        score += 0.7
    if "ward" in text or "hexproof" in text:
        score += 0.6
    if "draw" in text or "whenever" in text:
        score += 0.75
    return score


def _can_block_attacker(attacker: dict[str, Any], blocker: dict[str, Any]) -> bool:
    if attacker.get("flying") and not (blocker.get("flying") or blocker.get("reach")):
        return False
    return True


def choose_block_assignments(
    attackers: list[Any],
    blockers: list[Any],
    card_db: CardDatabase,
    defender_life: int = 20,
) -> list[dict[str, Any]]:
    attacker_views = [_build_combat_creature_view(attacker, card_db) for attacker in attackers]
    blocker_views = [_build_combat_creature_view(blocker, card_db) for blocker in blockers]

    assignments: list[dict[str, Any]] = []
    remaining_damage = sum(attacker["power"] for attacker in attacker_views)

    attacker_views.sort(
        key=lambda attacker: (-_score_combat_threat(attacker), -attacker["power"], -attacker["toughness"], attacker["name"])
    )

    def _evaluate_block(
        attacker: dict[str, Any],
        blocker: dict[str, Any],
        lethal_risk: bool,
        current_block_power: float = 0.0,
        extra_block: bool = False,
    ) -> float:
        attacker_threat = _score_combat_threat(attacker)
        blocker_value = _score_combat_threat(blocker)
        combined_block_power = current_block_power + blocker["power"]
        attacker_dies = combined_block_power >= attacker["toughness"]
        blocker_dies = attacker["power"] >= blocker["toughness"]
        prevented_damage = 0.0 if extra_block else attacker["power"]

        score = attacker_threat * (1.2 if extra_block else 1.35) + prevented_damage * (4.0 if lethal_risk else 1.5)

        if attacker_dies and not blocker_dies:
            score += 6.0 + attacker_threat * 0.4
        elif attacker_dies and blocker_dies:
            score += 4.0 + attacker_threat * 0.25
        elif blocker_dies and not attacker_dies:
            score -= 4.5 + blocker_value * 0.4
        else:
            score += 0.75

        if extra_block and attacker_dies:
            score += 4.0
        elif extra_block and not attacker_dies:
            score -= 2.5

        if attacker.get("lifelink"):
            score += 1.5
        if "flying" in attacker.get("text", ""):
            score += 1.0
        if blocker_value > attacker_threat and blocker_dies and not lethal_risk:
            score -= 2.0

        return score

    for attacker in attacker_views:
        if not blocker_views:
            break

        lethal_risk = remaining_damage >= max(1.0, float(defender_life))
        best_blocker: Optional[dict[str, Any]] = None
        best_score: Optional[float] = None

        for blocker in blocker_views:
            if not _can_block_attacker(attacker, blocker):
                continue
            score = _evaluate_block(attacker, blocker, lethal_risk=lethal_risk)
            if best_score is None or score > best_score:
                best_score = score
                best_blocker = blocker

        if best_blocker is None:
            continue

        if best_score is not None and best_score < 1.0 and not lethal_risk:
            continue

        assignments.append(
            {
                "attacker": attacker["ref"],
                "blocker": best_blocker["ref"],
                "attacker_name": attacker["name"],
                "blocker_name": best_blocker["name"],
                "score": round(float(best_score or 0.0), 2),
            }
        )
        blocker_views.remove(best_blocker)
        remaining_damage -= attacker["power"]

    while blocker_views:
        lethal_risk = remaining_damage >= max(1.0, float(defender_life))
        best_extra: Optional[tuple[dict[str, Any], dict[str, Any], float]] = None

        for attacker in attacker_views:
            existing_blocks = [pair for pair in assignments if pair["attacker"] is attacker["ref"]]
            if not existing_blocks:
                continue

            current_block_power = sum(
                _build_combat_creature_view(pair["blocker"], card_db)["power"] for pair in existing_blocks
            )
            if current_block_power >= attacker["toughness"]:
                continue

            for blocker in blocker_views:
                if not _can_block_attacker(attacker, blocker):
                    continue
                score = _evaluate_block(
                    attacker,
                    blocker,
                    lethal_risk=lethal_risk,
                    current_block_power=current_block_power,
                    extra_block=True,
                )
                if current_block_power + blocker["power"] < attacker["toughness"] and not lethal_risk:
                    score -= 3.0

                if best_extra is None or score > best_extra[2]:
                    best_extra = (attacker, blocker, score)

        if best_extra is None or best_extra[2] < 6.0:
            break

        attacker, blocker, score = best_extra
        assignments.append(
            {
                "attacker": attacker["ref"],
                "blocker": blocker["ref"],
                "attacker_name": attacker["name"],
                "blocker_name": blocker["name"],
                "score": round(float(score), 2),
            }
        )
        blocker_views.remove(blocker)

    return assignments


def simulate_combat_exchange(
    attackers: list[Any],
    blockers: list[Any],
    card_db: Optional[CardDatabase] = None,
    attacker_life: int = 20,
    defender_life: int = 20,
) -> dict[str, Any]:
    active_db = card_db or CardDatabase()
    if card_db is None:
        active_db.load()

    attacker_permanents = []
    for attacker in attackers:
        view = _build_combat_creature_view(attacker, active_db)
        permanent = dict(attacker if isinstance(attacker, dict) else {"name": str(attacker)})
        permanent.update({
            "name": view["name"],
            "is_creature": True,
            "is_land": False,
            "power": view["power"],
            "toughness": view["toughness"],
            "tapped": False,
            "turn_played": 0,
            "vigilance": "vigilance" in view["text"],
            "lifelink": view["lifelink"],
            "flying": view["flying"],
            "reach": view["reach"],
            "trample": view["trample"],
            "deathtouch": view["deathtouch"],
            "first_strike": view["first_strike"],
            "double_strike": view["double_strike"],
        })
        attacker_permanents.append(permanent)

    blocker_permanents = []
    for blocker in blockers:
        view = _build_combat_creature_view(blocker, active_db)
        permanent = dict(blocker if isinstance(blocker, dict) else {"name": str(blocker)})
        permanent.update({
            "name": view["name"],
            "is_creature": True,
            "is_land": False,
            "power": view["power"],
            "toughness": view["toughness"],
            "tapped": False,
            "turn_played": 0,
            "vigilance": "vigilance" in view["text"],
            "lifelink": view["lifelink"],
            "flying": view["flying"],
            "reach": view["reach"],
            "trample": view["trample"],
            "deathtouch": view["deathtouch"],
            "first_strike": view["first_strike"],
            "double_strike": view["double_strike"],
        })
        blocker_permanents.append(permanent)

    attacker_state = {"life": attacker_life, "battlefield": attacker_permanents, "graveyard": []}
    defender_state = {"life": defender_life, "battlefield": blocker_permanents, "graveyard": []}
    assignments = _assign_blockers(defender_state, attacker_permanents, active_db)
    result = _resolve_combat_damage(attacker_state, defender_state, attacker_permanents, assignments)
    result.update(
        {
            "attacker_life": attacker_state["life"],
            "defender_life": defender_state["life"],
            "attacker_graveyard": attacker_state["graveyard"],
            "defender_graveyard": defender_state["graveyard"],
            "blocks": assignments,
        }
    )
    return result


def simulate_stack_interaction(
    acting_spell: str,
    response_spell: Optional[str],
    card_db: CardDatabase,
    extra_responses: Optional[list[str]] = None,
) -> dict[str, Any]:
    ordered_stack = [{"name": acting_spell, "type": "spell"}]
    if response_spell:
        ordered_stack.append({"name": response_spell, "type": "response"})
    for extra in extra_responses or []:
        ordered_stack.append({"name": extra, "type": "response"})

    stack = ordered_stack.copy()
    resolution: list[str] = []
    target_removed = False

    while stack:
        item = stack.pop()
        card = card_db.lookup(item["name"])
        text = (card.text if card else "").casefold()

        if item["type"] == "response":
            if "counter target" in text:
                target_removed = True
                resolution.append(f"{item['name']} contrarresta el hechizo objetivo")
            elif any(marker in text for marker in ("destroy target", "exile target", "deal ", "fight")):
                target_removed = True
                resolution.append(f"{item['name']} responde y neutraliza el valor del objetivo")
            else:
                resolution.append(f"{item['name']} se resuelve primero")
        else:
            if target_removed:
                resolution.append(f"{item['name']} ya no tiene impacto completo al resolverse")
            else:
                resolution.append(f"{item['name']} se resuelve con normalidad")

    return {
        "stack": [item["name"] for item in ordered_stack],
        "resolution": resolution,
    }


def resolve_simple_trigger(
    source_name: str,
    trigger_type: str,
    controller_state: dict[str, Any],
    opponent_state: dict[str, Any],
    card_db: CardDatabase,
) -> dict[str, Any]:
    card = card_db.lookup(source_name)
    text = (card.text if card else "").casefold()
    trigger_key = trigger_type.casefold()

    triggered = False
    detail = "sin trigger relevante"

    if trigger_key == "etb" and any(marker in text for marker in ("enters", "enters the battlefield", "whenever another")):
        triggered = True
        if "gain life" in text or "lifelink" in text or "angel or cleric" in text:
            controller_state["life"] += 2
            detail = f"{source_name} dispara ETB y ganas 2 vidas"
        else:
            detail = f"{source_name} dispara un efecto al entrar al campo"

    elif trigger_key == "upkeep" and any(marker in text for marker in ("at the beginning of your upkeep", "upkeep")):
        triggered = True
        if "draw" in text and controller_state.get("library"):
            drawn = controller_state["library"].pop(0)
            controller_state.setdefault("hand", []).append(drawn)
            detail = f"{source_name} dispara upkeep y roba {drawn.name}"
        elif "lose" in text and "life" in text:
            controller_state["life"] = max(0, controller_state["life"] - 1)
            detail = f"{source_name} dispara upkeep y pierdes 1 vida"
        else:
            detail = f"{source_name} dispara un efecto de upkeep"

    elif trigger_key == "end_step" and any(marker in text for marker in ("at end of turn", "beginning of the end step", "end step")):
        triggered = True
        detail = f"{source_name} dispara un efecto de final de turno"

    return {
        "source": source_name,
        "triggered": triggered,
        "detail": detail,
    }


def _resolve_simple_spell_effect(entry: DeckEntry, controller: dict[str, Any], opponent: dict[str, Any], card_db: CardDatabase) -> str:
    if not entry.card:
        return "efecto no modelado"

    text = entry.card.text.casefold()
    if any(marker in text for marker in ("destroy target", "exile target", "fight", "target creature gets -")):
        targets = [permanent for permanent in opponent["battlefield"] if permanent["is_creature"]]
        if not targets:
            return "sin objetivo válido"
        target_name = choose_best_target([permanent["name"] for permanent in targets], card_db, source_name=entry.name)
        if target_name is None:
            return "sin objetivo legal"
        target = next((permanent for permanent in targets if permanent["name"] == target_name), targets[0])
        _remove_permanent(opponent, target)
        return f"remueve {target['name']}"

    if "deal " in text and "target" in text:
        damage = 2
        targets = [permanent for permanent in opponent["battlefield"] if permanent["is_creature"]]
        if targets:
            target_name = choose_best_target([permanent["name"] for permanent in targets], card_db, source_name=entry.name)
            if target_name is None:
                return "sin objetivo legal"
            target = next((permanent for permanent in targets if permanent["name"] == target_name), targets[0])
            if damage >= target["toughness"]:
                _remove_permanent(opponent, target)
            else:
                target["toughness"] -= damage
            return f"hace {damage} daño a {target['name']}"
        opponent["life"] = max(0, opponent["life"] - damage)
        return f"hace {damage} daño al rival"

    if "draw" in text and controller["library"]:
        controller["hand"].append(controller["library"].pop(0))
        return "roba 1 carta"

    if "gain life" in text:
        controller["life"] += 2
        return "gana 2 vidas"

    return "hechizo resuelto"


def _choose_response_spell(state: dict[str, Any], available_mana: int) -> Optional[DeckEntry]:
    for entry in state["hand"]:
        if not entry.card:
            continue
        text = entry.card.text.casefold()
        mana_value = int(entry.card.mana_value)
        if mana_value <= available_mana and any(marker in text for marker in ("destroy target", "exile target", "deal ", "fight", "counter target")):
            return entry
    return None


def _choose_spell_to_cast(
    state: dict[str, Any],
    opponent: dict[str, Any],
    card_db: CardDatabase,
    current_turn: int,
    precombat: bool,
) -> Optional[DeckEntry]:
    available_mana = _count_untapped_lands(state)
    if available_mana <= 0:
        return None

    active_weights = ensure_learning_file().get("weights", {})
    pressure = analyze_battlefield(
        my_battlefield=_battlefield_names(state),
        opponent_battlefield=_battlefield_names(opponent),
        my_life=state["life"],
        opponent_life=opponent["life"],
        card_db=card_db,
    )["danger_level"]

    suggestions = suggest_plays_from_hand(
        [entry.name for entry in state["hand"]],
        available_mana=available_mana,
        card_db=card_db,
        lands_played_this_turn=True,
        opponent_pressure=pressure,
        top_n=max(1, len(state["hand"])),
        my_battlefield=_battlefield_names(state),
        opponent_battlefield=_battlefield_names(opponent),
        my_life=state["life"],
        opponent_life=opponent["life"],
        learned_weights=active_weights,
    )

    attack_ready = any(
        permanent["is_creature"] and not permanent["tapped"] and permanent["turn_played"] < current_turn
        for permanent in state["battlefield"]
    )

    for suggestion in suggestions:
        candidate = next((entry for entry in state["hand"] if entry.name == suggestion["name"]), None)
        if candidate is None:
            continue

        card_types = {card_type.casefold() for card_type in (candidate.card.types if candidate.card else [])}
        rules_text = (candidate.card.text if candidate.card else "").casefold()
        reactive_spell = any(marker in rules_text for marker in ("destroy target", "exile target", "deal ", "fight", "counter target"))

        if precombat and attack_ready and "creature" in card_types and not reactive_spell:
            continue

        mana_value = int(candidate.card.mana_value if candidate.card else 0)
        if mana_value <= available_mana:
            return candidate

    return None


def _cast_spell(
    entry: DeckEntry,
    state: dict[str, Any],
    opponent: dict[str, Any],
    current_turn: int,
    card_db: CardDatabase,
) -> tuple[str, str]:
    mana_value = max(0, int(entry.card.mana_value if entry.card else 0))
    _tap_lands_for_cost(state, mana_value)
    state["hand"].remove(entry)

    response = _choose_response_spell(opponent, _count_untapped_lands(opponent))
    if response is not None:
        response_mana = max(0, int(response.card.mana_value if response.card else 0))
        _tap_lands_for_cost(opponent, response_mana)
        opponent["hand"].remove(response)
        opponent["graveyard"].append(response.name)

        extra_responses: list[str] = []
        second_response = _choose_response_spell(state, _count_untapped_lands(state))
        if second_response is not None and second_response is not entry:
            second_mana = max(0, int(second_response.card.mana_value if second_response.card else 0))
            _tap_lands_for_cost(state, second_mana)
            state["hand"].remove(second_response)
            state["graveyard"].append(second_response.name)
            extra_responses.append(second_response.name)

        stack_result = simulate_stack_interaction(entry.name, response.name, card_db, extra_responses=extra_responses)
        if any("contrarresta" in line or "neutraliza" in line for line in stack_result["resolution"]):
            state["graveyard"].append(entry.name)
            return entry.name, f"lanza {entry.name}; respuesta: {' -> '.join(stack_result['stack'][1:])}; {stack_result['resolution'][-1]}"

    card_types = {card_type.casefold() for card_type in (entry.card.types if entry.card else [])}
    if any(card_type in card_types for card_type in ("creature", "artifact", "enchantment", "planeswalker")):
        state["battlefield"].append(_make_permanent(entry, turn_played=current_turn))
        return entry.name, f"lanza {entry.name}"

    effect_summary = _resolve_simple_spell_effect(entry, state, opponent, card_db)
    state["graveyard"].append(entry.name)
    return entry.name, f"lanza {entry.name} y {effect_summary}"


def _declare_attackers(state: dict[str, Any], opponent: dict[str, Any], current_turn: int, card_db: CardDatabase) -> tuple[list[dict[str, Any]], str]:
    combat_line = recommend_combat_line(
        my_battlefield=_battlefield_names(state),
        opponent_battlefield=_battlefield_names(opponent),
        my_life=state["life"],
        opponent_life=opponent["life"],
        card_db=card_db,
    )

    available_attackers = [
        permanent
        for permanent in state["battlefield"]
        if permanent["is_creature"] and not permanent["tapped"] and permanent["turn_played"] < current_turn
    ]

    if combat_line["plan"] != "attack":
        return [], combat_line["summary"]

    for attacker in available_attackers:
        if not attacker.get("vigilance"):
            attacker["tapped"] = True

    return available_attackers, combat_line["summary"]


def _assign_blockers(
    opponent: dict[str, Any],
    attackers: list[dict[str, Any]],
    card_db: CardDatabase,
) -> list[dict[str, dict[str, Any]]]:
    blockers = [permanent for permanent in opponent["battlefield"] if permanent["is_creature"] and not permanent["tapped"]]
    assignments = choose_block_assignments(
        attackers=attackers,
        blockers=blockers,
        card_db=card_db,
        defender_life=int(opponent.get("life", 20)),
    )
    return [{"attacker": pair["attacker"], "blocker": pair["blocker"]} for pair in assignments]


def _resolve_combat_damage(
    attacker_state: dict[str, Any],
    defender_state: dict[str, Any],
    attackers: list[dict[str, Any]],
    blocks: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    dead_attackers: list[dict[str, Any]] = []
    dead_blockers: list[dict[str, Any]] = []
    grouped_blocks: dict[int, dict[str, Any]] = {}
    damage_to_player = 0
    trample_damage = 0
    first_strike_used = False
    double_strike_used = False

    def _mark_dead(permanent: dict[str, Any], bucket: list[dict[str, Any]]) -> None:
        if permanent not in bucket:
            bucket.append(permanent)

    for pair in blocks:
        attacker = pair["attacker"]
        blocker = pair["blocker"]
        key = id(attacker)
        grouped_blocks.setdefault(key, {"attacker": attacker, "blockers": []})["blockers"].append(blocker)

    for group in grouped_blocks.values():
        attacker = group["attacker"]
        grouped_blockers = sorted(group["blockers"], key=lambda blocker: (blocker["toughness"], blocker["power"]))
        alive_blockers = list(grouped_blockers)
        attacker_alive = True

        attacker_first = bool(attacker.get("first_strike") or attacker.get("double_strike"))
        blocker_first_present = any(blocker.get("first_strike") or blocker.get("double_strike") for blocker in alive_blockers)
        if attacker.get("double_strike") or any(blocker.get("double_strike") for blocker in alive_blockers):
            double_strike_used = True

        def _attacker_hits(assign_to_player_on_trample: bool) -> None:
            nonlocal damage_to_player, trample_damage, alive_blockers
            remaining_attack_damage = float(attacker["power"])

            for blocker in list(alive_blockers):
                lethal_needed = 1 if attacker.get("deathtouch") and remaining_attack_damage > 0 else blocker["toughness"]
                if remaining_attack_damage >= lethal_needed:
                    _mark_dead(blocker, dead_blockers)
                    alive_blockers.remove(blocker)
                    remaining_attack_damage -= lethal_needed
                else:
                    remaining_attack_damage = 0
                    break

            if attacker.get("trample") and assign_to_player_on_trample and remaining_attack_damage > 0:
                trample_damage += int(remaining_attack_damage)
                damage_to_player += int(remaining_attack_damage)

            if attacker.get("lifelink"):
                attacker_state["life"] += int(attacker["power"])

        def _blockers_hit(first_strike_only: bool) -> None:
            nonlocal attacker_alive
            relevant_blockers = [
                blocker
                for blocker in alive_blockers
                if (blocker.get("first_strike") or blocker.get("double_strike")) == first_strike_only
                or (not first_strike_only and blocker.get("double_strike"))
            ]
            if not relevant_blockers:
                return

            total_block_power = sum(blocker["power"] for blocker in relevant_blockers)
            has_deathtouch = any(blocker.get("deathtouch") and blocker["power"] > 0 for blocker in relevant_blockers)
            if total_block_power >= attacker["toughness"] or has_deathtouch:
                _mark_dead(attacker, dead_attackers)
                attacker_alive = False

            for blocker in relevant_blockers:
                if blocker.get("lifelink"):
                    defender_state["life"] += int(blocker["power"])

        if attacker_first or blocker_first_present:
            first_strike_used = True
            if attacker_first and attacker_alive:
                _attacker_hits(assign_to_player_on_trample=True)
            _blockers_hit(first_strike_only=True)

        if attacker_alive and (not attacker_first or attacker.get("double_strike")):
            _attacker_hits(assign_to_player_on_trample=True)

        if attacker_alive:
            _blockers_hit(first_strike_only=False)

    unblocked_attackers = [attacker for attacker in attackers if id(attacker) not in grouped_blocks]
    for attacker in unblocked_attackers:
        strike_count = 2 if attacker.get("double_strike") else 1
        if attacker.get("double_strike"):
            double_strike_used = True
            first_strike_used = True
        damage_to_player += int(attacker["power"] * strike_count)
        if attacker.get("lifelink"):
            attacker_state["life"] += int(attacker["power"] * strike_count)

    defender_state["life"] = max(0, defender_state["life"] - damage_to_player)

    for permanent in dead_attackers:
        _remove_permanent(attacker_state, permanent)
    for permanent in dead_blockers:
        _remove_permanent(defender_state, permanent)

    if trample_damage and double_strike_used:
        summary = f"double strike y arrolla: el defensor recibe {damage_to_player} daño"
    elif trample_damage and first_strike_used:
        summary = f"first strike y arrolla: el defensor recibe {damage_to_player} daño"
    elif trample_damage:
        summary = f"el atacante arrolla y el defensor recibe {damage_to_player} daño"
    elif double_strike_used and damage_to_player:
        summary = f"double strike golpea dos veces y el defensor recibe {damage_to_player} daño"
    elif first_strike_used:
        summary = "first strike resolvió el combate entre criaturas"
    elif damage_to_player:
        summary = f"el defensor recibe {damage_to_player} daño"
    elif any(len(group["blockers"]) > 1 for group in grouped_blocks.values()):
        summary = "el combate se resolvió con doble bloqueo"
    elif blocks:
        summary = "todo el daño se resolvió en combate entre criaturas"
    else:
        summary = "no hubo daño de combate"

    return {
        "damage_to_player": damage_to_player,
        "summary": summary,
    }


def _cleanup_hand(state: dict[str, Any], hand_limit: int = 7) -> list[str]:
    discarded: list[str] = []
    while len(state["hand"]) > hand_limit:
        to_discard = max(state["hand"], key=lambda entry: (entry.card.mana_value if entry.card else 0, entry.name))
        state["hand"].remove(to_discard)
        state["graveyard"].append(to_discard.name)
        discarded.append(to_discard.name)
    return discarded


def _run_turn(
    active_state: dict[str, Any],
    defending_state: dict[str, Any],
    current_turn: int,
    card_db: CardDatabase,
    skip_draw: bool = False,
) -> dict[str, Any]:
    phases: list[dict[str, Any]] = []

    _untap_all(active_state)
    phases.append({"step": "untap", "detail": "endereza todos sus permanentes", "priority": False})

    upkeep_events = []
    for permanent in active_state["battlefield"]:
        event = resolve_simple_trigger(
            permanent["name"],
            "upkeep",
            controller_state=active_state,
            opponent_state=defending_state,
            card_db=card_db,
        )
        if event["triggered"]:
            upkeep_events.append(event["detail"])
    upkeep_detail = "; ".join(upkeep_events) if upkeep_events else "sin triggers de upkeep modelados"
    phases.append({"step": "upkeep", "detail": upkeep_detail, "priority": True})

    drawn_card = None
    if not skip_draw and active_state["library"]:
        drawn = active_state["library"].pop(0)
        active_state["hand"].append(drawn)
        drawn_card = drawn.name
        draw_detail = f"roba {drawn_card}"
    elif skip_draw:
        draw_detail = "omite el robo por ir en play"
    else:
        draw_detail = "no roba carta"
    phases.append({"step": "draw", "detail": draw_detail, "priority": True})

    land_played = _play_land_from_hand(active_state, current_turn=current_turn)
    precombat_choice = _choose_spell_to_cast(active_state, defending_state, card_db, current_turn=current_turn, precombat=True)
    main1_play = "ninguna"
    if precombat_choice is not None:
        spell_name, main1_play = _cast_spell(precombat_choice, active_state, defending_state, current_turn=current_turn, card_db=card_db)
        etb_events = []
        for permanent in active_state["battlefield"]:
            if permanent["name"] == spell_name and permanent["turn_played"] == current_turn:
                event = resolve_simple_trigger(
                    permanent["name"],
                    "etb",
                    controller_state=active_state,
                    opponent_state=defending_state,
                    card_db=card_db,
                )
                if event["triggered"]:
                    etb_events.append(event["detail"])
        if etb_events:
            main1_play += f"; triggers: {'; '.join(etb_events)}"
    main1_detail = f"tierra: {land_played or 'ninguna'}; acción: {main1_play}"
    phases.append({"step": "main1", "detail": main1_detail, "priority": True})

    phases.append({"step": "begin_combat", "detail": "ventana de prioridad antes de declarar atacantes", "priority": True})
    attackers, combat_plan_summary = _declare_attackers(active_state, defending_state, current_turn=current_turn, card_db=card_db)
    attacker_names = [attacker["name"] for attacker in attackers]
    attack_detail = f"ataca con {', '.join(attacker_names)}" if attacker_names else f"no ataca; {combat_plan_summary}"
    phases.append({"step": "declare_attackers", "detail": attack_detail, "priority": True})

    blocks = _assign_blockers(defending_state, attackers, card_db=card_db)
    if blocks:
        block_detail = "; ".join(f"{pair['blocker']['name']} bloquea a {pair['attacker']['name']}" for pair in blocks)
    else:
        block_detail = "sin bloqueadores"
    phases.append({"step": "declare_blockers", "detail": block_detail, "priority": True})

    combat_result = _resolve_combat_damage(active_state, defending_state, attackers, blocks)
    phases.append({"step": "combat_damage", "detail": combat_result["summary"], "priority": True})
    phases.append({"step": "end_combat", "detail": "fin del combate", "priority": True})

    postcombat_choice = _choose_spell_to_cast(active_state, defending_state, card_db, current_turn=current_turn, precombat=False)
    main2_play = "ninguna"
    if postcombat_choice is not None:
        spell_name, main2_play = _cast_spell(postcombat_choice, active_state, defending_state, current_turn=current_turn, card_db=card_db)
        etb_events = []
        for permanent in active_state["battlefield"]:
            if permanent["name"] == spell_name and permanent["turn_played"] == current_turn:
                event = resolve_simple_trigger(
                    permanent["name"],
                    "etb",
                    controller_state=active_state,
                    opponent_state=defending_state,
                    card_db=card_db,
                )
                if event["triggered"]:
                    etb_events.append(event["detail"])
        if etb_events:
            main2_play += f"; triggers: {'; '.join(etb_events)}"
    phases.append({"step": "main2", "detail": f"acción: {main2_play}", "priority": True})

    end_events = []
    for permanent in active_state["battlefield"]:
        event = resolve_simple_trigger(
            permanent["name"],
            "end_step",
            controller_state=active_state,
            opponent_state=defending_state,
            card_db=card_db,
        )
        if event["triggered"]:
            end_events.append(event["detail"])
    end_detail = "; ".join(end_events) if end_events else "sin acciones de end step modeladas"
    phases.append({"step": "end_step", "detail": end_detail, "priority": True})
    discarded = _cleanup_hand(active_state)
    cleanup_detail = f"descarta {', '.join(discarded)}" if discarded else "sin descarte; terminan efectos hasta end of turn"
    phases.append({"step": "cleanup", "detail": cleanup_detail, "priority": False})

    chosen_play = next(
        (play for play in (main1_play, main2_play) if play != "ninguna"),
        "Pasar con maná abierto",
    )

    return {
        "turn": current_turn,
        "active_player": active_state["name"],
        "draw": drawn_card,
        "land_played": land_played,
        "mana_available": sum(1 for permanent in active_state["battlefield"] if permanent["is_land"]),
        "play": chosen_play,
        "hand_size": len(active_state["hand"]),
        "attackers": attacker_names,
        "damage_dealt": combat_result["damage_to_player"],
        "life": active_state["life"],
        "opponent_life": defending_state["life"],
        "phases": phases,
    }


def simulate_opening_hand(
    deck: Deck,
    card_db: CardDatabase,
    seed: Optional[int] = None,
    hand_size: int = 7,
) -> dict[str, Any]:
    rng = random.Random(seed)
    card_pool = _expand_deck_to_card_pool(deck)
    if not card_pool:
        return {
            "hand": [],
            "land_count": 0,
            "average_mana_value": 0.0,
            "castable_early_plays": 0,
            "decision": "mulligan",
            "reason": "el deck está vacío o no tiene cartas cargables",
        }

    rng.shuffle(card_pool)
    hand = card_pool[: min(hand_size, len(card_pool))]

    land_count = sum(1 for entry in hand if _is_land_entry(entry))
    spells = [entry for entry in hand if not _is_land_entry(entry)]
    early_plays = sum(1 for entry in spells if entry.card and entry.card.mana_value <= 2)
    avg_mv = round(
        sum((entry.card.mana_value if entry.card else 2.0) for entry in spells) / max(1, len(spells)),
        2,
    ) if spells else 0.0

    decision = "keep"
    reasons: list[str] = []

    if land_count == 0:
        decision = "mulligan"
        reasons.append("mano sin tierras")
    elif land_count >= 6:
        decision = "mulligan"
        reasons.append("demasiadas tierras")
    elif land_count in {1, 5}:
        decision = "mulligan" if early_plays == 0 else "keep"
        reasons.append("reparto de tierras arriesgado")
    else:
        reasons.append("base de maná aceptable")

    if early_plays >= 2:
        reasons.append("buena curva temprana")
    elif early_plays == 0 and decision != "mulligan":
        reasons.append("mano lenta")

    if avg_mv > 3.8 and land_count <= 2:
        decision = "mulligan"
        reasons.append("curva demasiado pesada para esta mano")

    return {
        "hand": [entry.name for entry in hand],
        "land_count": land_count,
        "average_mana_value": avg_mv,
        "castable_early_plays": early_plays,
        "decision": decision,
        "reason": "; ".join(dict.fromkeys(reasons)) or "mano razonable",
    }


def simulate_opening_turns(
    deck: Deck,
    card_db: CardDatabase,
    turns: int = 3,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    active_state = _build_player_state(deck, rng)
    dummy_opponent = {"name": "Goldfish", "hand": [], "library": [], "battlefield": [], "graveyard": [], "life": 20}
    turn_log: list[dict[str, Any]] = []

    for turn in range(1, max(1, turns) + 1):
        turn_log.append(
            _run_turn(
                active_state,
                dummy_opponent,
                current_turn=turn,
                card_db=card_db,
                skip_draw=(turn == 1),
            )
        )

    return {
        "turns_simulated": max(1, turns),
        "turn_log": turn_log,
        "final_hand_size": len(active_state["hand"]),
        "lands_in_play": sum(1 for permanent in active_state["battlefield"] if permanent["is_land"]),
        "battlefield": _battlefield_names(active_state),
        "life": active_state["life"],
        "opponent_life": dummy_opponent["life"],
    }


def simulate_matchup(
    deck_a: Deck,
    deck_b: Deck,
    card_db: CardDatabase,
    turns: int = 3,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    player_a = _build_player_state(deck_a, rng)
    player_b = _build_player_state(deck_b, rng)

    result_a = {"turn_log": []}
    result_b = {"turn_log": []}
    turn_history: list[dict[str, Any]] = []

    for turn in range(1, max(1, turns) + 1):
        if turn % 2 == 1:
            active_state, defending_state = player_a, player_b
            target_log = result_a["turn_log"]
        else:
            active_state, defending_state = player_b, player_a
            target_log = result_b["turn_log"]

        turn_result = _run_turn(
            active_state,
            defending_state,
            current_turn=turn,
            card_db=card_db,
            skip_draw=(turn == 1 and active_state is player_a),
        )
        turn_history.append(turn_result)
        target_log.append(turn_result)

    result_a.update(
        {
            "turns_simulated": len(result_a["turn_log"]),
            "final_hand_size": len(player_a["hand"]),
            "lands_in_play": sum(1 for permanent in player_a["battlefield"] if permanent["is_land"]),
            "battlefield": _battlefield_names(player_a),
        }
    )
    result_b.update(
        {
            "turns_simulated": len(result_b["turn_log"]),
            "final_hand_size": len(player_b["hand"]),
            "lands_in_play": sum(1 for permanent in player_b["battlefield"] if permanent["is_land"]),
            "battlefield": _battlefield_names(player_b),
        }
    )

    board_a = analyze_battlefield(
        my_battlefield=result_a["battlefield"],
        opponent_battlefield=result_b["battlefield"],
        my_life=player_a["life"],
        opponent_life=player_b["life"],
        card_db=card_db,
    )
    board_b = analyze_battlefield(
        my_battlefield=result_b["battlefield"],
        opponent_battlefield=result_a["battlefield"],
        my_life=player_b["life"],
        opponent_life=player_a["life"],
        card_db=card_db,
    )

    weights = ensure_learning_file().get("weights", {})
    board_weight = weights.get("board_presence", 1.0)
    card_weight = weights.get("card_advantage", 1.0)
    mana_weight = weights.get("mana_efficiency", 1.0)
    lethal_weight = weights.get("lethal_pressure", 1.0)

    score_a = round(
        (board_a["my_attack"] + board_a["my_defense"] * 0.6) * board_weight
        + len(player_a["hand"]) * 0.25 * card_weight
        + result_a["lands_in_play"] * 0.75 * mana_weight
        + max(0, 20 - player_b["life"]) * lethal_weight,
        2,
    )
    score_b = round(
        (board_b["my_attack"] + board_b["my_defense"] * 0.6) * board_weight
        + len(player_b["hand"]) * 0.25 * card_weight
        + result_b["lands_in_play"] * 0.75 * mana_weight
        + max(0, 20 - player_a["life"]) * lethal_weight,
        2,
    )

    if player_a["life"] <= 0 < player_b["life"]:
        winner = deck_b.name
    elif player_b["life"] <= 0 < player_a["life"]:
        winner = deck_a.name
    elif score_a > score_b:
        winner = deck_a.name
    elif score_b > score_a:
        winner = deck_b.name
    else:
        winner = "draw"

    return {
        "deck_a": deck_a.name,
        "deck_b": deck_b.name,
        "turns_simulated": max(1, turns),
        "score_a": score_a,
        "score_b": score_b,
        "life_a": player_a["life"],
        "life_b": player_b["life"],
        "winner": winner,
        "summary": f"{deck_a.name} {score_a} vs {deck_b.name} {score_b}",
        "turn_history": turn_history,
        "result_a": result_a,
        "result_b": result_b,
    }


def simulate_many_matchups(
    deck_a: Deck,
    deck_b: Deck,
    card_db: CardDatabase,
    simulations: int = 100,
    turns: int = 3,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    total_runs = max(1, simulations)
    rng = random.Random(seed)
    results = [
        simulate_matchup(deck_a, deck_b, card_db=card_db, turns=turns, seed=rng.randint(1, 10_000_000))
        for _ in range(total_runs)
    ]

    wins_a = sum(1 for result in results if result["winner"] == deck_a.name)
    wins_b = sum(1 for result in results if result["winner"] == deck_b.name)
    draws = sum(1 for result in results if result["winner"] == "draw")

    return {
        "deck_a": deck_a.name,
        "deck_b": deck_b.name,
        "simulations": total_runs,
        "turns": turns,
        "winrate_a": round((wins_a / total_runs) * 100, 2),
        "winrate_b": round((wins_b / total_runs) * 100, 2),
        "draw_rate": round((draws / total_runs) * 100, 2),
        "average_score_a": round(sum(result["score_a"] for result in results) / total_runs, 2),
        "average_score_b": round(sum(result["score_b"] for result in results) / total_runs, 2),
        "sample_results": results[:5],
    }


def simulate_many_opening_hands(
    deck: Deck,
    card_db: CardDatabase,
    simulations: int = 100,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    total_runs = max(1, simulations)
    rng = random.Random(seed)
    results = [
        simulate_opening_hand(deck, card_db=card_db, seed=rng.randint(1, 10_000_000))
        for _ in range(total_runs)
    ]

    keep_count = sum(1 for result in results if result["decision"] == "keep")
    average_lands = round(sum(result["land_count"] for result in results) / total_runs, 2)
    average_curve = round(sum(result["average_mana_value"] for result in results) / total_runs, 2)
    average_early = round(sum(result["castable_early_plays"] for result in results) / total_runs, 2)

    example_keeps = [result["hand"] for result in results if result["decision"] == "keep"][:3]
    example_mulls = [result["hand"] for result in results if result["decision"] == "mulligan"][:3]

    return {
        "simulations": total_runs,
        "keep_rate": round((keep_count / total_runs) * 100, 2),
        "mulligan_rate": round(((total_runs - keep_count) / total_runs) * 100, 2),
        "average_lands": average_lands,
        "average_curve": average_curve,
        "average_early_plays": average_early,
        "example_keeps": example_keeps,
        "example_mulligans": example_mulls,
    }


def _score_entry_for_play(
    entry: DeckEntry,
    hand: Deck,
    available_mana: int,
    opponent_pressure: str = "medium",
    learned_weights: Optional[dict[str, float]] = None,
) -> Optional[dict[str, Any]]:
    if _is_land_entry(entry):
        return None

    pressure = opponent_pressure.casefold().strip()
    weights = learned_weights or {
        "board_presence": 1.0,
        "card_advantage": 1.0,
        "mana_efficiency": 1.0,
        "lethal_pressure": 1.0,
    }
    pressure_bonus = {"low": 1.0, "medium": 3.0, "high": 7.5}.get(pressure, 3.0) * weights.get("lethal_pressure", 1.0)
    mana_value = entry.card.mana_value if entry.card else max(1.0, min(float(available_mana or 1), 3.0))
    if mana_value > available_mana:
        return None

    score = 2.0
    reasons: list[str] = []
    hand_text = " ".join(item.name for item in hand.entries).casefold()
    card_text = (entry.card.text if entry.card else "").casefold()
    card_types = {item.casefold() for item in (entry.card.types if entry.card else [])}

    if available_mana > 0:
        score += max(0.0, 4.0 - abs(available_mana - mana_value) * 1.25) * weights.get("mana_efficiency", 1.0)
        reasons.append("aprovecha bien el maná disponible")

    if not entry.card:
        score -= 1.0
        reasons.append("la carta no está del todo identificada, así que la prioridad es conservadora")
    else:
        if "creature" in card_types:
            score += 2.5 * weights.get("board_presence", 1.0)
            reasons.append("desarrolla mesa")
            if mana_value <= 2 and available_mana <= 3:
                score += 1.5 * weights.get("mana_efficiency", 1.0)
                reasons.append("mejora tu curva temprana")

        if "planeswalker" in card_types:
            score += 3.5 * weights.get("card_advantage", 1.0)
            reasons.append("genera ventaja sostenida")

        if any(marker in card_text for marker in ("destroy target", "exile target", "deal ", "fight", "target creature gets -")):
            score += pressure_bonus
            if pressure == "high":
                score += 3.5
            reasons.append("te ayuda a responder la presión rival")

        if "draw" in card_text:
            score += 1.5 * weights.get("card_advantage", 1.0)
            reasons.append("te da ventaja de cartas")

        if "lifelink" in card_text or "gain life" in card_text:
            score += {"low": 0.5, "medium": 1.5, "high": 3.0}.get(pressure, 1.5)
            reasons.append("mejora tu carrera de vidas")

        if "flying" in card_text:
            score += 1.0
            reasons.append("aporta evasión")

        if "counter target" in card_text:
            score += {"low": 3.0, "medium": 2.0, "high": 0.5}.get(pressure, 2.0)
            reasons.append("permite jugar a velocidad de respuesta")

        if "add " in card_text and "mana" in card_text and available_mana <= 3:
            score += 1.25
            reasons.append("acelera tu desarrollo")

        if "angel" in card_text and "angel" in hand_text:
            score += 1.25
            reasons.append("sinergiza con otras cartas de tu mano")

        if "dragon" in card_text and "dragon" in hand_text:
            score += 1.25
            reasons.append("mantiene la sinergia del mazo")

    return {
        "name": entry.name,
        "score": round(score, 2),
        "mana_value": mana_value,
        "reason": "; ".join(dict.fromkeys(reasons)) or "línea sólida para este turno",
    }


def suggest_plays_from_hand(
    hand_cards: str | list[str],
    available_mana: int,
    card_db: CardDatabase,
    lands_played_this_turn: bool = True,
    opponent_pressure: str = "medium",
    top_n: int = 3,
    my_battlefield: str | list[str] | None = None,
    opponent_battlefield: str | list[str] | None = None,
    my_life: int = 20,
    opponent_life: int = 20,
    learned_weights: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    hand = _parse_cards_as_deck(hand_cards, deck_name="CurrentHand", card_db=card_db)

    suggestions: list[dict[str, Any]] = []
    land_entries = [entry for entry in hand.entries if _is_land_entry(entry)]

    battlefield_summary: Optional[dict[str, Any]] = None
    if my_battlefield is not None or opponent_battlefield is not None:
        battlefield_summary = analyze_battlefield(
            my_battlefield=my_battlefield or [],
            opponent_battlefield=opponent_battlefield or [],
            my_life=my_life,
            opponent_life=opponent_life,
            card_db=card_db,
        )
        if battlefield_summary["danger_level"] == "high" and opponent_pressure != "high":
            opponent_pressure = "high"

    if learned_weights is None:
        learned_weights = ensure_learning_file().get("weights", {})

    for entry in hand.entries:
        suggestion = _score_entry_for_play(
            entry,
            hand,
            available_mana=available_mana,
            opponent_pressure=opponent_pressure,
            learned_weights=learned_weights,
        )
        if suggestion:
            suggestions.append(suggestion)

    suggestions = sorted(suggestions, key=lambda item: (-item["score"], item["mana_value"], item["name"]))

    if not suggestions and land_entries and not lands_played_this_turn:
        suggestions.append(
            {
                "name": f"Jugar {land_entries[0].name}",
                "score": 5.0,
                "mana_value": 0,
                "reason": "todavía no bajaste tierra y necesitas desarrollar tu maná",
            }
        )
    elif not suggestions:
        suggestions.append(
            {
                "name": "Pasar con maná abierto",
                "score": 0.0,
                "mana_value": 0,
                "reason": "no hay hechizos casteables con el maná actual",
            }
        )

    if battlefield_summary:
        combat_line = recommend_combat_line(
            my_battlefield=my_battlefield or [],
            opponent_battlefield=opponent_battlefield or [],
            my_life=my_life,
            opponent_life=opponent_life,
            card_db=card_db,
        )
        suggestions[0]["combat_summary"] = battlefield_summary["combat_recommendation"]
        suggestions[0]["danger_level"] = battlefield_summary["danger_level"]
        suggestions[0]["lethal_available"] = battlefield_summary["lethal_available"]
        suggestions[0]["combat_plan"] = combat_line["plan"]
        suggestions[0]["combat_plan_summary"] = combat_line["summary"]

    if land_entries and not lands_played_this_turn:
        suggestions[0]["note"] = f"Antes de todo, considera jugar {land_entries[0].name} si aún no has bajado tierra este turno."

    return suggestions[:top_n]


def ensure_learning_file(path: Optional[Path] = None) -> dict[str, Any]:
    learning_path = Path(path) if path else LEARNING_FILE
    default_state = {
        "format": "alchemy",
        "weights": {
            "board_presence": 1.0,
            "card_advantage": 1.0,
            "mana_efficiency": 1.0,
            "lethal_pressure": 1.0,
        },
        "history": [],
    }

    try:
        if learning_path.exists() and learning_path.read_text(encoding="utf-8").strip():
            return json.loads(learning_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        pass

    learning_path.write_text(json.dumps(default_state, indent=2), encoding="utf-8")
    return default_state


def record_learning_event(event_type: str, payload: dict[str, Any], path: Optional[Path] = None) -> None:
    learning_path = Path(path) if path else LEARNING_FILE
    state = ensure_learning_file(learning_path)
    state.setdefault("history", []).append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event_type,
            "payload": payload,
        }
    )
    learning_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def train_from_matchup_results(
    matchup_result: dict[str, Any],
    learning_path: Optional[Path] = None,
) -> dict[str, Any]:
    path = Path(learning_path) if learning_path else LEARNING_FILE
    state = ensure_learning_file(path)
    weights = state.setdefault(
        "weights",
        {
            "board_presence": 1.0,
            "card_advantage": 1.0,
            "mana_efficiency": 1.0,
            "lethal_pressure": 1.0,
        },
    )

    winrate_a = float(matchup_result.get("winrate_a", 50.0))
    winrate_b = float(matchup_result.get("winrate_b", 50.0))
    score_a = float(matchup_result.get("average_score_a", 0.0))
    score_b = float(matchup_result.get("average_score_b", 0.0))

    edge = (winrate_a - winrate_b) / 100.0
    score_gap = (score_a - score_b) / max(1.0, abs(score_a) + abs(score_b))

    weights["board_presence"] = round(max(0.1, weights["board_presence"] + edge * 0.15), 3)
    weights["card_advantage"] = round(max(0.1, weights["card_advantage"] + score_gap * 0.1), 3)
    weights["mana_efficiency"] = round(max(0.1, weights["mana_efficiency"] + edge * 0.08), 3)
    weights["lethal_pressure"] = round(max(0.1, weights["lethal_pressure"] + score_gap * 0.12), 3)

    state.setdefault("history", []).append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": "weights_updated",
            "payload": {
                "matchup_result": matchup_result,
                "weights": weights,
            },
        }
    )
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


__all__ = [
    "analyze_battlefield",
    "CardDatabase",
    "choose_best_target",
    "choose_block_assignments",
    "recommend_blocks",
    "recommend_combat_line",
    "Deck",
    "DeckEntry",
    "META_DECKS_DIR",
    "REGISTERED_DECKS_DIR",
    "ensure_learning_file",
    "format_deck",
    "load_decks_from_folder",
    "normalize_card_name",
    "parse_decklist",
    "rank_matching_decks",
    "record_learning_event",
    "resolve_simple_trigger",
    "save_imported_deck",
    "simulate_matchup",
    "simulate_many_matchups",
    "simulate_combat_exchange",
    "simulate_stack_interaction",
    "simulate_many_opening_hands",
    "simulate_opening_hand",
    "simulate_opening_turns",
    "suggest_plays_from_hand",
    "summarize_deck",
    "train_from_matchup_results",
]
