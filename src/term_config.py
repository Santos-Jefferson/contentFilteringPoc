import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from src.filtering import normalize_word


def load_term_config(config_path: str) -> dict:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Filter config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_category_names(config: dict) -> List[str]:
    return sorted(config.get("categories", {}).keys())


def get_default_categories(config: dict) -> List[str]:
    defaults: List[str] = []
    categories = config.get("categories", {})
    for name, category_cfg in categories.items():
        if category_cfg.get("enabled_by_default", True):
            defaults.append(name)
    return sorted(defaults)


def resolve_terms_from_categories(
    config: dict,
    selected_categories: Iterable[str],
) -> Tuple[Set[str], Set[str], Dict[str, Set[str]]]:
    categories = config.get("categories", {})
    selected = set(selected_categories or [])
    if not selected:
        selected = set(categories.keys())

    token_terms: Set[str] = set()
    caption_terms: Set[str] = set()
    context_exclusions: Dict[str, Set[str]] = {}

    for category_name in selected:
        category_cfg = categories.get(category_name)
        if not isinstance(category_cfg, dict):
            continue

        for raw_term in category_cfg.get("word_terms", []):
            normalized = normalize_word(str(raw_term))
            if normalized:
                token_terms.add(normalized)
                caption_terms.add(normalized)

        for raw_phrase in category_cfg.get("phrase_terms", []):
            phrase = " ".join(str(raw_phrase).lower().split())
            if phrase:
                caption_terms.add(phrase)

        raw_exclusions = category_cfg.get("context_exclusions", {})
        if isinstance(raw_exclusions, dict):
            for key_term, excluded_terms in raw_exclusions.items():
                normalized_key = normalize_word(str(key_term))
                if not normalized_key:
                    continue
                context_exclusions.setdefault(normalized_key, set())
                for excluded in excluded_terms:
                    normalized_excluded = normalize_word(str(excluded))
                    if normalized_excluded:
                        context_exclusions[normalized_key].add(normalized_excluded)

    return token_terms, caption_terms, context_exclusions


def get_scene_category_names(config: dict) -> List[str]:
    return sorted(config.get("scene_detection", {}).get("categories", {}).keys())


def get_default_scene_categories(config: dict) -> List[str]:
    defaults: List[str] = []
    categories = config.get("scene_detection", {}).get("categories", {})
    for name, category_cfg in categories.items():
        if isinstance(category_cfg, dict) and category_cfg.get("enabled_by_default", False):
            defaults.append(name)
    return sorted(defaults)


def get_scene_detection_config(config: dict) -> dict:
    return config.get("scene_detection", {})

