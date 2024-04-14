import json
import pathlib
from nltk import word_tokenize, SnowballStemmer

# Количество компаний для обработки
NUM_COMPANIES = 276

# Путь к файлу со словарем синонимов
FILE_PATH = pathlib.Path("data") / "dict.json"
with open(FILE_PATH, 'r', encoding='utf-8') as file:
    SYNON_DICT = json.load(file)

# Создание стеммера для русского языка
STEMMER = SnowballStemmer("russian")

# Создание списка маркеров для каждой компании
MARKERS = [f'%{i}%' for i in range(NUM_COMPANIES)]


def process_text(text: str, synon_dict: dict) -> tuple:
    tokens = word_tokenize(text, language='russian')
    tokens = [token.lower() for token in tokens]

    stem_tokens = [STEMMER.stem(token) for token in tokens if len(token) > 1]

    return replace_phrases_with_markers(
        stem_tokens, tokens, maximize_phrase_coverage(stem_tokens, synon_dict, 5), MARKERS
    )


def maximize_phrase_coverage(tokens: list[str], dictionary: dict, max_window_size: int = 20) -> list[list[int]]:
    def binary_search(target: str) -> int:
        return len(target.split()) if target in dictionary else 0

    if not tokens:
        return []

    token_ids, text_tokens = zip(*((idx, token) for idx, token in enumerate(tokens) if token))
    token_count = len(text_tokens)

    max_sums = [0] * token_count
    window_sizes = [-1] * token_count

    max_sums[0] = binary_search(text_tokens[0])

    for token_idx in range(1, token_count):
        current_window = ''
        for sz in range(1, min(max_window_size, token_idx + 1) + 1):
            current_window = f'{text_tokens[token_idx - sz + 1]} {current_window}'.strip()

            coverage_value = binary_search(current_window)
            prev_token_idx = token_idx - sz
            if coverage_value != 0 and max_sums[token_idx] < max_sums[prev_token_idx] + coverage_value:
                max_sums[token_idx] = max_sums[prev_token_idx] + coverage_value
                window_sizes[token_idx] = sz

    return reconstruct_token_groups(token_ids, window_sizes)


def reconstruct_token_groups(token_ids: tuple, window_sizes: list[int]) -> list[list[int]]:
    bounds, current_index = [], len(window_sizes) - 1
    while current_index >= 0:
        window_size = window_sizes[current_index]
        bounds.insert(0, window_size)
        current_index -= max(window_size, 1)

    token_groups, current_index = [], 0
    for window_size in bounds:
        if window_size > 0:
            group = token_ids[current_index:current_index + window_size]
            token_groups.append(group)
            current_index += window_size

    return token_groups


def replace_phrases_with_markers(stem_tokens: list[str], tokens: list[str], groups: list[list[int]], markers: list[str]) -> tuple:
    new_tokens, marker_positions = [], {}
    last_index = 0

    for group in groups:
        if group:
            phrase = " ".join(stem_tokens[idx] for idx in group)
            marker_index = SYNON_DICT.get(phrase)

            if marker_index is not None and marker_index <= 274:
                new_tokens.extend(stem_tokens[last_index:group[0]])
                new_tokens.append(markers[marker_index])
                marker_positions.setdefault(marker_index, []).append(len(new_tokens) - 1)
                last_index = group[-1] + 1

    new_tokens.extend(stem_tokens[last_index:])
    return new_tokens, marker_positions


def extract_company_context(tokens: list[str], mentioned_companies: dict, window_size: int = 3) -> dict:
    company_contexts = {}
    for company_id, positions in mentioned_companies.items():
        contexts = [tokens[max(0, pos - window_size):min(len(tokens), pos + window_size + 1)] for pos in positions]
        company_contexts[company_id] = [tok for context in contexts for tok in context]

    return company_contexts
