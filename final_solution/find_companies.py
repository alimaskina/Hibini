import json
import pathlib
from nltk import word_tokenize, SnowballStemmer

num_companies = 276

file_path = pathlib.Path("data") / "dict.json"
with open(file_path, 'r', encoding='utf-8') as file:
    synon_dict = json.load(file)

stemmer = SnowballStemmer("russian")

markers = [f'%{i}%' for i in range(num_companies)]


def process_text(text, synon_dict):
    tokens = word_tokenize(text, language='russian')
    tokens = [token.lower() for token in tokens]

    stem_tokens = [stemmer.stem(token) for token in tokens if len(token) > 1]

    # Находим компании и заменяем на маркеры
    return replace_phrases_with_markers(stem_tokens, tokens, maximize_phrase_coverage(stem_tokens, synon_dict, 5), markers)


def maximize_phrase_coverage(tokens: list[str], dictionary, max_window_size=20) -> list[list[int]]:
    def binary_search(target: str) -> int:
        if target in dictionary:
            return len(target.split())  # Фраза найдена
        return 0  # Фраза не найдена

    # Фильтруем только непустые токены
    if len(tokens) == 0:
        return []
    token_ids, text_tokens = zip(*[[idx, token] for idx, token in enumerate(tokens) if token])
    token_count = len(text_tokens)

    # max_sums[token_idx] - максимальное значение суммы значений для префикса [0...token_idx]
    max_sums = [0] * token_count
    window_sizes = [-1] * token_count

    # Инициализация для первого токена
    max_sums[0] = binary_search(text_tokens[0])

    for token_idx in range(1, token_count):
        current_window = ''
        for sz in range(1, max_window_size + 1):
            prev_token_idx = token_idx - sz
            if prev_token_idx < -1:
                break

            if current_window == '':
                current_window = text_tokens[prev_token_idx + 1]
            else:
                current_window = text_tokens[prev_token_idx + 1] + ' ' + current_window

            coverage_value = binary_search(current_window)

            if current_window == 'moex' and text_tokens[prev_token_idx] == '(' and text_tokens[prev_token_idx + 2] == ':':
                continue

            if coverage_value != 0 and max_sums[token_idx] < max_sums[prev_token_idx] + coverage_value:
                max_sums[token_idx] = max_sums[prev_token_idx] + coverage_value
                window_sizes[token_idx] = sz

    # Восстановление размеров окон токенов
    bounds = []
    current_index = token_count - 1
    while current_index != -1:
        window_size = window_sizes[current_index]
        bounds.append(window_size)
        window_size = 1 if window_size == -1 else window_size
        current_index -= window_size
    bounds.reverse()

    # Восстановление групп токенов
    token_groups = []
    current_index = 0
    for window_size in bounds:
        if window_size == -1:
            current_index += 1
            continue
        group = token_ids[current_index:current_index + window_size]
        token_groups.append(group)
        current_index += window_size

    return token_groups


def replace_phrases_with_markers(stem_tokens, tokens, groups, markers):
    """
    Функция находит и заменяет группы токенов, соответствующие записям в словаре,
    на маркеры, а также сохраняет позиции маркеров в тексте.

    Параметры:
        tokens (list[str]): Исходный список слов.
        groups (list[list[int]]): Список групп индексов, соответствующих найденным фразам.
        markers (dict): Словарь маркеров, где ключ - это индекс фразы в dictionary, а значение - маркер.

    Возвращает:
        tuple: Модифицированный список слов и словарь с позициями маркеров.
    """
    new_tokens = []
    marker_positions = {}
    last_index = 0  # Следим за последним индексом, который был добавлен в new_tokens

    for group in groups:
        if not group:
            continue
        # Проверяем, соответствует ли группа записи в словаре
        phrase = " ".join(stem_tokens[idx] for idx in group)
        marker_index = synon_dict[phrase]  # Получаем маркер для найденной фразы, если он есть

        if marker_index and marker_index <= 274:
            # Добавляем все токены до начала текущей группы
            new_tokens.extend(stem_tokens[last_index:group[0]])
            # Добавляем маркер в новый список
            # new_tokens.append(markers[marker_index])
            # Записываем позицию маркера
            if marker_index not in marker_positions:
                marker_positions[marker_index] = []
            marker_positions[marker_index].append(len(new_tokens) - 1)

            # Обновляем последний обработанный индекс
            last_index = group[-1] + 1

    # Добавляем все оставшиеся токены после последней группы
    new_tokens.extend(stem_tokens[last_index:])

    return new_tokens, marker_positions


def extract_company_context(tokens, mentioned_companies, window_size=3):
    # Словарь для хранения контекстов для каждой компании
    company_contexts = {}

    # Перебираем все упоминания компаний
    for company_id, positions in mentioned_companies.items():
        # Список контекстов для текущей компании
        contexts = []

        for pos in positions:
            # Определяем начало и конец контекстного окна
            start = max(0, pos - window_size)
            end = min(len(tokens), pos + window_size + 1)

            # Собираем слова в окне вокруг вхождения компании
            context_tokens = tokens[start:end]

            # Добавляем полученный контекст в список контекстов компании
            for tok in context_tokens:
                contexts.append(tok)

        # Записываем список контекстов для текущей компании в общий словарь
        company_contexts[company_id] = contexts

    return company_contexts

