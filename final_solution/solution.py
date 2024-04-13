import typing as tp
from .find_companies import process_text, synon_dict, extract_company_context
from .model import predict

EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[
    EntityScoreType
]  # list of entity scores,
#    for example, [(entity_id, entity_score) for entity_id, entity_score in entities_found]


def score_texts(
    messages: tp.Iterable[str], *args, **kwargs
) -> tp.Iterable[MessageResultType]:
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)

    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 10 ** 11 for m in messages]) # all messages are shorter than 2048 characters
    """

    answer = []
    for i, message in enumerate(messages):
        answer.append([])
        tokens, mentioned_companies = process_text(message, synon_dict)
        company_context = extract_company_context(tokens, mentioned_companies, window_size=4)
        for company, token_window in company_context.items():
            answer[i].append((company, predict(token_window)))
    return answer


