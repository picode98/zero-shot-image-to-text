from typing import Iterable, Callable

from nltk import StemmerI, PorterStemmer
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.translate.meteor_score import _generate_enums, _enum_align_words, _count_chunks


# Implementation copied and modified from nltk.translate.meteor_score.single_meteor_score
# Calculates a corpus-level METEOR score using aggregate statistics.
def corpus_meteor_score(
    ref_list: Iterable[Iterable[str]],
    hyp_list: Iterable[Iterable[str]],
    preprocess: Callable[[str], str] = str.lower,
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    matches_count = 0
    translation_length = 0
    reference_length = 0
    chunk_count = 0

    for hypothesis, reference in zip(hyp_list, ref_list):
        enum_hypothesis, enum_reference = _generate_enums(
            hypothesis, reference, preprocess=preprocess
        )
        translation_length += len(enum_hypothesis)
        reference_length += len(enum_reference)
        matches, _, _ = _enum_align_words(
            enum_hypothesis, enum_reference, stemmer=stemmer, wordnet=wordnet
        )
        matches_count += len(matches)
        chunk_count += _count_chunks(matches)

    try:
        precision = float(matches_count) / translation_length
        recall = float(matches_count) / reference_length
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        frag_frac = float(chunk_count) / matches_count
    except ZeroDivisionError:
        return 0.0
    penalty = gamma * frag_frac ** beta
    return (1 - penalty) * fmean