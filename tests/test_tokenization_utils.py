import pytest
from tokenization_utils import tokenize_pretokenized_input
from transformers import AutoTokenizer


@pytest.mark.parametrize(
    ("tokenizer_name", "tokenizer_kwargs", "text", "words", "expected_tokens", "expected_offsets"),
    [
        (
            "bert-base-cased",  # wordpiece-based tokenizer
            {"use_fast": False},
            "Barrie enrolled at the University of Edinburgh where he wrote drama reviews for the Edinburgh Evening Courant.",  # noqa: E501
            ["Barrie", "enrolled", "at", "the", "University", "of", "Edinburgh", "where", "he", "wrote", "drama", "reviews", "for", "the", "Edinburgh", "Evening", "Courant", "."],  # noqa: E501
            ["Barr", "##ie", "enrolled", "at", "the", "University", "of", "Edinburgh", "where", "he", "wrote", "drama", "reviews", "for", "the", "Edinburgh", "Evening", "Co", "##ura", "##nt", "."],  # noqa: E501
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 17],
        ),
        (
            "bert-base-cased",  # wordpiece-based tokenizer
            {"use_fast": True},
            "Barrie enrolled at the University of Edinburgh where he wrote drama reviews for the Edinburgh Evening Courant.",  # noqa: E501
            ["Barrie", "enrolled", "at", "the", "University", "of", "Edinburgh", "where", "he", "wrote", "drama", "reviews", "for", "the", "Edinburgh", "Evening", "Courant", "."],  # noqa: E501
            ["Barr", "##ie", "enrolled", "at", "the", "University", "of", "Edinburgh", "where", "he", "wrote", "drama", "reviews", "for", "the", "Edinburgh", "Evening", "Co", "##ura", "##nt", "."],  # noqa: E501
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 17],
        ),
        (
            "tohoku-nlp/bert-base-japanese",  # wordpiece-based tokenizer
            {"use_fast": False},
            "バリーはエディンバラ大学に入学し、エディンバラ・イーブニング・クーラントに演劇批評を執筆した。",  # noqa: E501
            ["バリー", "は", "エディンバラ大学", "に", "入学", "し", "、", "エディンバラ", "・", "イーブニング", "・", "クーラント", "に", "演劇", "批評", "を", "執筆", "し", "た", "。"],  # noqa: E501
            ["バリー", "は", "エディンバラ", "大学", "に", "入学", "し", "、", "エディンバラ", "・", "イー", "##ブ", "##ニング", "・", "クー", "##ラント", "に", "演劇", "批評", "を", "執筆", "し", "た", "。"],  # noqa: E501
            [0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # noqa: E501
        ),
        (
            "tohoku-nlp/bert-base-japanese",  # wordpiece-based tokenizer
            {"use_fast": True},
            "バリーはエディンバラ大学に入学し、エディンバラ・イーブニング・クーラントに演劇批評を執筆した。",  # noqa: E501
            ["バリー", "は", "エディンバラ大学", "に", "入学", "し", "、", "エディンバラ", "・", "イーブニング", "・", "クーラント", "に", "演劇", "批評", "を", "執筆", "し", "た", "。"],  # noqa: E501
            ["バリー", "は", "エディンバラ", "大学", "に", "入学", "し", "、", "エディンバラ", "・", "イー", "##ブ", "##ニング", "・", "クー", "##ラント", "に", "演劇", "批評", "を", "執筆", "し", "た", "。"],  # noqa: E501
            [0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # noqa: E501
        ),
        (
            "roberta-base",  # BPE-based (w/o sentencepiece) tokenizer
            {"use_fast": False, "add_prefix_space": False},
            "Barrie enrolled at the University of Edinburgh where he wrote drama reviews for the Edinburgh Evening Courant.",  # noqa: E501
            ["Barrie", "enrolled", "at", "the", "University", "of", "Edinburgh", "where", "he", "wrote", "drama", "reviews", "for", "the", "Edinburgh", "Evening", "Courant", "."],  # noqa: E501
            ["Bar", "rie", "Ġenrolled", "Ġat", "Ġthe", "ĠUniversity", "Ġof", "ĠEdinburgh", "Ġwhere", "Ġhe", "Ġwrote", "Ġdrama", "Ġreviews", "Ġfor", "Ġthe", "ĠEdinburgh", "ĠEvening", "ĠCour", "ant", "."],  # noqa: E501
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17],
        ),
        (
            "roberta-base",  # BPE-based (w/o sentencepiece) tokenizer
            {"use_fast":True, "add_prefix_space": False},
            "Barrie enrolled at the University of Edinburgh where he wrote drama reviews for the Edinburgh Evening Courant.",  # noqa: E501
            ["Barrie", "enrolled", "at", "the", "University", "of", "Edinburgh", "where", "he", "wrote", "drama", "reviews", "for", "the", "Edinburgh", "Evening", "Courant", "."],  # noqa: E501
            ["Bar", "rie", "Ġenrolled", "Ġat", "Ġthe", "ĠUniversity", "Ġof", "ĠEdinburgh", "Ġwhere", "Ġhe", "Ġwrote", "Ġdrama", "Ġreviews", "Ġfor", "Ġthe", "ĠEdinburgh", "ĠEvening", "ĠCour", "ant", "."],  # noqa: E501
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17],
        ),
        (
            "roberta-base",  # BPE-based (w/o sentencepiece) tokenizer
            {"use_fast": False, "add_prefix_space": True},
            "Barrie enrolled at the University of Edinburgh where he wrote drama reviews for the Edinburgh Evening Courant.",  # noqa: E501
            ["Barrie", "enrolled", "at", "the", "University", "of", "Edinburgh", "where", "he", "wrote", "drama", "reviews", "for", "the", "Edinburgh", "Evening", "Courant", "."],  # noqa: E501
            ["ĠBar", "rie", "Ġenrolled", "Ġat", "Ġthe", "ĠUniversity", "Ġof", "ĠEdinburgh", "Ġwhere", "Ġhe", "Ġwrote", "Ġdrama", "Ġreviews", "Ġfor", "Ġthe", "ĠEdinburgh", "ĠEvening", "ĠCour", "ant", "."],  # noqa: E501
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17],
        ),
        (
            "roberta-base",  # BPE-based (w/o sentencepiece) tokenizer
            {"use_fast":True, "add_prefix_space": True},
            "Barrie enrolled at the University of Edinburgh where he wrote drama reviews for the Edinburgh Evening Courant.",  # noqa: E501
            ["Barrie", "enrolled", "at", "the", "University", "of", "Edinburgh", "where", "he", "wrote", "drama", "reviews", "for", "the", "Edinburgh", "Evening", "Courant", "."],  # noqa: E501
            ["ĠBar", "rie", "Ġenrolled", "Ġat", "Ġthe", "ĠUniversity", "Ġof", "ĠEdinburgh", "Ġwhere", "Ġhe", "Ġwrote", "Ġdrama", "Ġreviews", "Ġfor", "Ġthe", "ĠEdinburgh", "ĠEvening", "ĠCour", "ant", "."],  # noqa: E501
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17],
        ),
    ],
)  # fmt: skip
def test_tokenize_pretokenized_input(
    tokenizer_name, tokenizer_kwargs, text, words, expected_tokens, expected_offsets
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    input_ids, offsets = tokenize_pretokenized_input(text, words, tokenizer)
    assert tokenizer.convert_ids_to_tokens(input_ids) == expected_tokens
    assert offsets == expected_offsets
