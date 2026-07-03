"""Transaction tokenizer.

The tokenizer is now NVIDIA's **flat causal-LM scheme** — every transaction
becomes ~12 tokens in one shared vocabulary and a card's history is a flat token
stream for next-token pretraining of a Llama decoder. The implementation lives in
``src/flat_tokenizer.py``; this module re-exports its public API so the pipeline's
``from src.tokenizer import ...`` sites keep working. (The previous field-split /
masked-feature scheme was replaced to match the NVIDIA transaction-FM blueprint.)
"""

from .flat_tokenizer import (  # noqa: F401
    FIELD_SPECS,
    OFFSETS,
    PRETRAIN_DROP,
    SEQ_LEN_BY_SCALE,
    SPECIALS,
    TOKENS_PER_TXN,
    VOCAB_SIZE,
    build_sequence,
    decode_tokens,
    encode_transactions,
    eval_normal_keep,
    make_tokenize_group_fn,
    tokenize_dataset,
    write_vocab,
)
