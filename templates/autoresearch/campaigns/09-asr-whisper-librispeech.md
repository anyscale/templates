# Campaign 09 — ASR (Whisper) on LibriSpeech: reproduce WER, then a fine-tune/cost beat

> **Status: SEED (not started).** The speech domain. A near-free eval gate and a clean Ray
> Train fine-tune + Ray Data batch-transcription story.

## 1. One-liner

Reproduce Whisper's published LibriSpeech WER through a Ray-parallelized eval, then beat it on a
target domain via fine-tune, or beat the reference's transcription cost via Ray Data batch
inference — WER held as the invariant.

## 2. Why this is a good autoresearch testbed

- **What makes it clean:** shipped inference/eval + checkpoints; text-normalization makes a clean,
  well-known reproduction gate; domain fine-tune and decoding are concrete levers.
- **Ray substrate it exercises:** Ray Data batch transcription over large audio corpora
  (audio-decode CPU stage → GPU ASR, scale-to-zero) + Ray Train for the domain fine-tune.

## 3. Reference

- **Repo:** `github.com/openai/whisper` — **MIT**, maintained (inference/eval). HF `transformers`
  (Apache-2.0) is the standard 2026 path for fine-tuning. **⚠️ `facebookresearch/fairseq`
  (wav2vec2) is archived (2026-03) — do not build on it;** prefer Whisper to dodge the
  LM-decoding/flashlight trap.
- **Ships:** Whisper = **inference + eval only, no training code**; checkpoints tiny→large-v3
  (+ HF mirror `openai/whisper-large-v3`). Fine-tuning is a HF `transformers` recipe.
- **Ray-native?** No — plain PyTorch (Whisper); fine-tune via HF Trainer + accelerate.
- **Reproduction landscape:** heavily reproduced. **Text normalization is load-bearing** (WER
  swings ~50% with/without the paper's `EnglishTextNormalizer`); test-other ≈ 2× test-clean.

## 4. The gate

- **Decision metric:** WER on LibriSpeech test-clean and test-other (report the **pair**, never
  one).
- **Published to match (the canonical gate is `whisper/notebooks/LibriSpeech.ipynb`, which
  applies the paper's normalizer):** large-v2 test-clean ≈ **2.5**. (wav2vec2 trap, for
  reference: the famous 1.8/3.3 needs a Transformer-LM decoder; shipped-checkpoint greedy is
  2.8/6.3 — another reason to use Whisper.)
- **Guard metrics:** test-other WER, RTF (real-time factor) / throughput, $/1000-hours-audio.
- **Artifact gate:** run the shipped notebook on the shipped checkpoint → match ≈2.5. **Pipeline
  gate:** reproduce it through the Ray Data batch-transcription harness with the same normalizer.

## 5. Pinned eval

The LibriSpeech test-clean/test-other splits + the **exact `EnglishTextNormalizer`** + the WER
implementation + decoding params (beam size, temperature, language) + the Whisper checkpoint
version. Content-hash the audio manifest. Dense — point estimates stable, but report a CI over
utterances.

## 6. Data

- **LibriSpeech** ~1000h, **CC BY 4.0, no gating/login** (openslr.org/12 or HF
  `openslr/librispeech_asr`). (LM/lexicon at openslr.org/11 only if doing wav2vec2 — skip for
  Whisper.)
- **Input audit:** audio resampling to 16kHz, the log-mel front-end params, chunking of long
  audio, and the **text normalization** (the single biggest silent-WER lever).

## 7. Rayification

| Stage | What | Ray lib |
|---|---|---|
| ingest | audio files → 16kHz log-mel | Ray Data (audio processing) |
| **transcribe** | Whisper forward over utterances | **Ray Data `map_batches` + ActorPoolStrategy** |
| score | normalize + WER | driver |
| fine-tune (push) | HF Whisper fine-tune on a domain | Ray Train (`TorchTrainer` / HF integration) |

## 8. Fidelity ladder

- **smoke:** a handful of utterances, whisper-tiny, 1 GPU/CPU — transcribe→normalize→WER runs.
- **proxy:** test-clean with whisper-base/small; rank fine-tune / decoding ideas.
- **full:** test-clean + test-other with large-v2/v3 (the gate); then the domain fine-tune push.
- **Proxy axis:** model size (tiny→large) + eval subset. **Rare-signal trap:** test-clean is
  easy; a decoding/fine-tune win on clean may not hold on test-other (the accented/noisy
  distribution) — always calibrate against the harder split.

## 9. "Beat it" hypotheses

1. **Domain fine-tune** — fine-tune Whisper on a target domain (e.g. accented/contact-center
   audio) and beat the zero-shot WER on that domain while not regressing LibriSpeech. Flag
   `finetune`.
2. **Cost beat** — Ray Data batch transcription at materially lower $/1000-hours vs the naive
   single-GPU loop, WER held within noise (the batch-inference cost story). Flag `batch_pipeline`.
3. **Decoding** — beam/temperature/normalization variants; the "readout" analog for ASR. Flag
   `decode`.

## 10. Budget

- **Wave 1** for the eval reproduction (near-free); **Wave 2** for the fine-tune push.
- **GPU-hours:** smoke <0.2 · reproduction (test-clean+other, large-v2) ~1–2 · proxy ~5 · full
  fine-tune push ~15–30 · **total ~25–40**.
- **GPU tier:** A10G/L4 for eval, A100/A10G for fine-tune; spot ON. **~$30–100 on-demand.**
- **Approval:** envelope (Wave 1); + each fine-tune full run (Wave 2).

## 11. Controls

- **Zero-shot Whisper is its own baseline** for the fine-tune (must not regress it).
- **Normalization ablation** as a control: report WER with and without the normalizer so the
  number is unambiguous (this is where most ASR repro disputes live).
- **Leak-audit trigger:** WER far below 2.5 on test-clean → a normalization or split bug (e.g.
  scoring against reference text the model was prompted with).

## 12. Risks

Text normalization is THE reproducibility trap — pin it and report both normalized/unnormalized.
Whisper ships no training code (fine-tune is a HF recipe — budget it as real integration work).
Avoid the archived fairseq/wav2vec2 LM-decoding path. test-clean is near-saturated — the domain
fine-tune and the cost beat are the durable claims, not shaving LibriSpeech WER.
