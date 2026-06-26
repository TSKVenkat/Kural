# Kural — Experiments & Learnings

Running log of what we tried, what we got, and *why*. Two tasks: **ASR** (Tamil
transcription) and **SER** (speech-emotion recognition).

## Results at a glance

### ASR — Whisper fine-tune on Common Voice Tamil
| Model | Method | WER ↓ | CER ↓ | Verdict |
|-------|--------|------|------|---------|
| `Venky0411/whisper-small-ta-saaras` | full fine-tune (smoke run) | 58.1 | 19.5 | usable |
| `Venky0411/whisper-small-ta-saaras-v2` | full fine-tune (longer) | **35.5** | **5.9** | good |

ASR worked because it was a **true full fine-tune** — every weight updated.

### SER — emotion (the hard part)
| # | Setup | Data | Split | Acc | Macro-F1 | Notes |
|---|-------|------|-------|-----|----------|-------|
| 1 | Frozen Whisper enc + **mean-pool** head (`WhisperForAudioClassification`) | EmoTa, 5-class | speaker-independent | 0.256 | 0.211 | ~chance; `fear` F1 = 0 |
| 2 | Frozen Whisper enc + **attention-pool** + classifier | EmoTa, 4-class | speaker-independent | 0.380 | 0.302 | degenerate: `happy` F1 = 0, collapses to `sad` |
| 3 | Frozen **Whisper** enc + attention-pool | **English** (CREMA-D) | speaker-independent | _pending_ | _pending_ | tests Whisper's strong-language features |

Reference point: the **EmoTa authors** report **F1 ≈ 0.90** using **XGBoost / Random Forest
on hand-crafted acoustic features** — not deep nets. ([paper](https://aclanthology.org/2025.chipsal-1.19/))

## Key learnings (the *why*)

1. **Train vs fine-tune vs linear-probe matters enormously.**
   - ASR = full fine-tune → worked.
   - Emotion exp 1/2 = *frozen encoder, train only a head* = a **linear probe**. Almost
     nothing learns. This is the single biggest reason emotion underperformed.

2. **Mean-pooling over Whisper features is broken for short clips.** Whisper pads every
   input to **30 s** and the classifier mean-pools all 1500 frames — for a ~3 s clip,
   ~90% of the pooled vector is silence. → **attention pooling** (a learnable query that
   attends to salient frames + masks padding) fixed this: 0.256 → 0.380.

3. **But the ceiling is the frozen features, not the pooling.** Whisper's encoder is
   trained for **transcription**, so it encodes *words*, not *affect*. A classifier on top
   can't separate happy/sad if that signal is weak in the features. Two experiments
   confirm: better pooling helped, but it still collapses to a couple of classes.

4. **Is 936 samples enough? Yes — for the right method.** Small datasets favour
   **XGBoost on eGeMAPS/openSMILE features** (EmoTa authors: ~0.90) over fine-tuning giant
   models. Deep fine-tuning of frozen features on ~750 clips is the worst combo.

5. **Language matters for frozen Whisper.** Whisper is strongest on English, weaker on
   Tamil — so frozen Tamil features are extra-weak for emotion. Exp 3 tests the same design
   on English (CREMA-D) to isolate language vs. feature-type.

6. **Better backbones for frozen SER: wav2vec2 / HuBERT.** Self-supervised on raw audio →
   encode prosody/affect; HuBERT especially strong *when frozen*. `superb/hubert-base-superb-er`
   is already emotion-tuned. ([benchmark](https://arxiv.org/abs/2111.02735))

## Paths forward (ranked)
1. **XGBoost on eGeMAPS features** — proven ~0.90 on EmoTa, data-efficient, no GPU. Best ROI.
2. **Frozen HuBERT + attention pooling** — keeps the "freeze encoder" design, much better features.
3. **Frozen Whisper + attention on English** (exp 3) — fair test of Whisper's best language.
4. **Unfreeze the Whisper encoder** — higher ceiling but risks overfit on small data.

## Dataset notes (gotchas we hit)
- **EmoTa / TamilSER-DB** — 936 clips, 22 speakers, 5 emotions (`ang/fea/hap/neu/sad`),
  48 kHz mono. **Gated (needs approval)** → kept out of git, uploaded privately to Kaggle.
- **Common Voice Tamil** — Mozilla retired their HF datasets (Oct 2025); used the Parquet
  mirror `abar-uwc/tamil-common-voice_v21`. The new `datasets` (4.x) **dropped loading
  scripts** → must use Parquet-native repos.
- **Tamil HF emotion sets** — `meghana-007` loads only via the `refs/convert/parquet`
  branch (raw repo has a broken `audio_8.wav` size-mismatch); `Thanushs25` unrecoverable.
- **English SER** — `narad/ravdess` is a dead loading script → use `confit/ravdess-parquet`
  (config `fold1`). MSP-Podcast / IEMOCAP / MELD are gated or noisy; **CREMA-D**
  (`confit/cremad-parquet`, 91 actors) is the best clean open option.
- **hf_xet** "file size mismatch" downloads → set `HF_HUB_DISABLE_XET=1`.

## Notebooks
- `notebooks/train_whisper_tamil_colab.ipynb` — ASR fine-tune (Tamil).
- `notebooks/train_emotion_whisper_colab.ipynb` — Tamil SER (frozen Whisper + attention, EmoTa).
- `notebooks/train_emotion_english_whisper.ipynb` — English SER (same design, CREMA-D).
