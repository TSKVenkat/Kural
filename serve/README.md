# Kural web app (`serve/`)

Dedicated FastAPI app: fast local Tamil ASR via **faster-whisper** (CTranslate2 int8)
+ prosody-based emotion/style tagging, with a simple record/upload UI.

```
serve/
├── server.py          # FastAPI: serves the UI + POST /api/analyze
└── static/            # index.html · style.css · app.js  (vanilla JS, no build step)
```

## Run

```bash
pip install -r requirements.txt

# one-time: convert an HF Whisper checkpoint to CTranslate2 int8
ct2-transformers-converter --model Venky0411/whisper-small-ta-saaras-v2 \
  --output_dir ct2-model --quantization int8

CT2_MODEL=ct2-model uvicorn server:app --app-dir serve --host 0.0.0.0 --port 7860
```

Then open <http://localhost:7860>.

`CT2_MODEL` (env) points at the converted model directory (default
`/tmp/sarvam-svc/ct2-v2`). To expose a temporary public link:
`cloudflared tunnel --url http://127.0.0.1:7860`.

## API

`POST /api/analyze` — multipart form field `audio` (wav / mp3 / webm / opus; mic
recordings are decoded via librosa+ffmpeg). Returns:

```json
{
  "transcript": "…", "emotion": "neutral", "emotion_conf": 0.75,
  "style": {"pace": "slow", "energy": "medium", "pitch_variation": "dynamic", "overall": "neutral"},
  "duration_s": 8.21, "infer_s": 4.5, "segments": [ … ], "prosody": { … }
}
```

`GET /api/health` → `{"ok": true, "model": "<path>"}`.

> Needs `ffmpeg` on PATH for non-wav inputs. Emotion is a prosody heuristic, not a
> trained classifier.
