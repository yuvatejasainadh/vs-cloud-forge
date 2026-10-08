# English Golden Test Dataset Structure

Place verified, labeled English speech audio test sets here for countermeasure evaluation:

```text
fixtures/
  english/
    real/       # Genuine/bonafide human English speech samples (.wav)
    spoof/      # Deepfake/synthetic/voice-converted English speech samples (.wav)
```

## Dataset Provenance Standards
- Only insert audio samples whose ground-truth provenance is verified (e.g., ASVspoof 2019/2021 logical access evaluation partition).
- Do not fabricate synthetic golden labels.
