# Phase 12 v2 runtime-lock correction

The initial V3.2 freeze script instantiated `StrictJournalOllamaClient`, which correctly enforced the immutable Phase-10 runtime lock (Ollama 0.32.14). The current machine uses Ollama 0.34.0 but the exact same `llama3:8b` model digest:

`365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1`

The correction does **not** edit the Phase-10 protocol. Instead it introduces a separate Phase-11R/Phase-12-v2 confirmatory runtime lock and uses that one common runtime for both:

- Phase-11 V2 **method replay under the confirmatory runtime**
- Objective Selection V3.2

This keeps the paired comparison fair while preserving historical provenance. The V2 result must not be described as a literal replay of the original Ollama 0.32.14 runtime.
