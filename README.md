# WSP Verify GitHub Action (`wsp-ci-action`)

[![WSP Verified](https://api.wsphub.dev/api/v1/badges/wsp-qwen3.5-0.8b-math-loop-exp8.svg)](https://wsphub.dev/patch/wsp-qwen3.5-0.8b-math-loop-exp8)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Continuous Behavioral, Safety & Cryptographic Regression Gate for Open-Weight Fine-Tunes, LoRAs, Model Merges & `.wsp` Patches.**

## Modes

1. **`fast-ci` (`$0.00` Free, `<30s` on CPU):**
   - Verifies `.wsp v2.1` `Ed25519` dual-artifact signatures + RFC 8785 canonical manifests against live [wsphub.dev](https://wsphub.dev) publisher keys & CRL revocations.
   - Enforces SPDX base-model license gates (`Apache-2.0`, `MIT`, `Llama`, `Gemma`) and blocks safety-removal / abliteration modifications.
   - Validates tensor shape/axis/adapter invariants (`Qwen3.5`, `Llama`, `Gemma`).
2. **`full-eval` (Metered via [Supernova](https://supernova.cool) Async Escrow with AEGIS 100% Refund-on-Failure):**
   - Runs multi-seed locked `MATH-500` / `GSM8K` / `IFEval` splits, secret perturbed `WSP-Holdout`, and `StrongREJECT` + `XSTest` safety non-inferiority gates.
   - Posts a signed Markdown scorecard & SHA-256 transparency attestation hash directly to the Pull Request summary.

---

## Usage

### Free Pull-Request Gate (`fast-ci`)
```yaml
name: Model Patch Verification
on: [pull_request]

jobs:
  wsp-fast-ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: WSP Fast-CI Gate
        uses: GhazwanAlemara/wsp-ci-action@v2
        with:
          mode: fast-ci
          hf-repo-id: Qwen/Qwen3.5-0.8B
          base-revision: 2fc06364715b967f1860aea9cf38778875588b17
          patch-path: ./patches/candidate.wsp
```

### Release Attestation Gate (`full-eval`)
```yaml
      - name: WSP Verify Full-Eval Attestation
        uses: GhazwanAlemara/wsp-ci-action@v2
        with:
          mode: full-eval
          hf-repo-id: Qwen/Qwen3.5-0.8B
          base-revision: 2fc06364715b967f1860aea9cf38778875588b17
          patch-path: ./patches/candidate.wsp
          param-count-b: "0.8"
          supernova-api-key: ${{ secrets.SUPERNOVA_API_KEY }}
```

---

## License

Apache License 2.0. Powered by [wsphub.dev](https://wsphub.dev) & [supernova.cool](https://supernova.cool).
