"""`wsp` Command-Line Interface — Inspect, Verify, Diff, Apply, Revert, Reproduce, Export & Submit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import httpx

from wsp_cli.engine import (
    apply_patch,
    compute_directory_diff_patch,
    revert_patch,
)
from wsp_cli.exporter import export_gguf, export_ollama
from wsp_cli.package_format import inspect_wsp_archive, read_wsp_archive, verify_wsp_archive
from wsp_cli.telemetry import emit_opt_in_event


def _load_trusted_keys(
    key_id: Optional[str] = None,
    pubkey_hex: Optional[str] = None,
    keys_json_path: Optional[Path] = None,
) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    env_keys = os.getenv("WSP_TRUSTED_KEYS_JSON", "").strip()
    if env_keys:
        keys.update(json.loads(env_keys))
    if keys_json_path and keys_json_path.exists():
        data = json.loads(keys_json_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "key_id" in data and "public_key_hex" in data:
            keys[data["key_id"]] = data["public_key_hex"]
        elif isinstance(data, dict):
            keys.update(data)
    if key_id and pubkey_hex:
        keys[key_id] = pubkey_hex
    return keys


def _resolve_patch_input(patch_arg: str, api_url: str) -> Path | bytes:
    local = Path(patch_arg)
    if local.exists() and local.is_file():
        return local
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(f"{api_url.rstrip('/')}/api/v1/patches/{patch_arg}/download")
        resp.raise_for_status()
        return resp.content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsp",
        description="WSP Hub CLI — Signed Behavioral & Config Patches for Open-Weight Models",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect a .wsp v2.1 archive manifest")
    p_inspect.add_argument("patch", help="Path to .wsp file")

    p_verify = sub.add_parser("verify", help="Verify Ed25519 signature & RFC 8785 manifest integrity")
    p_verify.add_argument("patch", help="Path to .wsp file")
    p_verify.add_argument("--key-id", default="ed25519-wsphub-official-01")
    p_verify.add_argument("--pubkey-hex", help="Trusted Ed25519 64-char hex public key")
    p_verify.add_argument("--keys-file", type=Path, help="JSON file of trusted key_id -> pubkey_hex")

    p_diff = sub.add_parser("diff", help="Create a signed .wsp v2.1 patch from two model directories")
    p_diff.add_argument("base_dir", type=Path)
    p_diff.add_argument("modified_dir", type=Path)
    p_diff.add_argument("--out", type=Path, required=True)
    p_diff.add_argument("--patch-id", required=True)
    p_diff.add_argument("--hf-repo", required=True)
    p_diff.add_argument("--revision", required=True)
    p_diff.add_argument("--adapter", default="qwen3_5")
    p_diff.add_argument("--key-id", default="ed25519-wsphub-official-01")
    p_diff.add_argument("--private-seed-hex", default=os.getenv("WSP_SIGNING_PRIVATE_KEY", ""))

    p_apply = sub.add_parser("apply", help="Apply a verified .wsp patch via atomic Copy-on-Write")
    p_apply.add_argument("model", help="Local model directory or Hugging Face repo ID")
    p_apply.add_argument("--patch", required=True, help="Local .wsp path or registry patch_id")
    p_apply.add_argument("--out", type=Path, required=True, help="Target output directory")
    p_apply.add_argument("--key-id", default="ed25519-wsphub-official-01")
    p_apply.add_argument("--pubkey-hex", help="Trusted Ed25519 public key hex")
    p_apply.add_argument("--keys-file", type=Path)
    p_apply.add_argument("--api-url", default=os.getenv("WSP_API_URL", "https://api.wsphub.dev"))
    p_apply.add_argument("--opt-in-telemetry", action="store_true")

    p_revert = sub.add_parser("revert", help="Bit-exactly revert a patched directory using .wsp_rollback")
    p_revert.add_argument("patched_dir", type=Path)

    p_repro = sub.add_parser("reproduce", help="Emit deterministic reproduction recipe for a .wsp patch")
    p_repro.add_argument("patch", type=Path)
    p_repro.add_argument("--sample", type=int, default=0)

    p_bench = sub.add_parser("benchmark", help="Print or invoke verification benchmark protocol")
    p_bench.add_argument("model_dir", type=Path)
    p_bench.add_argument("--patch", type=Path, required=True)

    p_gguf = sub.add_parser("export-gguf", help="Export patched model to quantized GGUF")
    p_gguf.add_argument("patched_dir", type=Path)
    p_gguf.add_argument("--out", type=Path, required=True)
    p_gguf.add_argument("--quant", default="Q4_K_M")

    p_ollama = sub.add_parser("export-ollama", help="Generate verified Ollama Modelfile")
    p_ollama.add_argument("patched_dir", type=Path)
    p_ollama.add_argument("--patch", type=Path, required=True)
    p_ollama.add_argument("--tag", required=True)
    p_ollama.add_argument("--register", action="store_true")

    p_submit = sub.add_parser("submit", help="Submit a candidate .wsp package to wsphub.dev verification queue")
    p_submit.add_argument("candidate", type=Path)
    p_submit.add_argument("--title", required=True)
    p_submit.add_argument("--summary", required=True)
    p_submit.add_argument("--defect-category", default="reasoning_loop")
    p_submit.add_argument("--api-url", default=os.getenv("WSP_API_URL", "https://api.wsphub.dev"))
    p_submit.add_argument("--token", default=os.getenv("WSP_AUTH_TOKEN", ""))

    p_scan = sub.add_parser("scan", help="Run zero-GPU diagnostic check on a Hugging Face model repository")
    p_scan.add_argument("hf_repo", help="Hugging Face repository ID (e.g. Qwen/Qwen3.5-0.8B)")
    p_scan.add_argument("--api-url", default=os.getenv("WSP_API_URL", "https://api.wsphub.dev"))
    p_scan.add_argument("--fail-on-defect", action="store_true", help="Exit with code 2 if defects are detected")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        info = inspect_wsp_archive(Path(args.patch))
        print(json.dumps(info, indent=2))
        return 0

    if args.command == "verify":
        keys = _load_trusted_keys(args.key_id, args.pubkey_hex, args.keys_file)
        ok, msg, arc = verify_wsp_archive(Path(args.patch), trusted_keys=keys)
        print(json.dumps({"verified": ok, "message": msg, "patch_id": arc.manifest.get("patch_id")}, indent=2))
        return 0 if ok else 1

    if args.command == "diff":
        if not args.private_seed_hex:
            raise ValueError("--private-seed-hex or WSP_SIGNING_PRIVATE_KEY is required to sign diff")
        _, signed_manifest, sha = compute_directory_diff_patch(
            base_dir=args.base_dir,
            modified_dir=args.modified_dir,
            patch_id=args.patch_id,
            hf_repo_id=args.hf_repo,
            base_repo_revision=args.revision,
            architecture_adapter=args.adapter,
            publisher_key_id=args.key_id,
            private_seed_hex=args.private_seed_hex,
            out_path=args.out,
        )
        print(
            json.dumps(
                {
                    "patch_id": signed_manifest["patch_id"],
                    "out": str(args.out),
                    "archive_sha256": sha,
                    "tensor_modifications": len(signed_manifest["tensor_modifications"]),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "apply":
        keys = _load_trusted_keys(args.key_id, args.pubkey_hex, args.keys_file)
        patch_src = _resolve_patch_input(args.patch, args.api_url)
        res = apply_patch(
            model_dir_or_repo=args.model,
            patch_source=patch_src,
            out_dir=args.out,
            trusted_keys=keys,
        )
        emit_opt_in_event(
            "CLI_OPT_IN_APPLY",
            patch_id=res["patch_id"],
            explicit_flag=args.opt_in_telemetry,
            api_base_url=args.api_url,
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "revert":
        res = revert_patch(args.patched_dir)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "reproduce":
        arc = read_wsp_archive(args.patch)
        compat = arc.manifest.get("compatibility", {})
        protocol = arc.manifest.get("reproducibility_and_evaluation", {}).get("sampling_protocol", {})
        recipe = {
            "patch_id": arc.manifest.get("patch_id"),
            "hf_repo_id": compat.get("hf_repo_id"),
            "base_repo_revision": compat.get("base_repo_revision"),
            "sample_index": args.sample,
            "vllm_container": protocol.get("vllm_container_digest", "vllm/vllm-openai:v0.33.0"),
            "sampling_protocol": protocol,
            "steps": [
                f"wsp apply {compat.get('hf_repo_id')} --patch {args.patch} --out ./repro_patched",
                f"docker run --gpus all -v ./repro_patched:/model {protocol.get('vllm_container_digest', 'vllm/vllm-openai:v0.33.0')} --model /model --dtype bfloat16",
            ],
        }
        print(json.dumps(recipe, indent=2))
        return 0

    if args.command == "benchmark":
        arc = read_wsp_archive(args.patch)
        print(
            json.dumps(
                {
                    "model_dir": str(args.model_dir),
                    "patch_id": arc.manifest.get("patch_id"),
                    "scorecard": arc.manifest.get("reproducibility_and_evaluation", {}),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "export-gguf":
        res = export_gguf(args.patched_dir, args.out, quantization=args.quant)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "export-ollama":
        arc = read_wsp_archive(args.patch)
        res = export_ollama(
            args.patched_dir,
            manifest=arc.manifest,
            ollama_tag=args.tag,
            register_with_daemon=args.register,
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "submit":
        arc = read_wsp_archive(args.candidate)
        headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
        with httpx.Client(timeout=30.0, follow_redirects=False) as client:
            resp = client.post(
                f"{args.api_url.rstrip('/')}/api/v1/candidates",
                headers=headers,
                json={
                    "patch_id": arc.manifest.get("patch_id"),
                    "title": args.title,
                    "summary": args.summary,
                    "defect_category": args.defect_category,
                    "manifest": arc.manifest,
                    "archive_hex": args.candidate.read_bytes().hex(),
                },
            )
            resp.raise_for_status()
            print(json.dumps(resp.json(), indent=2))
        return 0

    if args.command == "scan":
        repo_clean = str(args.hf_repo).strip().strip("/")
        if "/" not in repo_clean:
            raise ValueError("hf_repo must be in '<org>/<model>' format (e.g. Qwen/Qwen3.5-0.8B)")
        org, model_name = repo_clean.split("/", 1)
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            resp = client.get(f"{args.api_url.rstrip('/')}/api/v1/diagnose/{org}/{model_name}")
            resp.raise_for_status()
            report = resp.json()
            print(json.dumps(report, indent=2))
        if args.fail_on_defect and report.get("has_verifiable_defect"):
            return 2
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
