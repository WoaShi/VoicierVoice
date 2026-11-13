#!/usr/bin/env python3
"""Download, inspect, and smoke-test ONNX models for VoicierVoice Phase 1.

Downloads:
  - VoxLock v2 (arbazsiddiqui/voxlock on Hugging Face: onnx/v2/* INT8 & FP16 variants)
  - Silero-VAD (snakers4/silero-vad official silero_vad.onnx)
  - GTCRN (k2-fsa/sherpa-onnx speech-enhancement-models gtcrn_simple.onnx,
           Xiaobin-Rong/gtcrn simplified streaming model with embedded metadata)

Inspects all downloaded .onnx files with onnxruntime (and onnx if available),
runs dummy inference smoke tests on CPU (and CUDA if available), and writes
Assets/Models/model_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "Assets" / "Models"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"

VOXLOCK_REPO = "arbazsiddiqui/voxlock"
VOXLOCK_FILES = [
    ("onnx/v2/enroll.onnx", MODELS_DIR / "voxlock" / "v2" / "enroll.onnx"),
    ("onnx/v2/chunk_int8.onnx", MODELS_DIR / "voxlock" / "v2" / "chunk_int8.onnx"),
    ("onnx/v2/chunk_int8_full.onnx", MODELS_DIR / "voxlock" / "v2" / "chunk_int8_full.onnx"),
    ("onnx/v2/enroll_fp16.onnx", MODELS_DIR / "voxlock" / "v2" / "enroll_fp16.onnx"),
    ("onnx/v2/chunk_fp16.onnx", MODELS_DIR / "voxlock" / "v2" / "chunk_fp16.onnx"),
]

HTTP_MODELS = [
    {
        "id": "silero_vad",
        "url": "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
        "dest": MODELS_DIR / "silero-vad" / "silero_vad.onnx",
        "source": "https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/data/silero_vad.onnx",
    },
    {
        "id": "gtcrn_simple",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx",
        "dest": MODELS_DIR / "gtcrn" / "gtcrn_simple.onnx",
        "source": "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models",
    },
]

# Optional fallback conda python that has `onnx` + `onnxruntime-gpu` installed locally
CUDA_ENV_PYTHON = Path(r"D:\Software\anaconda3\envs\fc\python.exe")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_models(force: bool = False) -> dict[str, dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    import requests

    sources: dict[str, dict[str, Any]] = {}

    for remote_path, dest in VOXLOCK_FILES:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if force or not dest.exists() or dest.stat().st_size == 0:
            log(f"[download] HF {VOXLOCK_REPO}/{remote_path} -> {dest.relative_to(ROOT)}")
            cached = Path(hf_hub_download(repo_id=VOXLOCK_REPO, filename=remote_path))
            dest.write_bytes(cached.read_bytes())
        else:
            log(f"[skip] exists: {dest.relative_to(ROOT)}")
        sources[str(dest.relative_to(ROOT)).replace("\\", "/")] = {
            "source_repo": f"https://huggingface.co/{VOXLOCK_REPO}",
            "remote_path": remote_path,
        }

    for item in HTTP_MODELS:
        dest: Path = item["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if force or not dest.exists() or dest.stat().st_size == 0:
            log(f"[download] {item['url']} -> {dest.relative_to(ROOT)}")
            with requests.get(item["url"], stream=True, timeout=60) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        else:
            log(f"[skip] exists: {dest.relative_to(ROOT)}")
        sources[str(dest.relative_to(ROOT)).replace("\\", "/")] = {
            "source_repo": item["source"],
            "download_url": item["url"],
        }

    return sources


def ort_type_to_numpy(ort_type: str):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(ort_type, np.float32)


def inspect_onnx_protobuf(path: Path) -> dict[str, Any]:
    """Extract IR version, opsets, and symbolic tensor shapes using `onnx` if available."""
    try:
        import onnx
    except ImportError:
        return {"available": False}

    model = onnx.load(str(path), load_external_data=False)
    opsets = [
        {"domain": op.domain or "ai.onnx", "version": int(op.version)}
        for op in model.opset_import
    ]
    metadata_props = {p.key: p.value for p in model.metadata_props}

    def parse_value_info(vi):
        t = vi.type.tensor_type
        elem_type = onnx.TensorProto.DataType.Name(t.elem_type)
        dims = []
        if t.HasField("shape"):
            for d in t.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(int(d.dim_value))
                elif d.HasField("dim_param"):
                    dims.append(d.dim_param)
                else:
                    dims.append(None)
        return {"name": vi.name, "elem_type": elem_type, "shape": dims}

    init_names = {i.name for i in model.graph.initializer}
    inputs = [parse_value_info(i) for i in model.graph.input if i.name not in init_names]
    outputs = [parse_value_info(o) for o in model.graph.output]

    return {
        "available": True,
        "onnx_version": onnx.__version__,
        "ir_version": int(model.ir_version),
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "model_version": int(model.model_version),
        "opsets": opsets,
        "metadata_props": metadata_props,
        "inputs": inputs,
        "outputs": outputs,
        "node_count": len(model.graph.node),
    }


def make_dummy_inputs(model_key: str, inputs_meta: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Create realistic dummy tensors matching exact inspected model inputs."""
    feeds: dict[str, np.ndarray] = {}

    if "voxlock" in model_key and "enroll" in model_key:
        # VoxLock enroll takes (batch, samples) at 16 kHz, 3 seconds = 48000 samples
        for inp in inputs_meta:
            dt = ort_type_to_numpy(inp["type"])
            feeds[inp["name"]] = np.zeros((1, 48000), dtype=dt)
        return feeds

    if "voxlock" in model_key and "chunk" in model_key:
        # VoxLock chunk: wav (1, 512 + (K-1)*128); K=4 -> 896 samples
        # enroll_n (257, 192), k (6, 2, 16, 128), v (6, 2, 16, 128), h (6, 2, 128, 16), c (6, 2, 128, 16)
        for inp in inputs_meta:
            dt = ort_type_to_numpy(inp["type"])
            name = inp["name"]
            shape = inp["shape"]
            resolved = []
            for idx, dim in enumerate(shape):
                if isinstance(dim, int) and dim > 0:
                    resolved.append(dim)
                elif name == "wav" and idx == 0:
                    resolved.append(1)
                elif name == "wav" and idx == 1:
                    resolved.append(896)  # 4 frames: 512 + 3*128 = 896 samples
                elif name == "enroll_n" and idx == 1:
                    resolved.append(192)  # typical 3s enrollment frames (~298) or dynamic F_e
                elif idx == 0:
                    resolved.append(1)
                else:
                    resolved.append(1)
            feeds[name] = np.zeros(tuple(resolved), dtype=dt)
        return feeds

    if "silero_vad" in model_key:
        # Silero VAD v5/v6 at 16 kHz: 512 hop + 64 prepended context = 576 samples -> input (1, 576), state (2, 1, 128), sr () scalar int64=16000
        for inp in inputs_meta:
            dt = ort_type_to_numpy(inp["type"])
            name = inp["name"]
            shape = inp["shape"]
            if name == "sr":
                if len(shape) == 0:
                    feeds[name] = np.array(16000, dtype=np.int64)
                else:
                    feeds[name] = np.array([16000], dtype=np.int64)
            elif name == "input":
                feeds[name] = np.zeros((1, 576), dtype=dt)
            else:
                resolved = [d if isinstance(d, int) and d > 0 else 1 for d in shape]
                feeds[name] = np.zeros(tuple(resolved), dtype=dt)
        return feeds

    if "gtcrn" in model_key:
        for inp in inputs_meta:
            dt = ort_type_to_numpy(inp["type"])
            resolved = [d if isinstance(d, int) and d > 0 else 1 for d in inp["shape"]]
            feeds[inp["name"]] = np.zeros(tuple(resolved), dtype=dt)
        return feeds

    for inp in inputs_meta:
        dt = ort_type_to_numpy(inp["type"])
        resolved = [d if isinstance(d, int) and d > 0 else 1 for d in inp["shape"]]
        feeds[inp["name"]] = np.zeros(tuple(resolved), dtype=dt)
    return feeds


def run_ort_inspection(model_path: Path, providers: list[str]) -> dict[str, Any]:
    rel = str(model_path.relative_to(ROOT)).replace("\\", "/")
    so = ort.SessionOptions()
    so.log_severity_level = 3

    result: dict[str, Any] = {
        "ort_version": ort.__version__,
        "requested_providers": providers,
    }

    try:
        sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    except Exception as e:
        result["session_ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["session_ok"] = True
    result["active_providers"] = sess.get_providers()

    mm = sess.get_modelmeta()
    result["model_meta"] = {
        "producer_name": mm.producer_name,
        "graph_name": mm.graph_name,
        "domain": mm.domain,
        "description": mm.description,
        "graph_description": mm.graph_description,
        "version": int(mm.version),
        "custom_metadata_map": dict(mm.custom_metadata_map),
    }

    inputs_meta = [
        {"name": i.name, "type": i.type, "shape": list(i.shape)}
        for i in sess.get_inputs()
    ]
    outputs_meta = [
        {"name": o.name, "type": o.type, "shape": list(o.shape)}
        for o in sess.get_outputs()
    ]
    result["inputs"] = inputs_meta
    result["outputs"] = outputs_meta

    feeds = make_dummy_inputs(rel.lower(), inputs_meta)
    result["dummy_input_shapes"] = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)}
        for k, v in feeds.items()
    }

    try:
        outs = sess.run(None, feeds)
        result["inference_ok"] = True
        result["dummy_output_shapes"] = {
            o["name"]: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            for o, arr in zip(outputs_meta, outs)
        }
    except Exception as e:
        result["inference_ok"] = False
        result["inference_error"] = f"{type(e).__name__}: {e}"

    return result


def known_audio_params(rel_path: str, custom_meta: dict[str, str]) -> dict[str, Any]:
    """Attach verified audio DSP parameters from model metadata or official model source."""
    p = rel_path.lower()
    if "gtcrn" in p:
        return {
            "sample_rate": int(custom_meta.get("sample_rate", 16000)),
            "n_fft": int(custom_meta.get("n_fft", 512)),
            "hop_length": int(custom_meta.get("hop_length", 256)),
            "window_length": int(custom_meta.get("window_length", 512)),
            "window_type": custom_meta.get("window_type", "hann_sqrt"),
            "freq_bins": 257,
            "domain": "stft_complex_frame",
            "frame_duration_ms": 16.0,
            "window_duration_ms": 32.0,
        }
    if "silero_vad" in p:
        return {
            "supported_sample_rates": [8000, 16000],
            "sample_rate": 16000,
            "window_size_samples_16k": 512,
            "context_samples_16k": 64,
            "effective_model_input_samples_16k": 576,
            "window_size_samples_8k": 256,
            "frame_duration_ms": 32.0,
            "domain": "time_domain_pcm_float32",
        }
    if "voxlock" in p and "enroll" in p:
        return {
            "sample_rate": 16000,
            "min_enroll_samples": 48000,
            "min_enroll_seconds": 3.0,
            "fbank_win_length_samples": 400,
            "fbank_hop_length_samples": 160,
            "fbank_num_mel_bins": 80,
            "stft_n_fft": 512,
            "stft_hop_length": 128,
            "domain": "time_domain_pcm_float32",
        }
    if "voxlock" in p and "chunk" in p:
        return {
            "sample_rate": 16000,
            "n_fft": 512,
            "hop_length": 128,
            "reflect_pad_samples": 256,
            "chunk_formula_samples": "512 + (K - 1) * 128",
            "overlap_samples": 384,
            "algorithmic_latency_ms": 24.0,
            "window_type": "hann_periodic_squared_ola",
            "output_gain_multiplier": -0.5,
            "domain": "time_domain_pcm_float32",
        }
    return {}


def inspect_all(sources: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    onnx_files = sorted(MODELS_DIR.rglob("*.onnx"))
    avail_providers = ort.get_available_providers()
    models_out: list[dict[str, Any]] = []

    for path in onnx_files:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        log(f"\n=== Inspecting {rel} ({path.stat().st_size / (1024*1024):.2f} MB) ===")
        pb = inspect_onnx_protobuf(path)
        cpu_res = run_ort_inspection(path, ["CPUExecutionProvider"])

        cuda_res = None
        if "CUDAExecutionProvider" in avail_providers:
            cuda_res = run_ort_inspection(path, ["CUDAExecutionProvider", "CPUExecutionProvider"])

        custom_meta = (
            cpu_res.get("model_meta", {}).get("custom_metadata_map", {})
            or pb.get("metadata_props", {})
        )

        entry = {
            "path": rel,
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "source": (sources or {}).get(rel, {}),
            "onnx_protobuf": pb,
            "audio_params": known_audio_params(rel, custom_meta),
            "ort_cpu": cpu_res,
        }
        if cuda_res is not None:
            entry["ort_cuda"] = cuda_res

        # Print concise console summary
        log(f"  SHA256: {entry['sha256']}")
        log(f"  Custom metadata: {custom_meta}")
        if cpu_res.get("session_ok"):
            log("  Inputs:")
            for i in cpu_res["inputs"]:
                log(f"    - {i['name']}: {i['type']} {i['shape']}")
            log("  Outputs:")
            for o in cpu_res["outputs"]:
                log(f"    - {o['name']}: {o['type']} {o['shape']}")
            log(f"  CPU inference ok={cpu_res.get('inference_ok')}, outputs={cpu_res.get('dummy_output_shapes')}")
        else:
            log(f"  CPU session FAILED: {cpu_res.get('error')}")

        models_out.append(entry)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "onnxruntime_version": ort.__version__,
        "available_providers": avail_providers,
        "models": models_out,
    }


def run_secondary_env_probe() -> dict[str, Any] | None:
    """If primary python lacks `onnx` or `CUDAExecutionProvider`, probe CUDA_ENV_PYTHON cleanly."""
    if not CUDA_ENV_PYTHON.exists():
        return None
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    cmd = [str(CUDA_ENV_PYTHON), str(Path(__file__).resolve()), "--inspect-only", "--json-out", "-"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"ok": False, "returncode": proc.returncode, "stderr": proc.stderr[-2000:]}
    try:
        return {"ok": True, "report": json.loads(proc.stdout)}
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"JSONDecodeError: {e}", "stdout_tail": proc.stdout[-1000:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-only", action="store_true", help="Skip downloading")
    parser.add_argument("--force-download", action="store_true", help="Re-download existing files")
    parser.add_argument("--no-secondary-probe", action="store_true", help="Skip secondary CUDA/onnx env probe")
    parser.add_argument("--json-out", default=str(MANIFEST_PATH), help="Output JSON path or '-' for stdout")
    args = parser.parse_args()

    sources: dict[str, dict[str, Any]] = {}
    if not args.inspect_only:
        sources = download_models(force=args.force_download)

    report = inspect_all(sources=sources)

    if not args.inspect_only and not args.no_secondary_probe:
        need_onnx = not any(m.get("onnx_protobuf", {}).get("available") for m in report["models"])
        need_cuda = "CUDAExecutionProvider" not in report["available_providers"]
        if (need_onnx or need_cuda) and Path(sys.executable).resolve() != CUDA_ENV_PYTHON.resolve():
            log(f"\n=== Running secondary probe in {CUDA_ENV_PYTHON} (onnx + CUDA check) ===")
            sec = run_secondary_env_probe()
            if sec and sec.get("ok"):
                sec_rep = sec["report"]
                report["secondary_cuda_onnx_env"] = {
                    "python_executable": sec_rep["python_executable"],
                    "python_version": sec_rep["python_version"],
                    "onnxruntime_version": sec_rep["onnxruntime_version"],
                    "available_providers": sec_rep["available_providers"],
                }
                by_path = {m["path"]: m for m in sec_rep["models"]}
                for m in report["models"]:
                    sm = by_path.get(m["path"])
                    if not sm:
                        continue
                    if not m["onnx_protobuf"].get("available") and sm["onnx_protobuf"].get("available"):
                        m["onnx_protobuf"] = sm["onnx_protobuf"]
                    if "ort_cuda" in sm:
                        m["ort_cuda"] = sm["ort_cuda"]
                    m["ort_secondary_cpu"] = sm["ort_cpu"]
            elif sec:
                report["secondary_cuda_onnx_env"] = sec

    if args.json_out == "-":
        print(json.dumps(report, indent=2))
    else:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"\n[saved] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
