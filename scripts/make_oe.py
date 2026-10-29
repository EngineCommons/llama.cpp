"""Turn the engine assets of a ggml-org/llama.cpp release into linked .oe bundles.

Each asset named llama-TAG-bin-PLATFORM.EXT becomes llama-TAG-bin-PLATFORM.oe, a zip
holding only an openengine.json whose "download" field points at the original asset.
Dots in the asset name become hyphens, because bundle names may only contain
lowercase letters, numbers, and hyphens.

Usage:
    python make_oe.py [--repo ggml-org/llama.cpp] [--tag bNNNN] [--out dist]

Requires Python 3.9+. No third-party packages. Set GITHUB_TOKEN or GH_TOKEN to avoid
API rate limits.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARCHIVE_RE = re.compile(r"\.(zip|tar\.gz|tgz|tar\.xz)$")
BUILD_TAG_RE = re.compile(r"^b[0-9]+$")
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
VERSION_TOKEN_RE = re.compile(r"^[0-9][0-9.]*$")
ZIP_DATE = (2000, 1, 1, 0, 0, 0)

OSES = {"win": "windows", "ubuntu": "linux", "linux": "linux", "macos": "macos"}
OS_LABELS = {"win": "Windows", "ubuntu": "Ubuntu", "linux": "Linux", "macos": "macOS"}
ARCHES = {"x64", "arm64"}
BACKENDS = {
    "cpu": "cpu",
    "cuda": "cuda",
    "rocm": "rocm",
    "vulkan": "vulkan",
    "sycl": "sycl",
    "metal": "metal",
    "openvino": "intel-openvino",
    "adreno": "qualcomm-adreno",
    "snapdragon": "qualcomm-snapdragon",
    "zdnn": "ibm-zdnn",
}
BACKEND_LABELS = {
    "cpu": "the CPU",
    "cuda": "NVIDIA GPUs (CUDA{suffix})",
    "rocm": "AMD GPUs (ROCm{suffix})",
    "vulkan": "Vulkan GPUs from AMD, Intel, and NVIDIA",
    "sycl": "Intel GPUs (SYCL{suffix})",
    "metal": "Apple Silicon GPUs (Metal)",
    "intel-openvino": "Intel CPUs, GPUs, and NPUs (OpenVINO{suffix})",
    "qualcomm-adreno": "Qualcomm Adreno GPUs (OpenCL)",
    "qualcomm-snapdragon": "Qualcomm Snapdragon (Hexagon NPU and Adreno GPU)",
    "ibm-zdnn": "the IBM Z Deep Neural Network accelerator (zDNN)",
}
SERVER = "llama-server"
API = ["openai-chat", "openai-completions", "openai-responses", "openai-embeddings",
       "anthropic-messages"]
MODELS = ("Any GGUF file. Pass a local path, or change -m {model} to -hf {model} to download "
          "from Hugging Face (for example ggml-org/gemma-3-4b-it-GGUF:Q4_K_M).")


def github(url):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "make-oe"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return json.load(r)


def find_release(repo, tag):
    """Return the release for tag, or the newest published bNNNN release with assets.

    llama.cpp publishes its builds as prereleases, so /releases/latest does not find them.
    """
    if tag:
        return github(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
    for release in github(f"https://api.github.com/repos/{repo}/releases?per_page=30"):
        if not release["draft"] and BUILD_TAG_RE.match(release["tag_name"]) and release["assets"]:
            return release
    raise SystemExit(f"no bNNNN release with assets found in {repo}")


def parse_platform(rest):
    """Map the part after 'bin-' (such as 'win-cuda-12.4-x64') to spec values and labels.

    "os" or "arch" is None when the spec has no value for it (android, s390x).
    """
    tokens = rest.split("-")
    os_label = OS_LABELS.get(tokens[0], tokens[0].capitalize())
    arch = backend = version = variant = None
    for token in tokens[1:]:
        if token in ARCHES or token == "s390x":
            arch = token
        elif token in BACKENDS:
            backend = BACKENDS[token]
        elif VERSION_TOKEN_RE.match(token):
            if backend:
                version = token
            else:
                os_label += f" {token}"
        elif token.startswith("fp"):
            variant = token.upper()
    os_name = OSES.get(tokens[0])
    if backend is None:
        backend = "metal" if (os_name, arch) == ("macos", "arm64") else "cpu"
    suffix = "".join(f" {x}" for x in (version, variant) if x)
    return {
        "os": os_name,
        "arch": arch if arch in ARCHES else None,
        "backend": backend,
        "version": version,
        "target": f"{os_label} IBM Z (s390x)" if arch == "s390x" else f"{os_label} {arch}",
        "hardware": BACKEND_LABELS[backend].format(suffix=suffix),
    }


def describe(tag, p, unsupported):
    description = (
        f"llama.cpp server {tag} for {p['target']} on {p['hardware']}. Runs almost any GGUF "
        "model (Llama, Qwen, Gemma, Mistral, GLM, DeepSeek, gpt-oss) for local chat, "
        "embeddings, and reranking."
    )
    backend, notes = p["backend"], []
    if unsupported:
        notes.append(f"Built for {p['target']}, which the Open Engine platform field cannot "
                     "express yet, so this bundle has no platform field.")
    if backend == "cuda":
        notes.append(f"Requires an NVIDIA driver and the CUDA {p['version']} runtime. Upstream "
                     "ships the runtime separately as the matching cudart-llama archive.")
    elif backend == "rocm":
        notes.append("Requires an AMD GPU supported by this ROCm build.")
    elif backend == "vulkan":
        notes.append("Requires a Vulkan 1.2 driver.")
    elif backend == "sycl":
        notes.append("Requires the Intel oneAPI runtime and GPU drivers.")
    elif backend == "intel-openvino":
        notes.append("Requires the OpenVINO runtime.")
    if p["os"] == "windows":
        notes.append("Windows 10 or later.")
    return description, " ".join(notes) or None


def download(url, dest):
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def list_files(archive, name):
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            return [i.filename for i in zf.infolist() if not i.is_dir()]
    with tarfile.open(archive, "r:*") as tf:
        return [m.name for m in tf.getmembers() if m.isfile()]


def server_path(files):
    """Return the llama-server path as a runner sees it inside bin/ after extraction."""
    files = [re.sub(r"^(\./)+", "", f.replace("\\", "/")) for f in files]
    tops = {f.split("/", 1)[0] for f in files}
    if len(tops) == 1 and all("/" in f for f in files):
        files = [f.split("/", 1)[1] for f in files]
    matches = [f for f in files if f.rsplit("/", 1)[-1] in (SERVER, SERVER + ".exe")]
    return min(matches, key=len) if matches else None


def build(asset, tag, out_dir, tmp_dir, homepage):
    name_in = asset["name"]
    stem = ARCHIVE_RE.sub("", name_in)
    name = stem.lower().replace(".", "-")
    if not NAME_RE.match(name) or len(name) > 64:
        return f"skip {name_in}: '{name}' is not a valid bundle name"

    archive = Path(tmp_dir) / name_in
    try:
        sha256 = download(asset["browser_download_url"], archive)
        expected = (asset.get("digest") or "").removeprefix("sha256:")
        if expected and expected != sha256:
            raise SystemExit(f"{name_in}: sha256 {sha256} does not match GitHub digest {expected}")
        program = server_path(list_files(archive, name_in))
    finally:
        archive.unlink(missing_ok=True)
    if program is None:
        return f"skip {name_in}: no {SERVER} in the archive"

    p = parse_platform(stem.split("-bin-", 1)[1])
    unsupported = p["os"] is None or p["arch"] is None
    description, compatibility = describe(tag, p, unsupported)
    manifest = {
        "name": name,
        "description": description,
        "version": tag,
    }
    if not unsupported:
        manifest["platform"] = [f"{p['os']}-{p['arch']}-{p['backend']}"]
    if p["os"] == "windows":
        manifest["requires"] = {"os": {"windows": "10.0"}}
    manifest.update({
        "download": {"url": asset["browser_download_url"], "sha256": sha256},
        "run": [f"bin/{program}", "-m", "{model}", "--host", "{host}", "--port", "{port}"],
        "api": API,
        "health": "/health",
        "formats": ["gguf"],
        "models": MODELS,
    })
    if compatibility:
        manifest["compatibility"] = compatibility
    manifest["license"] = "MIT"
    manifest["metadata"] = {"homepage": homepage, "asset": name_in}

    info = zipfile.ZipInfo("openengine.json", ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(Path(out_dir) / f"{name}.oe", "w") as zf:
        zf.writestr(info, json.dumps(manifest, indent=2) + "\n")
    return f"made {name}.oe -> {name_in} (run {program})"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", default="ggml-org/llama.cpp")
    parser.add_argument("--tag", help="release tag (default: newest bNNNN release)")
    parser.add_argument("--out", default="dist")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    release = find_release(args.repo, args.tag)
    tag = release["tag_name"]
    pattern = re.compile(rf"^llama-{re.escape(tag)}-bin-.+{ARCHIVE_RE.pattern}")
    assets = [a for a in release["assets"] if pattern.match(a["name"])]
    if not assets:
        raise SystemExit(f"{args.repo} {tag} has no llama-{tag}-bin-* assets")
    print(f"{args.repo} {tag}: {len(assets)} engine assets", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    homepage = f"https://github.com/{args.repo}"
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(args.jobs) as pool:
        for line in pool.map(lambda a: build(a, tag, out, tmp, homepage), assets):
            print(line, flush=True)

    made = sorted(p.name for p in out.glob("*.oe"))
    if not made:
        raise SystemExit("no bundles were made")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"tag={tag}\nurl={release['html_url']}\ncount={len(made)}\n")


if __name__ == "__main__":
    sys.exit(main())
