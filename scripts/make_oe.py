"""Turn the engine assets of a ggml-org/llama.cpp release into linked .oe bundles.

Each asset named llama-TAG-bin-PLATFORM.EXT becomes llama-TAG-bin-PLATFORM.oe, a zip
holding only an openengine.json whose "download" field points at the original asset.
Dots in the asset name become hyphens, because bundle names may only contain
lowercase letters, numbers, and hyphens.

The manifest is filled from three sources:
- the asset name (OS, architecture, backend, toolkit version),
- the archive itself (minimum macOS, Linux kernel, glibc, and Android API levels read
  from the binaries, Hexagon NPU versions from the bundled libraries),
- how upstream builds each asset (CPU baselines, CUDA architectures from
  ggml/src/ggml-cuda/CMakeLists.txt, ROCm GPU targets from .github/workflows/release.yml).

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
import struct
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
GLIBC_RE = re.compile(rb"GLIBC_(2\.[0-9]+(?:\.[0-9]+)?)\0")
HTP_RE = re.compile(r"libggml-htp-(v[0-9]+)\.so$")
CPU_VARIANT_RE = re.compile(r"(?:^|/)(?:lib)?ggml-cpu-([a-z0-9_.]+?)\.(?:so|dll|dylib)$")
ZIP_DATE = (2000, 1, 1, 0, 0, 0)

OSES = {"win": "windows", "ubuntu": "linux", "linux": "linux", "macos": "macos",
        "android": "android"}
OS_LABELS = {"win": "Windows", "ubuntu": "Ubuntu", "linux": "Linux", "macos": "macOS",
             "android": "Android"}
SPEC_OSES = {"windows", "linux", "macos"}
ARCHES = {"x64", "arm64", "s390x"}
SPEC_ARCHES = {"x64", "arm64"}
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
ANDROID_VERSIONS = {28: "9", 29: "10", 30: "11", 31: "12", 32: "12L", 33: "13", 34: "14",
                    35: "15", 36: "16"}
SERVER = "llama-server"
API = ["openai-chat", "openai-completions", "openai-responses", "openai-embeddings",
       "anthropic-messages"]
MODELS = ("Any GGUF file. Pass a local path, or change -m {model} to -hf {model} to download "
          "from Hugging Face (for example ggml-org/gemma-3-4b-it-GGUF:Q4_K_M).")


def github(url, raw=False):
    headers = {"Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
               "User-Agent": "make-oe"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return r.read().decode("utf-8") if raw else json.load(r)


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


def rocm_targets(repo, tag):
    """Map (os token, ROCm major.minor) to the AMDGPU targets upstream builds for that tag."""
    try:
        workflow = github(f"https://api.github.com/repos/{repo}/contents/.github/workflows/"
                          f"release.yml?ref={tag}", raw=True)
    except OSError as e:
        print(f"warning: could not read upstream release.yml: {e}", file=sys.stderr)
        return {}
    targets = {}
    for job in re.split(r"\n  (?=[A-Za-z0-9_-]+:\n)", workflow):
        os_token = ("win" if "bin-win-rocm-" in job else
                    "ubuntu" if "bin-ubuntu-rocm-" in job else None)
        if os_token is None:
            continue
        for version, gpus in re.findall(r'ROCM_VERSION:\s*"([0-9.]+)"\s*\n\s*gpu_targets:\s*"([^"]+)"',
                                        job):
            targets[(os_token, ".".join(version.split(".")[:2]))] = gpus.split(";")
    if not targets:
        print("warning: no ROCm gpu_targets found in upstream release.yml", file=sys.stderr)
    return targets


def cuda_targets(version):
    """GPU targets of llama.cpp's default CMAKE_CUDA_ARCHITECTURES for a CUDA toolkit version."""
    v = tuple(int(n) for n in version.split("."))
    targets = ["compute_50", "compute_61", "compute_70"] if v < (13,) else []
    targets += ["compute_75", "compute_80", "sm_86"]
    if v >= (11, 8):
        targets += ["sm_89", "compute_90"]
    if v >= (12, 8):
        targets.append("sm_120")
    if v >= (12, 9):
        targets.append("sm_121")
    return targets


def parse_platform(rest):
    """Map the part after 'bin-' (such as 'win-cuda-12.4-x64') to spec values and labels."""
    tokens = rest.split("-")
    os_token = tokens[0]
    os_label = OS_LABELS.get(os_token, os_token.capitalize())
    arch = backend = version = variant = None
    for token in tokens[1:]:
        if token in ARCHES:
            arch = token
        elif token in BACKENDS:
            backend = BACKENDS[token]
        elif VERSION_TOKEN_RE.match(token):
            if backend:
                version = token
            else:
                os_label += f" {token}"
        elif token.startswith("fp"):
            variant = token
    os_name = OSES.get(os_token)
    if backend is None:
        backend = "metal" if (os_name, arch) == ("macos", "arm64") else "cpu"
    suffix = "".join(f" {x}" for x in (version, variant and variant.upper()) if x)
    return {
        "os_token": os_token,
        "os": os_name,
        "arch": arch,
        "backend": backend,
        "version": version,
        "variant": variant,
        "target": f"{os_label} IBM Z (s390x)" if arch == "s390x" else f"{os_label} {arch}",
        "hardware": BACKEND_LABELS[backend].format(suffix=suffix),
    }


def cpu_baseline(p, variants):
    """Return (metadata value, sentence) for the oldest CPU the build runs on.

    variants are the ggml-cpu-* libraries in the archive. Builds that ship them pick the
    best one at runtime; builds that do not use upstream's fixed compiler flags.
    """
    os_name, arch, backend = p["os"], p["arch"], p["backend"]
    if backend == "ibm-zdnn":
        return "z17", "Needs an IBM z17 (built with -march=arch15) with the zDNN accelerator."
    if arch == "s390x":
        return "z15", "Needs an IBM z15 or newer; z16 and z17 code is used when present."
    if os_name == "macos" and arch == "x64":
        return "native", ("Built with -march=native on GitHub's Intel macOS runner, so older Intel "
                          "Macs may lack the instructions it uses.")
    if os_name == "macos":
        return "apple-m1", "Needs Apple Silicon (M1 or newer)."
    if arch == "x64" and variants:
        return "x86-64", "Runs on any x86-64 CPU; AVX2, AVX-512, and AMX code is used when present."
    if arch == "x64":
        return "x86-64-v3", ("Needs an x86-64 CPU with AVX2, FMA, F16C, and BMI2 (Intel Haswell, "
                             "AMD Zen, or newer).")
    if backend == "qualcomm-snapdragon" and os_name == "android":
        return "armv8.7-a", "Needs an ARMv8.7-A CPU with FP16, dot product, and int8 matrix multiply."
    if backend == "qualcomm-snapdragon":
        return "armv8.2-a", "Needs an ARMv8.2-A CPU with FP16 and dot product."
    if os_name == "windows":
        return "armv8.7-a", "Needs an ARMv8.7-A CPU, such as Snapdragon X."
    if variants:
        return "armv8-a", "Runs on any ARMv8-A CPU; dot product, SVE, int8 matrix multiply, and SME code is used when present."
    return "armv8-a", "Needs an ARMv8-A CPU."


def elf_notes(data):
    """Return the minimum Linux kernel (GNU ABI tag) and Android API level of an ELF file."""
    out = {}
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return out
    end = "<" if data[5] == 1 else ">"
    phoff, = struct.unpack_from(end + "Q", data, 0x20)
    phentsize, phnum = struct.unpack_from(end + "HH", data, 0x36)
    for i in range(phnum):
        p_type, _, p_offset, _, _, p_filesz = struct.unpack_from(end + "IIQQQQ", data,
                                                                 phoff + i * phentsize)
        if p_type != 4:
            continue
        pos, stop = p_offset, p_offset + p_filesz
        while pos + 12 <= stop:
            namesz, descsz, ntype = struct.unpack_from(end + "III", data, pos)
            name = data[pos + 12:pos + 12 + namesz].rstrip(b"\0")
            desc = pos + 12 + ((namesz + 3) & ~3)
            pos = desc + ((descsz + 3) & ~3)
            if ntype != 1:
                continue
            if name == b"GNU" and descsz >= 16:
                out["kernel"] = struct.unpack_from(end + "IIII", data, desc)[1:]
            elif name == b"Android" and descsz >= 4:
                out["android_api"] = struct.unpack_from(end + "I", data, desc)[0]
    return out


def macho_minos(data):
    """Return the minimum macOS version of a 64-bit Mach-O file, or None."""
    if data[:4] != b"\xcf\xfa\xed\xfe":
        return None
    ncmds, = struct.unpack_from("<I", data, 16)
    pos = 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, pos)
        if cmd in (0x32, 0x24):  # LC_BUILD_VERSION, LC_VERSION_MIN_MACOSX
            v, = struct.unpack_from("<I", data, pos + (12 if cmd == 0x32 else 8))
            return (v >> 16, (v >> 8) & 0xFF, v & 0xFF)
        pos += size
    return None


def version_text(parts):
    parts = list(parts)
    while len(parts) > 2 and parts[-1] == 0:
        parts.pop()
    return ".".join(str(n) for n in parts)


def strip_wrapper(files):
    """Return paths as a runner sees them in bin/: without one shared top-level folder."""
    files = [re.sub(r"^(\./)+", "", f.replace("\\", "/")) for f in files]
    tops = {f.split("/", 1)[0] for f in files}
    if len(tops) == 1 and all("/" in f for f in files):
        return [f.split("/", 1)[1] for f in files]
    return files


def inspect(archive, name):
    """Read the facts a manifest needs from the archive's file list and binaries."""
    facts = {"glibc": set()}
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            originals = [i.filename for i in zf.infolist() if not i.is_dir()]
            readers = {}
    else:
        tf = tarfile.open(archive, "r:*")
        members = [m for m in tf.getmembers() if m.isfile()]
        originals = [m.name for m in members]
        readers = dict(zip(originals, members))
    files = strip_wrapper(originals)
    servers = [(f, o) for f, o in zip(files, originals)
               if f.rsplit("/", 1)[-1] in (SERVER, SERVER + ".exe")]
    facts["program"], server = min(servers, key=lambda s: len(s[0])) if servers else (None, None)
    facts["htp"] = sorted({m.group(1) for f in files if (m := HTP_RE.search(f))},
                          key=lambda v: int(v[1:]))
    facts["cpu_variants"] = sorted({m.group(1) for f in files if (m := CPU_VARIANT_RE.search(f))})
    if readers:
        try:
            for original, member in readers.items():
                with tf.extractfile(member) as f:
                    head = f.read(4)
                    if head not in (b"\x7fELF", b"\xcf\xfa\xed\xfe"):
                        continue
                    data = head + f.read()
                facts["glibc"].update(GLIBC_RE.findall(data))
                if original == server:
                    facts.update(elf_notes(data))
                    if (minos := macho_minos(data)):
                        facts["macos"] = minos
        finally:
            tf.close()
    facts["glibc"] = max((tuple(int(n) for n in g.decode().split(".")) for g in facts["glibc"]),
                         default=None)
    return facts


def find_cudart(asset_name, tag, names):
    """Return the upstream cudart archive that matches a CUDA asset, if any."""
    for candidate in (f"cudart-{asset_name}", f"cudart-{asset_name.replace(f'-{tag}', '', 1)}"):
        if candidate in names:
            return candidate
    return None


def describe(tag, p, facts, cudart, spec_platform):
    description = (
        f"llama.cpp server {tag} for {p['target']} on {p['hardware']}. Runs almost any GGUF "
        "model (Llama, Qwen, Gemma, Mistral, GLM, DeepSeek, gpt-oss) for local chat, "
        "embeddings, and reranking."
    )
    os_name, backend, version = p["os"], p["backend"], p["version"]
    notes = []
    if os_name == "windows":
        notes.append("Windows 10 or later.")
    elif facts.get("macos"):
        notes.append(f"macOS {version_text(facts['macos'])} or later.")
    elif facts.get("android_api"):
        api = facts["android_api"]
        release = ANDROID_VERSIONS.get(api)
        notes.append(f"Android {release} (API {api}) or later." if release else
                     f"Android API {api} or later.")
    if facts.get("glibc"):
        notes.append(f"Needs glibc {version_text(facts['glibc'])} or later.")
    notes.append(cpu_baseline(p, facts["cpu_variants"])[1])
    if backend == "cuda":
        runtime = f" ({cudart})" if cudart else ""
        notes.append(f"Needs an NVIDIA driver for CUDA {version} and the CUDA {version} runtime, "
                     f"which upstream ships separately{runtime}.")
    elif backend == "rocm":
        notes.append(f"Needs an AMD GPU listed in requires.gpu and a driver for ROCm {version}.")
    elif backend == "vulkan":
        notes.append("Needs a Vulkan 1.2 driver.")
    elif backend == "sycl":
        if os_name == "windows":
            notes.append("Bundles the oneAPI runtime; needs Intel GPU drivers with Level Zero.")
        else:
            notes.append("Needs the Intel oneAPI runtime and Intel GPU drivers with Level Zero.")
    elif backend == "intel-openvino":
        notes.append(f"Bundles OpenVINO {version}; Intel GPU and NPU drivers enable those devices.")
    elif backend == "qualcomm-adreno":
        notes.append("Needs a Snapdragon X Adreno GPU driver with OpenCL.")
    elif backend == "qualcomm-snapdragon" and facts["htp"]:
        notes.append(f"Uses Hexagon NPU {', '.join(facts['htp'])} and the Adreno GPU through OpenCL.")
    if not spec_platform:
        notes.append("Open Engine has no platform value for this target yet.")
    compatibility = " ".join(notes)
    if len(compatibility) > 500:
        raise SystemExit(f"compatibility is over 500 characters: {compatibility}")
    return description, compatibility


def download(url, dest):
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def build(asset, ctx):
    name_in, tag = asset["name"], ctx["tag"]
    stem = ARCHIVE_RE.sub("", name_in)
    name = stem.lower().replace(".", "-")
    if not NAME_RE.match(name) or len(name) > 64:
        return f"skip {name_in}: '{name}' is not a valid bundle name"

    archive = Path(ctx["tmp"]) / name_in
    try:
        sha256 = download(asset["browser_download_url"], archive)
        expected = (asset.get("digest") or "").removeprefix("sha256:")
        if expected and expected != sha256:
            raise SystemExit(f"{name_in}: sha256 {sha256} does not match GitHub digest {expected}")
        facts = inspect(archive, name_in)
    finally:
        archive.unlink(missing_ok=True)
    if facts["program"] is None:
        return f"skip {name_in}: no {SERVER} in the archive"

    p = parse_platform(stem.split("-bin-", 1)[1])
    os_name, arch, backend, version = p["os"], p["arch"], p["backend"], p["version"]
    spec_platform = os_name in SPEC_OSES and arch in SPEC_ARCHES
    cudart = find_cudart(name_in, tag, ctx["names"]) if backend == "cuda" else None
    description, compatibility = describe(tag, p, facts, cudart, spec_platform)

    requires = {}
    if spec_platform:
        if os_name == "windows":
            requires["os"] = {"windows": "10.0"}
        elif os_name == "macos" and facts.get("macos"):
            requires["os"] = {"macos": version_text(facts["macos"])}
        elif os_name == "linux" and facts.get("kernel"):
            requires["os"] = {"linux": version_text(facts["kernel"])}
    if backend == "cuda" and version:
        requires["gpu"] = cuda_targets(version)
    elif backend == "rocm" and version:
        gpus = ctx["rocm"].get((p["os_token"], version))
        if gpus:
            requires["gpu"] = gpus
    elif backend == "qualcomm-snapdragon" and facts["htp"]:
        requires["gpu"] = [f"qualcomm-hexagon-{v}" for v in facts["htp"]]

    metadata = {
        "homepage": ctx["homepage"],
        "release": ctx["release_url"],
        "asset": name_in,
        "os": os_name or p["os_token"],
        "arch": arch or "unknown",
        "backend": backend,
        "cpu": cpu_baseline(p, facts["cpu_variants"])[0],
    }
    if version:
        metadata["backend-version"] = version
    if p["variant"]:
        metadata["precision"] = p["variant"]
    if facts.get("glibc"):
        metadata["glibc"] = version_text(facts["glibc"])
    if facts.get("android_api"):
        metadata["android-api"] = str(facts["android_api"])
    if cudart:
        metadata["cuda-runtime"] = f"{ctx['download_base']}/{cudart}"

    manifest = {"name": name, "description": description, "version": tag}
    if spec_platform:
        manifest["platform"] = [f"{os_name}-{arch}-{backend}"]
    if requires:
        manifest["requires"] = requires
    manifest.update({
        "download": {"url": asset["browser_download_url"], "sha256": sha256},
        "run": [f"bin/{facts['program']}", "-m", "{model}", "--host", "{host}", "--port", "{port}"],
        "api": API,
        "health": "/health",
        "formats": ["gguf"],
        "models": MODELS,
        "compatibility": compatibility,
        "license": "MIT",
        "metadata": metadata,
    })

    info = zipfile.ZipInfo("openengine.json", ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(Path(ctx["out"]) / f"{name}.oe", "w") as zf:
        zf.writestr(info, json.dumps(manifest, indent=2) + "\n")
    return f"made {name}.oe -> {name_in} (run {facts['program']})"


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
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(args.jobs) as pool:
        ctx = {
            "tag": tag,
            "out": out,
            "tmp": tmp,
            "names": {a["name"] for a in release["assets"]},
            "rocm": rocm_targets(args.repo, tag),
            "homepage": f"https://github.com/{args.repo}",
            "release_url": release["html_url"],
            "download_base": f"https://github.com/{args.repo}/releases/download/{tag}",
        }
        for line in pool.map(lambda a: build(a, ctx), assets):
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
