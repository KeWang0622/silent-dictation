#!/usr/bin/env bash
# Install Silent Dictation: a Python environment plus checksum-verified model files.
# Usage: ./setup.sh [--with-llm] [--no-flare] [--accept-model-license]
#   --with-llm               also install the on-device language model for AI clean-up (+2 GB, recommended)
#   --no-flare               skip the FLARE lip reader (a 4.8 GB download, 0.8 GB on disk; ~30% fewer mistakes)
#   --accept-model-license   do not ask before downloading the model weights
set -euo pipefail
cd "$(dirname "$0")"

WITH_LLM=0
WITH_FLARE=1
ACCEPT=0
for arg in "$@"; do
  case "$arg" in
    --with-llm) WITH_LLM=1 ;;
    --no-llm) WITH_LLM=0 ;;
    --no-flare) WITH_FLARE=0 ;;
    --accept-model-license) ACCEPT=1 ;;
    -h|--help) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || { echo "Silent Dictation currently supports macOS only." >&2; exit 1; }
[[ "$(uname -m)" == "arm64" ]] || echo "Note: Silent Dictation is tested on Apple Silicon; Intel Macs will be slow." >&2
command -v uv >/dev/null || { echo "Install uv first: brew install uv" >&2; exit 1; }

cat <<'EOF'
Model terms
  * The lip-reading weights are released by their authors for non-commercial use: FLARE (Zhang et al.,
    CC BY-NC 4.0) and Auto-AVSR (Ma et al., trained on LRS3, non-commercial research use). They are
    downloaded from third-party hosts, they are NOT covered by this project's Apache-2.0 licence, and a
    personal model fine-tuned from them inherits their terms.
  * MediaPipe Face Landmarker and the optional Qwen3-4B language model are Apache-2.0.
EOF
if [[ $ACCEPT -eq 0 ]]; then
  read -r -p "Download the models and accept their terms? [y/N] " answer
  [[ "$answer" =~ ^[Yy] ]] || { echo "Aborted."; exit 1; }
fi

uv venv --allow-existing --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
if [[ $WITH_LLM -eq 1 ]]; then
  uv pip install --python .venv/bin/python -r requirements-llm.txt
fi

fetch() {  # url destination sha256
  local url="$1" dest="$2" sum="$3"
  if [[ -f "$dest" && "$(shasum -a 256 "$dest" | cut -d' ' -f1)" == "$sum" ]]; then
    echo "ok     $dest"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  echo "fetch  $dest"
  curl -fL --retry 3 -C - --progress-bar -o "$dest.part" "$url"  # resumes an interrupted download
  if [[ "$(shasum -a 256 "$dest.part" | cut -d' ' -f1)" != "$sum" ]]; then
    rm -f "$dest.part"
    echo "checksum mismatch for $dest" >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
}

HF=https://huggingface.co
VSR_REV=c0b413f56b164565865a63f00774ee55a9738cc5
LM_REV=f4e39b6d0904b2a900c53a2fff0377796c797bdc
AVSR_REV=182b62837773ab01052d4ac21ef1d2203ea7d267
fetch "$HF/Amanvir/LRS3_V_WER19.1/resolve/$VSR_REV/model.json" models/vsr_lrs3/model.json \
  e062e8c58579f54cb51087e639e674781189809905db4162efe0fe1325356d4f
fetch "$HF/Amanvir/LRS3_V_WER19.1/resolve/$VSR_REV/model.pth" models/vsr_lrs3/model.pth \
  e740cef369abeabd0ba2c18e37a0661342e1d94d432d6caa77755a11821d8fe3
fetch "$HF/Amanvir/lm_en_subword/resolve/$LM_REV/model.json" models/lm_lrs3/model.json \
  6e2d2004e4066af3e5a3d74c53b0d648fd53904da07178671d1cbe28a206d548
fetch "$HF/Amanvir/lm_en_subword/resolve/$LM_REV/model.pth" models/lm_lrs3/model.pth \
  c75aa39020dec98f432c8689b145d3f4cc407d4daa90a0c64202386c34f83c18
fetch "https://raw.githubusercontent.com/mpc001/auto_avsr/$AVSR_REV/spm/unigram/unigram5000.model" \
  models/unigram5000.model 2c1d648ccf5fdce6612ecbea2ffbd1cab5aabc90458781186175c3911c4bdb1e
fetch "https://raw.githubusercontent.com/mpc001/auto_avsr/$AVSR_REV/spm/unigram/unigram5000_units.txt" \
  models/unigram5000_units.txt 29001d967ad290e28f6f1d7d47e07e314354bf67d3b1f494e936fa2fed9a9122
fetch "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
  models/mediapipe/face_landmarker.task 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

if [[ $WITH_FLARE -eq 1 && ! -f models/flare_large/model.pt ]]; then
  FLARE_REV=6187ad650dc1786a8efc8b6bfbee37bbf62c4b2e
  echo "FLARE lip reader: downloading 4.8 GB once, then converting it to a 0.8 GB file"
  fetch "$HF/Anony94/FLARE-checkpoints/resolve/$FLARE_REV/exp_22_2_large/finetuned/checkpoints/checkpoint_last.pt" \
    models/flare_large/checkpoint_last.pt ceaaf364ab47cda05dddc05dc109752cede57c989d7f9481b765e29e227cbf25
  PYTHONPATH="$PWD" .venv/bin/python -m silentdict.flare.convert models/flare_large/checkpoint_last.pt \
    models/flare_large/model.pt
  rm -f models/flare_large/checkpoint_last.pt
fi

if [[ $WITH_LLM -eq 1 ]]; then
  echo "fetch  models/llm/Qwen3-4B-Instruct-2507-4bit (about 2 GB)"
  .venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("mlx-community/Qwen3-4B-Instruct-2507-4bit",
                  revision="50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
                  local_dir="models/llm/Qwen3-4B-Instruct-2507-4bit")
EOF
fi

chmod +x bin/silentdict bin/build-app
echo
echo "Done. Next:"
echo "  bin/build-app --install     # builds Silent Dictation.app into /Applications and opens it"
