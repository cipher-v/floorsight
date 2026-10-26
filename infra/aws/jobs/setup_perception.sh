#!/bin/bash
# AWS job: real clips + perception environments + Apache-licensed candidate models.
# Runs detached (systemd-run) so an SSM timeout cannot cut it; log and receipts go to S3 jobs/<JOB>/.
# Needs: JOB, BUCKET. Nothing is deleted; every download is checked or hashed.
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
if [ -z "${DETACHED:-}" ]; then
  [ -e "$J" ] && { echo "job dir $J exists; refusing to reuse it"; exit 2; }
  mkdir -p "$J"; cp "$0" "$J/run.sh"
  systemd-run --unit="fs-$JOB" --setenv=DETACHED=1 --setenv=JOB="$JOB" --setenv=BUCKET="$BUCKET" --property=WorkingDirectory=/opt/floorsight bash "$J/run.sh"
  echo "started fs-$JOB"; exit 0
fi
exec > "$J/log.txt" 2>&1
set -x
R=us-east-1
UA="anisha-robotics-floorsight/1.0 (research reuse of CC-BY-4.0 dataset doi:10.17632/xjmtb22pff.1)"
D=/opt/floorsight/data/private/clips
M=/opt/floorsight/models
mkdir -p "$D" "$M/hf" "$M/onnx"
STATUS=0

# ---- 1. real clips, straight from the publisher, SHA-256 checked ---------------------------
aws s3 cp --region $R "s3://$BUCKET/private/clips-manifest-private.json" "$J/clips-manifest-private.json"
python3 - "$J/clips-manifest-private.json" "$D" "$UA" > "$J/clips-receipt.json" <<'PY'
import hashlib, json, subprocess, sys, urllib.request
man, out, ua = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
rec = {"dataset": man["dataset"], "clips": {}}
for name, c in man["clips"].items():
    path = f"{out}/{name}.mp4"
    r = {"bytesExpected": c["size"], "sha256Expected": c["sha256"]}
    try:
        req = urllib.request.Request(c["url"], headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "xb") as f:
            r["finalHost"] = resp.geturl().split("/")[2]
            n = 0
            while True:
                b = resp.read(1 << 20)
                if not b:
                    break
                n += len(b)
                if n > 64 << 20:
                    raise RuntimeError("over 64 MiB limit")
                f.write(b)
        d = hashlib.sha256(open(path, "rb").read()).hexdigest()
        r.update(bytes=n, sha256=d, match=(d == c["sha256"] and n == c["size"]))
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                            "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", path], capture_output=True, text=True)
        r["stream"] = json.loads(p.stdout)["streams"][0] if p.returncode == 0 else {"error": p.stderr[-300:]}
        if c.get("expectFrames"):
            r["framesMatchScreening"] = int(r["stream"].get("nb_read_frames", -1)) == c["expectFrames"]
    except Exception as e:  # preserve partial files; record the error
        r["error"] = f"{type(e).__name__}: {e}"
    rec["clips"][name] = r
print(json.dumps(rec, indent=1))
PY
python3 -c "import json,sys;d=json.load(open('$J/clips-receipt.json'));bad=[k for k,v in d['clips'].items() if not v.get('match') or v.get('framesMatchScreening') is False];print('CLIPS_BAD',bad);sys.exit(1 if bad else 0)" || STATUS=1
aws s3 cp --region $R --recursive "$D" "s3://$BUCKET/private/clips/" --only-show-errors

# ---- 2. perception environments (COOL cv2 via PYTHONPATH; stock wheel twin for the benchmark) --
PIPS="torch torchvision transformers==4.57.3 timm huggingface_hub safetensors scipy==1.15.3 numpy==2.2.6 pillow psutil"
python3.12 -m venv /opt/floorsight/venv-perc
/opt/floorsight/venv-perc/bin/pip install -q $PIPS > "$J/pip-perc.log" 2>&1 || { echo PIP_PERC_FAILED; STATUS=1; }
python3.12 -m venv /opt/floorsight/venv-stock
/opt/floorsight/venv-stock/bin/pip install -q $PIPS "opencv-python-headless==5.0.0.93" fastapi==0.115.12 "uvicorn[standard]==0.34.2" pydantic==2.11.4 > "$J/pip-stock.log" 2>&1 || { echo PIP_STOCK_FAILED; STATUS=1; }
cat > /opt/floorsight/perc <<'SH'
#!/bin/bash
# perception interpreter with the COOL OpenCV build
set -a; . /etc/floorsight.opencv.env; set +a
export HF_HOME=/opt/floorsight/models/hf HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
exec /opt/floorsight/venv-perc/bin/python "$@"
SH
cat > /opt/floorsight/perc-stock <<'SH'
#!/bin/bash
# same packages, stock pip OpenCV wheel (no COOL paths) - benchmark leg B only
unset PYTHONPATH LD_LIBRARY_PATH
export HF_HOME=/opt/floorsight/models/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /opt/floorsight/venv-stock/bin/python "$@"
SH
chmod 755 /opt/floorsight/perc /opt/floorsight/perc-stock
/opt/floorsight/perc -c "import cv2,torch,transformers,numpy;print('perc',cv2.__version__,cv2.__file__,'torch',torch.__version__,'tf',transformers.__version__,'np',numpy.__version__)"
/opt/floorsight/perc-stock -c "import cv2,torch;print('stock',cv2.__version__,cv2.__file__,'torch',torch.__version__)"
/opt/floorsight/venv-perc/bin/pip freeze > "$J/freeze-perc.txt"; /opt/floorsight/venv-stock/bin/pip freeze > "$J/freeze-stock.txt"

# ---- 3. candidate models (Apache-2.0), licence read from each source -------------------------
cd "$M/onnx"
for m in yolox_s yolox_m yolox_l; do
  curl -fsSL --max-time 600 -A "$UA" -o "$m.onnx" "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/$m.onnx" || { echo "DOWNLOAD_FAILED $m"; STATUS=1; }
done
curl -fsSL --max-time 60 -A "$UA" -o YOLOX-LICENSE.txt https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/LICENSE || true
cp /opt/floorsight/app/models/face_detection_yunet_2023mar.onnx /opt/floorsight/app/models/THIRD_PARTY.md . 2>/dev/null || true
sha256sum *.onnx > "$J/onnx-sha256.txt"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 /opt/floorsight/perc - "$J/hf-models.json" <<'PY'
import json, sys
from huggingface_hub import HfApi, snapshot_download
ids = ["omlab/omdet-turbo-swin-tiny-hf", "IDEA-Research/grounding-dino-tiny", "google/owlv2-base-patch16-ensemble",
       "Zigeng/SlimSAM-uniform-77", "facebook/sam-vit-base", "PekingU/rtdetr_v2_r50vd"]
api, out = HfApi(), {}
for i in ids:
    r = {}
    try:
        info = api.model_info(i)
        card = info.card_data.to_dict() if info.card_data else {}
        r["license"] = card.get("license") or [t for t in (info.tags or []) if t.startswith("license:")]
        r["sha"] = info.sha
        r["path"] = snapshot_download(i, revision=info.sha, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "LICENSE*", "README.md"])
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
    out[i] = r
    print(i, r)
json.dump(out, open(sys.argv[1], "w"), indent=1)
PY
du -sh "$M"/* | tee "$J/models-size.txt"
chown -R floorsight: /opt/floorsight/models /opt/floorsight/data /opt/floorsight/venv-perc /opt/floorsight/venv-stock

echo "SETUP_STATUS=$STATUS"
aws s3 cp --region $R --recursive "$J" "s3://$BUCKET/jobs/$JOB/" --exclude "clips-manifest-private.json" --only-show-errors
echo "SETUP_DONE $JOB status=$STATUS" > "$J/DONE"
aws s3 cp --region $R "$J/DONE" "s3://$BUCKET/jobs/$JOB/DONE" --only-show-errors
