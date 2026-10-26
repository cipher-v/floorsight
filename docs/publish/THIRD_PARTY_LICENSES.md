# Third-party models, libraries and data used by Floorsight (DRAFT for owner review, 23 Oct 2026)

Licences were read from each source on 23 Oct 2026 (model licences from the model cards' `license` field via the
Hugging Face API, recorded in `eval/aws-setup-01/hf-models.json`; ONNX hashes in `eval/aws-setup-01/onnx-sha256.txt`).
No Ultralytics (AGPL-3.0) code or weights are used by the default pipeline or installed on the AWS instance.

## Models (downloaded at setup, not stored in the repository)

| Model | Used for | Source (pinned revision) | Licence |
|---|---|---|---|
| RT-DETRv2 R50 (COCO) | people and vehicle boxes, every frame | `PekingU/rtdetr_v2_r50vd` @ `2824940756` | Apache-2.0 |
| Grounding DINO tiny | carried loads ("stack of crates", ...) | `IDEA-Research/grounding-dino-tiny` @ `a2bb814dd3` | Apache-2.0 |
| SAM ViT-B / SlimSAM-77 | box-prompted forklift and load outlines | `facebook/sam-vit-base` @ `70c1a07f89` / `Zigeng/SlimSAM-uniform-77` @ `79c09c1ce6` | Apache-2.0 |
| YOLOX-L / -M / -S ONNX | YOLOX-S: second people detector for the privacy filter (blur only, `cv2.dnn`); YOLOX-L/-M: alternative people backend; floor-calibration privacy filter | github.com/Megvii-BaseDetection/YOLOX release `0.1.1rc0` | Apache-2.0 |
| YuNet 2023mar ONNX | faces (`cv2.FaceDetectorYN`) | OpenCV Zoo (see `models/THIRD_PARTY.md`) | MIT |
| OmDet-Turbo Swin-T, OWLv2 base | evaluated as load/forklift detectors (not in the final run unless noted in the report) | `omlab/omdet-turbo-swin-tiny-hf`, `google/owlv2-base-patch16-ensemble` | Apache-2.0 |

## Libraries (versions on the AWS instance)

| Library | Version | Licence |
|---|---|---|
| OpenCV (COOL build on AWS: 5.1.0-dev, KleidiCV 0.7.0; stock wheel `opencv-python-headless` 5.0.0.93 elsewhere) | 5.x | Apache-2.0 (COOL AMI itself: AWS Marketplace EULA; the AMI is not redistributed) |
| PyTorch / torchvision | 2.14.1 / 0.29.1 | BSD-3-Clause |
| Hugging Face transformers / huggingface_hub / safetensors / timm | 4.57.3 / 0.36.2 / 0.8.0 / 1.0.30 | Apache-2.0 |
| NumPy / SciPy | 2.2.6 / 1.15.3 | BSD-3-Clause |
| Pillow | 13.0.0 | MIT-CMU (HPND) |
| FastAPI / Uvicorn / Pydantic | 0.115.12 / 0.34.2 / 2.11.4 | MIT / BSD-3-Clause / MIT |
| boto3 (AWS SDK) | 1.43.114 | Apache-2.0 |
| anthropic (Claude SDK, Bedrock client) | 1.15.0 | MIT |
| Playwright (video recording only) | 1.55 | Apache-2.0 |
| FFmpeg (system package, video encode) | Ubuntu 24.04 | LGPL-2.1+/GPL (system binary, not redistributed) |

## Data

| Data | Licence | Credit |
|---|---|---|
| Six clips from "Video Dataset for Safe and Unsafe Behaviours" (Önal & Dandıl, 2024), Mendeley Data V1, doi:10.17632/xjmtb22pff.1 | CC BY 4.0 | See `docs/publish/PUBLISH-CHECKLIST.md`; changes: clips trimmed, heads and faces pixelated. Raw clips are not redistributed. |
