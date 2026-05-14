# IR-DKFL-UAV-Detection

**Infrared Domain Knowledge-Enhanced Feature Learning for UAV Detection**  
*Published in PLOS ONE* · [Paper Link]

---

## Overview

This repository contains the core implementation of **IR-DKFL**, an infrared domain knowledge-enhanced UAV detection framework built upon [YOLOE](https://github.com/THU-MIG/yoloe). Our contributions include:

- **`infrared_knowledge.py`** — IR domain knowledge fusion module (IR-DKFL)
- **`loss.py`** — STAR Loss implementation
- **`main.py`** — Full fine-tuning training script
- **`dataset.yaml`** — Dataset configuration for the UAV infrared dataset

## Requirements

```bash
# 1. Clone and install YOLOE first (required base framework)
git clone https://github.com/THU-MIG/yoloe.git
cd yoloe
pip install -r requirements.txt
pip install -e .

# 2. Install additional dependencies
pip install torch>=2.0.0 torchvision
```

## Usage

```bash
# Place our files into your YOLOE directory, then run:
python main.py --data dataset.yaml --epochs 100
```

## File Structure
IR-DKFL-UAV-Detection/
├── README.md
├── LICENSE
├── infrared_knowledge.py   # IR-DKFL core module
├── loss.py                 # STAR Loss
├── main.py    # Training script
└── dataset.yaml            # Dataset config

## Acknowledgements

This work is built upon the following open-source projects:

- [YOLOE](https://github.com/THU-MIG/yoloe) (Wang et al., ICCV 2025) — base detection framework
- [Ultralytics](https://github.com/ultralytics/ultralytics) — underlying YOLO infrastructure (AGPL-3.0)
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World) — referenced architecture components

## Citation

If you use this code, please cite our paper:

```bibtex
@article{yourname2025irdkfl,
  title   = {Infrared Domain Knowledge-Enhanced Feature Learning for UAV Detection},
  author  = {Your Name and Co-authors},
  journal = {PLOS ONE},
  year    = {2025},
  doi     = {10.xxxx/xxxxx}
}
```

Please also cite YOLOE:

```bibtex
@misc{wang2025yoloerealtimeseeing,
  title         = {YOLOE: Real-Time Seeing Anything},
  author        = {Ao Wang and Lihao Liu and Hui Chen and Zijia Lin and Jungong Han and Guiguang Ding},
  year          = {2025},
  eprint        = {2503.07465},
  archivePrefix = {arXiv}
}
```

## License

This project is licensed under the **AGPL-3.0 License** in accordance with the upstream
[Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) dependency.  
See [LICENSE](./LICENSE) for details.
