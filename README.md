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
