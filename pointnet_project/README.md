# PointNet ModelNet-10 Project

This project implements:
- Custom dataset preprocessing for point clouds
- Vanilla PointNet classifier
- Training + validation logging and curve plots
- Permutation invariance test
- Critical point extraction + visualization
- Sparse critical-point robustness experiment

## Project Structure

- `dataset.py`
- `model.py`
- `train.py`
- `permutation_test.py`
- `critical_analysis.py`
- `utils/common.py`
- `outputs/` (created after training)

## Dataset Layout

Expected input layout:

```
ModelNet-10/
  train/
    class_a/*.ply
    class_b/*.ply
  test/
    class_a/*.ply
    class_b/*.ply
```

Default path used by scripts:

`../ModelNet-10-20260404T105413Z-3-001/ModelNet-10`

## Run

Train:

```powershell
c:/Users/hexlive63/OneDrive/Desktop/opencv3/.venv/Scripts/python.exe train.py --epochs 50 --batch-size 32 --num-points 1024
```

Permutation invariance test:

```powershell
c:/Users/hexlive63/OneDrive/Desktop/opencv3/.venv/Scripts/python.exe permutation_test.py
```

Critical point extraction + robustness:

```powershell
c:/Users/hexlive63/OneDrive/Desktop/opencv3/.venv/Scripts/python.exe critical_analysis.py --num-visuals 5
```

## Quick smoke run

```powershell
c:/Users/hexlive63/OneDrive/Desktop/opencv3/.venv/Scripts/python.exe train.py --epochs 1 --max-train-samples 256 --max-test-samples 128
c:/Users/hexlive63/OneDrive/Desktop/opencv3/.venv/Scripts/python.exe permutation_test.py --max-test-samples 128
c:/Users/hexlive63/OneDrive/Desktop/opencv3/.venv/Scripts/python.exe critical_analysis.py --max-test-samples 128 --num-visuals 5
```
