# Q3 Report: Learning on Point Clouds

## 3.1 Losses and Metrics

### Plots
1. Training loss curve
2. Validation loss curve
3. Training accuracy curve
4. Validation accuracy curve

You can plot all in one image as save as losses.png or make multiple images if you want to show losses/accuracies separately.

### Final Classification Metrics
- Test loss: 0.2752
- Test accuracy: 90.42%

## 3.2 Permutation Invariance

- Original accuracy: 89.98%
- Permuted accuracy (after random shuffle of input points): 89.98%

Report the percentage of test samples for which the predicted class changes after permutation:

- Percentage changed: 0.0000% (0 out of 908 samples)

Reason for 0%:

PointNet is permutation-invariant by architecture. It applies shared per-point MLPs and then uses a symmetric max-pooling operation across points, so reordering input points does not change the global feature or predicted class.

Is the result what you expect?

Yes. This is expected for PointNet because each point is processed with shared MLPs and aggregated with symmetric max pooling, which is permutation-invariant to input point order.

## 3.3 Critical Point Analysis and Robustness

### Visualizations (5 test samples)
For each sample, provide side-by-side view:
- Left: full point cloud
- Right: critical points highlighted over faint full cloud

Save either as **critical_points.png** or individual images for each sample.

### Robustness Experiment (critical-points-only input)
- Accuracy on original subset: 89.98%
- Accuracy on sparse critical-points-only subset: 90.42%

Does the accuracy drop? Briefly explain why based on the network’s architecture.

There is no drop in this run (slight increase of +0.44%). This is consistent with PointNet's design: global max pooling keeps information from the most activating points, so sparse critical points can preserve most discriminative cues for classification.


## Wandb link

DETAILED REPORT: https://api.wandb.ai/links/hexlive170-iiit-hyderabad/74tiyx57