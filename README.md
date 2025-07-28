# Deep Biomechanically-Guided Interpolation for Keypoint-Based Brain Shift Registration

### Framework Overview

Our framework estimates a dense and physically plausible displacement field from a sparse set of matched keypoints between pre- and intra-operative images. The core of our method is a deep, biomechanically guided interpolator that refines an initial displacement field.

The process is as follows:

  1. Synthetic Data Generation: We generate a large dataset of realistic, synthetic brain deformations using biomechanical simulations based on the UPENN-GBM dataset. This simulates brain deformations caused by factors like gravity and tissue resection.

  2. Keypoint Simulation: Sparse matched keypoints are simulated. Keypoints are extracted from the preoperative MRI using 3D SIFT, and their corresponding displacement vectors are retrieved from the ground-truth synthetic deformation fields.

  3. Deep Interpolation:

        - An initial dense displacement field is created from the sparse keypoints using a standard interpolation method (e.g., Linear or Thin-Plate Spline).

        - A residual 3D U-Net takes the preoperative MRI and the initial displacement field as input.

        - The network is trained to predict a residual displacement field, which refines the initial estimate to be more physically plausible and accurate.

        - The model is trained with a voxel-wise error loss and a Jacobian-based regularization loss to encourage smooth deformations.

![Overview of the proposed framework](imgs/framework_pipeline.png)
Fig 1. Overview of the proposed framework.
