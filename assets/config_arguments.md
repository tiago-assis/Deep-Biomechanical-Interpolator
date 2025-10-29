# Training Configs

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | `str` | `None` | Path to the data directory. |
| `--test` | `flag` | `False` | Whether to run the model on the test set only. |
| `--checkpoint` | `flag` | `False` | Save additional checkpoints during training. By default, checkpoints are automatically saved when the validation loss improves. This increases the frequency of checkpointing to every [--evaluate_every] epochs. |
| `--load_model` | `str` | `None` | Path to load a saved model checkpoint. |
| `--save_dir` | `str` | `'checkpoints'` | Directory to save training checkpoints. |
| `--configs` | `str` | `None` | JSON file to override CLI arguments. |
| `--split` | `float` | `[0.10, 0.15]` | Validation and test data split ratios. The first value is the ratio of cases in the validation set, and the second value is the ratio of cases in the test set. |
| `--batch_size` | `int` | `1` | Batch size (currently only 1 supported). |
| `--size` | `int, nargs=3` | `(160,192,144)` | Input tensor size to the model. |
| `--model` | `str` | `'res-unet-se'` | Model architecture to use (vxm, unet, res-unet, res-unet-se). |
| `--predict_residual` | `flag` | `False` | Predict residuals `x + f(x,y)` or full displacements `f(x,y)`. |
| `--layer_order` | `str` | `'cil'` | Order of components in convolution blocks. Check ![../model_pipeline/networks/unet3d/buildingblocks.py](../model_pipeline/networks/unet3d/buildingblocks.py) for options. |
| `--num_features`, <br> `--nfeats` | `int or str` | `32` | Feature channels per level. |
| `--num_levels`, <br> `--nlevels` | `int` | `4` | Encoder/decoder levels. |
| `--num_groups`, <br> `--ngroups` | `int` | `8` | Groups for GroupNorm. |
| `--se_mode` | `str` | `'scse'` | Squeeze-and-Excitation module (scse, cse, or sse). |
| `--upsample_mode` | `str` | `'nearest'` | Upsampling method for the decoder (nearest or trilinear). |
| `--pool_mode` | `str` | `'max'` | Pooling method for the encoder (max or avg). |
| `--mse_w` | `float` | `1.0` | Weight for the MSE loss. |
| `-e`, `--epochs` | `int` | `20` | Number of training epochs. |
| `--lr` | `float` | `1e-4` | Learning rate for the Adam optimizer. |
| `--lncc` | `flag` | `False` | Flag to use the LNCC loss. |
| `--lncc_w` | `float` | `1.0` | Weight for the LNCC loss. |
| `--lncc_window` | `int` | `2` | LNCC window size. |
| `--be`, `--hess`, <br> `--hessian` | `flag` | `False` | Flag to use bending energy penalty. |
| `--jdet` | `flag` | `False` | Flag to use Jacobian determinant regularization. |
| `--jdet_mode` | `str` | `'negative'` | Jacobian mode (`'negative'` or `'unit'`). |
| `--reg_w` | `float` | `1.0` | Regularization weight. |
| `--extra_eval` | `flag` | `False` | Flag to compute TRE, Dice, HD95 metrics during validation. |
| `--augment` | `flag` | `False` | Enable data augmentation for the preoperative scans (intensity only). |
| `--evaluate_every` | `int` | `1` | Evaluate model every N epochs. |
| `--train_save_every` | `int` | `121` | Save training metrics every N batches. If N > total number of cases, then the training metrics are saved every epoch. If N < 0, saving training metrics is disabled. |
| `--local_plot_save` | `flag` | `False` | Flag to save training plots locally. By default, plots are only saved to TensorBoard. |
| `--save_warps` | `flag` | `False` | Flag to save warped image outputs to TensorBoard for visualization. |
| `--metrics_dir` | `str` | `'metric_saves'` | Directory to save metrics/plots. |
| `--interp` | `str` | `None` | Initial interpolation method (`tps`, `linear`, None). If None, the interpolation method is randomly chosen. |
| `--num_kpts` | `int` | `None` | Number of keypoints to sample from a list of keypoint coordinates and corresponding displacements. Overrides the minimum and maximum number of keypoints flags. |
| `--min_kpts` | `int` | `20` | Minimum number of keypoints to sample from a list of keypoint coordinates and corresponding displacements. |
| `--max_kpts` | `int` | `20` | Maximum number of keypoints to sample from a list of keypoint coordinates and corresponding displacements. |
| `--run_prefix` | `str` | `''` | Filename prefix for saved run files (TensorBoard). |
| `--device` | `str` | `'cuda'` | Device to run the model (cuda or cpu). |
| `--seed` | `int` | `None` | Random seed for reproducibility. |
