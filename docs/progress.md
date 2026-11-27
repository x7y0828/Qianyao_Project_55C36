# Project Progress Log

## Stage 1: Project Setup
- Selected Track I: Image Processing.
- Created the GitHub repository.
- Studied Chambolle's TV denoising method.

## Stage 2: Baseline Implementation
- Generated synthetic images with Gaussian noise.
- Implemented gradient and divergence operators.
- Implemented and tested the baseline algorithm.
- Checked the boundary conditions and operator consistency.

## Stage 3: Modified Method
- Implemented MAD-based noise estimation.
- Added noise-adaptive regularisation.
- Selected parameters using validation cases.

## Stage 4: Experimental Comparison
- Compared both methods at three noise levels.
- Evaluated MSE, PSNR, Gradient MSE and runtime.
- Found that the adaptive method performed better at low
  noise levels but worse at high noise levels.

## Stage 5: Further Analysis
- Conducted a stress test and an ablation study.
- Observed texture loss caused by TV denoising.
- Investigated the limitations of adaptive regularisation.

## Stage 6: Documentation
- Created experiment configurations.
- Completed requirements.txt and README.md.
- Verified the smoke and full experiment runs.

## Next Steps
- Review the code.
- Test reproducibility from a clean clone.
- Complete the report and presentation.
- Prepare the final submission.