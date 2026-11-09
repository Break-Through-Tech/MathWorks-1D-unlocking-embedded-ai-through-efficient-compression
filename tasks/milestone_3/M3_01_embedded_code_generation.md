# M3 Task 1: Embedded Code Generation

For this stretch goal, we used MATLAB Coder to generate C code from the
project's projected, fine-tuned 1-D CNN. The goal was to see whether the
trained model could run outside the usual MATLAB prediction workflow and to
check that code generation did not change its predictions.

## Model and input

We chose `data/models/projected/projected_finetuned_net.mat` (variable `net`).
In the earlier validation comparison, this model kept 100% accuracy while
using about 80% fewer learnable parameters than the baseline.

The generated entry point expects one raw, single-precision signal with
5,000 samples. It standardizes the signal using the training-set `mu` and
`sigma` in `data/scaler.mat`, runs the projected, fine-tuned network, and
returns three class scores, one for each fault category. This run also checked
the model's predictions on the held-out test set.

## What we ran

MATLAB Online was running R2026b. MATLAB Coder and Deep Learning Toolbox
licenses were available, and the session had GCC 12. We generated a MEX build
first so it could be called from MATLAB and checked against the regular
network. Then we generated a static library and its C source.

The reproducible command, from the project folder in MATLAB, is:

```matlab
run("scripts/M3_01_generate_embedded_code.m")
```

The script compares the MATLAB and generated scores for all 102 held-out test
signals. It checks that predicted classes match and that the largest absolute
score difference is no more than `1e-4`.

## Results

Both generation steps succeeded! MATLAB and the generated MEX had the same
predicted class for every test signal, and both were correct on 102 out of
102 signals. The largest score difference was `6.111804e-10`, far below the
script's `1e-4` tolerance.

The generated C source and static library are in MATLAB Drive under
`embedded_code_goal/codegen/lib/M3_01_embedded_predict/`. The library is
`M3_01_embedded_predict.a`. The MEX build and its report are under
`embedded_code_goal/codegen/mex/M3_01_embedded_predict/`. These generated
files are in the MATLAB Online workspace, not in this Git repository; the
script can recreate them.

To open the generated C entry point in MATLAB Online:

```matlab
edit("codegen/lib/M3_01_embedded_predict/M3_01_embedded_predict.c")
```

## What this tells us

MATLAB Coder generated C for the model, and the generated predictions matched
MATLAB's on the held-out test set. That completes our code-generation stretch goal for Milestone 3!

Quick Note: For this stretch goal and project as a whole, code
generation and prediction verification were the results we were looking for. They show
that this model can be generated and run in the tested MATLAB environment, but
they do not tell us how fast it would run or how much memory it would use on a
particular device. In the future, a hardware follow-up would be cool to explore, but we would need a target board and
toolchain, followed by measurements on that hardware as well.

## Common questions

- **What goes into the generated function?** One raw signal containing
  5,000 samples, passed as single-precision data.
- **Does the input need preprocessing first?** No separate preprocessing step
  is needed when using this entry point. It applies the saved training-set
  `mu` and `sigma` before running the network.
- **What does the function return?** Three class scores, one per fault
  category. The highest score corresponds to the predicted class.
- **Why use the projected, fine-tuned model?** In the validation comparison,
  it kept 100% accuracy while reducing learnable parameters by about 80%
  compared with the baseline. We then checked its predictions on the held-out
  test set.
- **What do the test results say?** MATLAB and generated MEX predictions
  agreed on all 102 test signals. Both classified all 102 correctly, and the
  maximum difference between their scores was `6.111804e-10`.
- **What is the MEX file for?** It lets MATLAB call the compiled function, so
  we could compare generated-code results with the MATLAB network. It is a
  verification step, not the embedded program itself.
- **What code did Coder generate?** It generated C source and a static library.
  The files are in the MATLAB Online workspace, not this Git repository.
- **Does this mean the model has been tested on a device?** No. It confirms
  code generation and prediction agreement in MATLAB Online. Testing on a
  particular embedded board would require building for that board and
  measuring its memory use and speed!