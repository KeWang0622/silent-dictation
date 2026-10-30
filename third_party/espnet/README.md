# Vendored ESPnet subset

Only the modules needed to run and fine-tune the Auto-AVSR visual speech recognition model.
Origin: ESPnet (Apache-2.0) as shipped with Auto-AVSR (Apache-2.0) and Chaplin (MIT); see the
top-level NOTICE. Local modifications:

- replaced `distutils` imports (removed in Python 3.12) with local equivalents / `packaging.version`;
- `nets/ctc_prefix_score.py` keeps tensors on the input's device, which adds Apple MPS support;
- removed unused `import pdb` lines.

Cross-attention caching for fast beam search lives in `silentdict/vsr.py` and does not modify these files.
