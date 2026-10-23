# AIDDD lab image: Boltz-2 + OpenFold3 (OpenBind checkpoints) + pose scoring on Ray 2.59.
# boltz pins an old scipy that does not build on py3.13, so this image is py3.12.
# numba (boltz featurizer) needs numpy<=2.1, so both packages resolve under one pin.
FROM anyscale/ray:2.59.0-py312-cu128

RUN echo "numpy<2.2" > /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt \
        "boltz[cuda]" "openfold3>=0.5.0" biopython pandas pyarrow fastapi && \
    boltz --help >/dev/null && \
    python -c "import openfold3, Bio, scipy, numba; print('image ok')"

# The released EV-A71 2A fine-tuned checkpoint (of3p2-ft-ev2a) is an OF3-p2
# checkpoint, which loads only in openfold3 <0.4.4. It gets its own venv that
# reuses the image's torch/CUDA via system site-packages. The image runs as
# user ray, so /opt needs sudo first.
RUN sudo mkdir -p /opt/of3-legacy && sudo chown ray /opt/of3-legacy && \
    python -m venv --system-site-packages /opt/of3-legacy && \
    /opt/of3-legacy/bin/pip install --no-cache-dir -c /tmp/constraints.txt "openfold3==0.4.3" && \
    /opt/of3-legacy/bin/run_openfold --help >/dev/null && echo "legacy of3 ok"
