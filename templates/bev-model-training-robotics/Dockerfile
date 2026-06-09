# BYOD image for bev-model-training-robotics.
# Bakes libgl1 + the verified dep recipe into EVERY node (head + workers). The
# notebook's `--no-deps nuscenes` pip cell only runs on the head, so Ray Data
# preprocessing on workers otherwise hits missing-nuscenes / libGL; baking the
# deps + system libGL into the image fixes all nodes consistently.
FROM anyscale/ray-llm:2.55.1-py311-cu128

# libGL.so.1 for any OpenCV/GUI codepath that runs on workers.
RUN sudo apt-get update -y && sudo apt-get install -y libgl1-mesa-glx && sudo rm -rf /var/lib/apt/lists/*

# nuscenes-devkit pins matplotlib<3.6 (no py311 wheel -> source build) and GUI
# opencv; install it WITHOUT deps, then its runtime deps as py311 wheels. Do NOT
# install opencv (base already ships headless 4.13; listing it re-triggers
# numpy>=2) and keep the base's numpy 1.x.
RUN pip install --no-cache-dir --no-deps nuscenes-devkit==1.1.11
RUN pip install --no-cache-dir pyquaternion==0.9.9 "shapely<2.0" descartes==1.1.0 fire==0.7.1 pycocotools==2.0.11 "numpy<2"
