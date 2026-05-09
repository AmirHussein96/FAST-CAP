# docker build --progress plain -t gitlab-master.nvidia.com/sanggill/bigvgan:24.02-py3-v2 .
# docker push gitlab-master.nvidia.com/sanggill/bigvgan:24.02-py3-v2
# cd /lustre/fsw/portfolios/adlr/users/sanggill/docker
# enroot import -o bigvgan:24.02-py3-v2.sqsh docker://gitlab-master.nvidia.com/sanggill/bigvgan:24.02-py3-v2
#############################
# start of Dockerfile
#############################
FROM gitlab-master.nvidia.com/sanggill/playground:24.02-py3-v2
WORKDIR /app
#############################
# additional packages used by BigVGAN
#############################
# visqol, probably the the best i can get in obj. metrics for vocoder/codec. install bazel=5.3.2 to build visqol-3.3.3
RUN apt install apt-transport-https curl gnupg -y
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
RUN mv bazel-archive-keyring.gpg /usr/share/keyrings
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update && apt-get install -y \
    bazel=5.3.2 \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/google/visqol
# bazel build output defaults to /root/.cache/bazel, which does not exist after building the image (after starting the container in the cluster, /root is actually mapped to ~)
# Modify the setup.py file to change the os.system calls to point --output_user_root to /app/bazel
RUN sed -i 's|os.system("bazel build|os.system("bazel --output_user_root=/app/bazel build|g' /app/visqol/setup.py
RUN pip install -v /app/visqol
RUN pip install auraloss pesq pytorch_warmup fastdtw pysptk torchcrepe pytorch_lightning sacrebleu bitarray nnAudio torchsynth
RUN pip install --no-deps fairseq
RUN pip install --no-deps mel-cepstral-distance