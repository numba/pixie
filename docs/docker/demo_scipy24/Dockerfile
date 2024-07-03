FROM --platform=linux/amd64 continuumio/miniconda3:latest

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN conda create -n scipy24 numba/label/dev::numba=0.61.0dev0 numba/label/dev::llvmlite=0.44.0dev0 setuptools setuptools_scm clang=14 python=3.11 gcc_linux-64 gxx_linux-64 cython -y

RUN conda init && source /root/.bashrc && conda activate scipy24

RUN mkdir -p /root/git

RUN git clone https://github.com/numba/pixie.git /root/git/pixie

RUN git clone https://github.com/numba/numba.git /root/git/numba

RUN conda init && source /root/.bashrc && conda activate scipy24 && cd /root/git/pixie && python -m pip install --no-deps --no-index --no-build-isolation -vv -e .

RUN echo "conda activate scipy24" >> /root/.bashrc

ENTRYPOINT ["/bin/bash"]
