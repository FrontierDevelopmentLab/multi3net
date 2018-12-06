FROM nvcr.io/nvidia/pytorch:18.06-py3

RUN pip install opencv-python numpy mkl-random pandas rasterio matplotlib

# install NGC telemetry
RUN pip install --extra-index-url=https://packages.nvidia.com/ngc/ngc-sdk/pypi/simple telemetry --upgrade

ENV PYTHONPATH "${PYTHONPATH}:/model"

ENV RESULTS_PATH "/results"

RUN mkdir -p /data/train
RUN mkdir -p /data/valid
RUN mkdir -p /results

ENV TRAINDATA_PATH "/data/train"
ENV VALIDATA_PATH "/data/valid"

COPY . /model

WORKDIR "/model"
