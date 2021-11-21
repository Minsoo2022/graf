# ["dev", "runtime"]
ARG ENV

FROM nvidia/cuda:11.1.1-devel-ubuntu20.04 as base

# Set system environment
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Set working folder
WORKDIR /app

# Install python
RUN apt-get update \
 && apt-get install -y --no-install-recommends git python3 python3-distutils python3-venv

# Install python dependencies
COPY requirements.txt /tmp/requirements.txt
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip \
 && pip install wheel \
 && pip install -r /tmp/requirements.txt

RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential python3-dev libgl1-mesa-glx

## Dev build
FROM base as dev
ONBUILD RUN chmod -R 777 /app
ONBUILD RUN chmod -R 777 $VIRTUAL_ENV

ONBUILD ARG USER_ID
ONBUILD ARG GROUP_ID

ONBUILD RUN addgroup --gid $GROUP_ID user
ONBUILD RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
ONBUILD RUN apt-get install -y --no-install-recommends sudo
ONBUILD RUN usermod -aG sudo user
ONBUILD RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ONBUILD RUN rm -rf /var/lib/apt/lists/* /tmp/*
ONBUILD USER user

# Runtim build
FROM base as runtime
ONBUILD COPY . /app
ONBUILD RUN rm -rf /var/lib/apt/lists/* /tmp/*

# ENTRYPOINT python
ONBUILD ENTRYPOINT ["python"]

FROM ${ENV}