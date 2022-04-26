ARG CONDA_VER=4.9.2

###########
FROM continuumio/miniconda3:${CONDA_VER} as scitools-builder
ARG USERNAME=container
ARG USERID=10001
RUN useradd --no-create-home --uid ${USERID} --user-group --shell /bin/bash ${USERNAME} \
    && sed -i -r "/^(${USERNAME}|root)/!d" /etc/group /etc/passwd \
    && sed -i -r 's#^(.*):[^:]*$#\1:/sbin/nologin#' /etc/passwd
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get --allow-releaseinfo-change update \
    && apt-get install --yes --no-install-recommends \
    build-essential=12.6 \
    gcc=4:8.3.0-1 \
    python3-dev=3.7.3-1 \
    libcurl4-openssl-dev=7.64.0-4+deb10u2 \
    libssl-dev=1.1.1d-0+deb10u7 \
    libboost-dev=1.67.0.1 \
    libboost-system-dev=1.67.0.1 \
    libboost-filesystem-dev=1.67.0.1

COPY dockers/environment.yml /tmp/environment.yml
ENV PIP_NO_CACHE_DIR=1
RUN conda env create  --name sci-tools -f /tmp/environment.yml \
    && conda clean -afy

# Use conda-pack to create a standalone enviornment in /venv:
RUN mkdir /venv
WORKDIR /venv
RUN conda install -c conda-forge conda-pack
RUN conda-pack -n sci-tools -o /tmp/env.tar --dest-prefix /venv --compress-level 0 --n-threads -1 --exclude "*.a" --exclude "*.pyc" --exclude "*.js.map" \
    && tar xf /tmp/env.tar \
    && rm /tmp/env.tar

###########
FROM debian:buster-20210511-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=0
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates=20200601~deb10u2 \
    libcurl4=7.64.0-4+deb10u2 \
    && apt-get -y autoremove --purge \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=scitools-builder /etc/passwd /etc/group /etc/shadow /etc/
## venv
COPY --from=scitools-builder --chown=10001 /venv /venv

WORKDIR /app
COPY --chown=10001 bin ./bin
COPY --chown=10001 serpml ./serpml
RUN chmod a+x,g+x /app/bin/filter/*
RUN chmod a+x,g+x /app/bin/ctr/*
USER 10001

