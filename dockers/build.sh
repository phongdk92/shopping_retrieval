#!/usr/bin/env bash

VERSION_BUILD=$(date +"%Y%m%d")
VERSION="1.0.0.${VERSION_BUILD}"

if [[ -z "$1" ]]; then
  VERSION="${VERSION}"
else
  VERSION="${VERSION}-$1"
fi

#speed up docker
#alias docker='DOCKER_BUILDKIT=1 docker'

IMAGE_TAG=ccr.itim.vn/bi/shopping

#login to docker-hub
docker build -t $IMAGE_TAG:$VERSION -f dockers/Dockerfile .
#docker login -u "$CI_REGISTRY_USER" -p "$CI_REGISTRY_PASSWORD" $CI_REGISTRY
#docker push $IMAGE_TAG:$VERSION
echo $IMAGE_TAG:$VERSION