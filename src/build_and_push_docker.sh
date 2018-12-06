#!/bin/bash

docker build -t segnet .
docker tag segnet nvcr.io/seti-fdl00/t1/segnet:$(whoami)
docker push nvcr.io/seti-fdl00/t1/segnet:$(whoami)
