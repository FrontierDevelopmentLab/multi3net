#!/usr/bin/env bash

mkdir -p data
cd data
wget https://s3.eu-central-1.amazonaws.com/corupublic/AAAI_harvey_data/harvey.zip
unzip harvey.zip
rm harvey.zip