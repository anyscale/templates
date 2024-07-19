#!/bin/bash
set -ex
set -o pipefail
(which aws) || (apt-get update && apt-get install -y awscli)
aws s3 sync s3://air-example-data-2/llava_example_kid_drawings/ /mnt/local_storage/kid_drawings/
