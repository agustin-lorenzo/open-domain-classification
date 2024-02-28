#!/bin/bash

set -x
ls
rm fake_data.csv
touch fake_data.csv
python3 -W ignore classify.py
set +x
