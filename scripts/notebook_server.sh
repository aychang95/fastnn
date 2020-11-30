#!/usr/bin/env bash

set -e

jupyter lab --ip='*' --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.iopub_data_rate_limit=1000000000.0


