#!/bin/bash
# export our database
export MPLCONFIGDIR="/tmp"
export PYTHONPATH="${PYTHONPATH}:${PWD}"
export PATH="/venv/bin:$PATH"

# check directory if not exists
 ## declare an array dir
declare -a arr=("models" "data")

## now loop throu gh the above array
for i in "${arr[@]}"
do
   dir="${DATA_DIR}/${i}"
   [ -d "${dir}" ] || mkdir "${dir}"
done