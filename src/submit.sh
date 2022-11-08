#!/bin/bash
#PBS -l select=2:system=polaris

# shellcheck disable=SC2039
ss=( 0.0 0.1 0.5 0.9 0.95 0.99 )
ss=( 0.9 0.95 )
for s in "${ss[@]}"
do
  bash src/GLUE.sh "$s"
  bash src/MLM.sh "$s"
  bash src/CLM.sh "$s"
  bash src/T5-WMT.sh "$s"
  bash src/T5-QA.sh "$s"
done
