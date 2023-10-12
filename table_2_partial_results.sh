#!/bin/bash

datapath=./datasets/VL
executablepath=./build/src
calib=$datapath/calib.txt

for model in scaling translation six_dof
do
  echo "Model: $model"
  executable=$executablepath/global_divergence_$model

  for s in 2 3
  do
    echo "  Neighborhood size: $s"

    for seq in 2D-1 2D-3 2D-5 2D-7 3D
    do
      echo "    Sequence: $seq"

      pathseq=$datapath/$seq
      events=$pathseq/events.es
      divergence=$pathseq/divergence.txt
      timestamps=$pathseq/timestamps.txt

      $executable $events $calib $divergence $timestamps -s $s -e 0.01 -m 1000 -l 1
    done
  done
done
