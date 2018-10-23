#!/bin/sh

for f in *TestCase.py ;do
  echo $f
  python $f
done
