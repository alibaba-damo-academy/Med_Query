#!/usr/bin/env bash
TAG=$1
res=$(ps aux | grep train.py| grep $TAG | grep -v grep | awk '{print $2}')
if [ "$res" ]
then
  kill $res
else
  echo "no process found with tag "$TAG
fi