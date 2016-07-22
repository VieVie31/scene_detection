#! /bin/sh

echo "Converting videos"
# creating tmp encoded datas destination folder
# /!\ Removed when container exits
mkdir /datas

cd videos
for f in *;
do
  echo "Encoding $f"
  ffmpeg -i $f -s 20x20 -vf hue=s=0 -an -r 5 /datas/$f
done

echo "Running hasher"
ls /datas
python3.5 -u /src/main.py /datas/*
