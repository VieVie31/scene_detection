#! /bin/sh

echo "Converting videos"
cd videos
for f in *;
do
  echo "Encoding $f"
  ffmpeg -i $f -s 20x20 -vf hue=s=0 -an -r 10 encoded_$f
done

echo "Running hasher"
cd ..
python3 main.py videos/encoded_*
