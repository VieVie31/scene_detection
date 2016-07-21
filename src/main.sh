#! /bin/sh

echo "Converting videos"
cd videos
for f in *;
do
  if [[ "$f" == *"encoded"* ]]
  then
    echo 'Skipping already encoded video.'
  else
    echo "Encoding $f"
    ffmpeg -i $f -s 20x20 -vf hue=s=0 -an -r 10 encoded_$f
  fi
done

echo "Running hasher"
cd ..
ls
python3.5 -u main.py videos/encoded_*
