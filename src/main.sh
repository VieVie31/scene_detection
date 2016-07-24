#! /bin/sh

# created files prefix
export PREFIX=out_
SOURCE_DIRECTORY=/videos

# Set to /videos to store ffmpeg encoded videos on fs or something else
# for temp folder usage
DESTINATION_DIRECTORY=/videos


echo "Converting videos"
# creating tmp encoded datas destination folder
# /!\ Removed when container exits
mkdir -p $DESTINATION_DIRECTORY

cd $SOURCE_DIRECTORY

# remove old png exports, ensuring analysis won't try to use them as videos
rm -vf $PREFIX*.png

current=1
# Ignore previous analysis traces in total
((total = $(ls $SOURCE_DIRECTORY | wc -l) - $(ls $PREFIX* 2>/dev/null | wc -l)))

for f in *;
do
  # Does not reencode cached output from a previous analysis
  if [[ $f == $PREFIX* ]];
  then
    continue
  fi

  echo "[$current / $total] $f"
  # increment file counter
  ((current++))
  ffmpeg  -loglevel warning \
          -stats \
          -i "$f" \
          -s 8x8 \
          -vf hue=s=0 \
          -an \
          -r 1 \
          "$DESTINATION_DIRECTORY/$PREFIX$f"

done
cd /

echo "Running hasher"
python3.5 -u /src/main.py $DESTINATION_DIRECTORY/$PREFIX*
