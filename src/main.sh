#! /bin/sh

# created files prefix
export PREFIX=out_
SOURCE_DIRECTORY=/videos

# Set to /cache to store ffmpeg encoded videos on fs or something else
# for temp folder usage
if [ $CACHE == '1' ];
then
  CACHE_DIRECTORY=/cache
else
  CACHE_DIRECTORY=/tmp
fi

echo "Converting videos"
mkdir -p /cache/stats

cd $SOURCE_DIRECTORY

current=1
# Ignore previous analysis traces in total
((total = $(ls $SOURCE_DIRECTORY | wc -l) - $(ls $PREFIX* 2>/dev/null | wc -l)))

for f in *;
do
  # OBSOLETE
  # Does not reencode cached output from a previous analysis
  # if [[ $f == $PREFIX* ]];
  # then
    # continue
  # fi

  echo "[$current / $total] $f"
  # increment file counter
  ((current++))
  ffmpeg  -loglevel warning \
          -stats \
          -i "$f" \
          -s $RESOLUTION \
          -vf hue=s=0 \
          -an \
          -r $FPS \
          "$CACHE_DIRECTORY/$PREFIX$f"
  if [ ! -e "$CACHE_DIRECTORY/$PREFIX$f" ];
  then
    echo "Encoding error : File not found"
    exit 1 # Exit if encoding fail
  fi
done
cd /

echo "Running hasher"
python3.5 -u /src/main.py $CACHE_DIRECTORY/$PREFIX*
