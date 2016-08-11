#! /bin/sh

# create an id for the video to transform with specified paramters
function make_id {
  md5=$(md5sum $1)
  out=$(echo ${md5}_${RESOLUTION}_${FPS})
  echo $out 
}

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

  video_id=$(make_id $f)

  echo "[$current / $total] $f $video_id"
  # increment file counter
  ((current++))
  ffmpeg  -loglevel warning \
          -stats \
          -i "$f" \
          -s $RESOLUTION \
          -vf hue=s=0 \
          -an \
          -r $FPS \
          "$CACHE_DIRECTORY/$video_id.mp4"
  if [ ! -e "$CACHE_DIRECTORY/$video_id.mp4" ];
  then
    echo "Encoding error : File not found"
    exit 1 # Exit if encoding fail
  fi
done
cd /

echo "Running hasher"
python3.5 -u /src/main.py $CACHE_DIRECTORY/*mp4
