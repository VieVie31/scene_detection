#! /bin/sh
echo -n "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa - aaaaaaaaaaaaaaaaaaaaaaaaaaaaa" > /cache/test.txt
cat /cache/test.txt
# create an id for the video to transform with specified paramters
function make_id {
  fingerprint=$(echo -e $@ | base64)
  echo -e "${fingerprint}_${RESOLUTION}_${FPS}"
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

# clear all the cached files
if [ $CLEAR_CACHE == '1' ];
then
  echo "Clearing cached files"
  rm -fr $CACHE_DIRECTORY/*
  touch  $CACHE_DIRECTORY/.gitkeep
fi

echo "Converting videos"
mkdir -p $CACHE_DIRECTORY/stats

cd $SOURCE_DIRECTORY

current=1
encoded_videos=''

# Ignore previous analysis traces in total
rm $CACHE_DIRECTORY/indexes.txt
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

  # FIX : Remove white spaces from encoded_videos id
  video_id="$(echo -e "${video_id}" | tr -d '[[:space:]]')"

  echo "$current;$f;$video_id" >> $CACHE_DIRECTORY/indexes.txt
  echo "[$current / $total] $f $video_id"
  # increment file counter
  ((current++))

  if [ ! -e "$CACHE_DIRECTORY/$video_id.mp4" ]
  then
    yes N | ffmpeg  -loglevel warning \
          -stats \
          -i "$f" \
          -s $RESOLUTION \
          -vf hue=s=0 \
          -an \
          -r $FPS \
          "$CACHE_DIRECTORY/$video_id.mp4"
  fi
  if [ ! -e "$CACHE_DIRECTORY/$video_id.mp4" ]
  then
    echo "Encoding error : File not found"
    exit 1 # Exit if encoding fail
  fi
  encoded_videos="$encoded_videos$video_id.mp4\n"
done

# Concatenate all encoded videos
cd $CACHE_DIRECTORY
echo "Concatenate videos"
#Creating ffmpeg input file for concatenation
echo -e "${encoded_videos::-2}" | \
  while read tab; do echo "file '$tab'"; done > list.txt


# Encoding
yes | ffmpeg \
        -safe 0 \
        -loglevel warning \
        -stats \
        -f concat \
        -i list.txt \
        -c copy \
        -s $RESOLUTION \
        -an \
        -r $FPS \
        /output.mp4

# rm -f list.txt

echo "Running hasher"
python3.5 -u /src/main.py $CACHE_DIRECTORY/indexes.txt /output.mp4
