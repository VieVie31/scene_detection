if [ "$1" != "cache" ]; then
    echo "For not building again and again write :"
    echo "$ ./run.sh cache"

    #build
    docker build -t scd .
fi

#run with videos in the volume...
docker run -v $(pwd)/src:/src -v $(pwd)/videos:/videos scd

#rm -fr ./src/__pycache__