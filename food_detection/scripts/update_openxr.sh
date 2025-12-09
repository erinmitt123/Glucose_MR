#!/usr/bin/env bash
set -e

set -e

hello_xr=$HOME/code/securemr/openxr/src/tests/hello_xr
if [ -n "$1" ];then
    hello_xr=$1
fi

pushd $hello_xr

if [ -d $hello_xr/.cxx ];then
    read -p "Find .cxx, remove and rebuild? [Y/n]" answer
    if [[ $answer = "n" ]]; then
        echo "Skip build hello-xr" 
    else
        rm -rf $hello_xr/.cxx
        rm -rf $hello_xr/build
        ./gradlew build
    fi
else
    echo "No found .cxx, build project."
    ./gradlew buildOpenGLESDebug
fi

popd

inc_root=$hello_xr/.cxx/Debug
lib_root=$hello_xr/build/intermediates/cxx/Debug
lib_so=

for inc_file in `find ${inc_root} -name openxr.h`; do
    if [[ $inc_file != *"arm64-v8a"*  ]]; then
        continue
    fi
    inc_root=$(dirname $inc_file)
    break
done

for lib_file in `find ${lib_root} -name libopenxr_loader.so`; do
    if [[ $lib_file != *"arm64-v8a"*  ]]; then
        continue
    fi
    lib_so=$lib_file
    break
done

dst_inc_path="external/openxr/include/openxr"
dst_lib_path="external/openxr/lib/libopenxr_loader.so"

cp $inc_root/*.h ${dst_inc_path}/
cp $lib_so ${dst_lib_path}

echo "Updated in ${dst_inc_path} and ${dst_lib_path}"
