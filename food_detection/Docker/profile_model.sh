#!/bin/bash

# Copyright (2025) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

_curfile=$(realpath $0)
script_dir=$(dirname $_curfile)

function usage() {
    echo "usage:"
    echo "profile_model.sh [-m model_file (required)] [-i input_raw (required)] [-n name (optional)] [-o output_dir (optional)] [-h show help message]"
}

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

model_file=""
input_raw=""
name=""
output=""

while getopts "m:i:n:o:h:" arg
do
    case $arg in 
        m)
            model_file=${OPTARG}
            ;;
        i)
            input_raw=${OPTARG}
            ;;
        n)
            name=${OPTARG}
            ;;
        o)
            output=${OPTARG}
            ;;
        h)
            usage
            exit
            ;;
        ?)
            usage
            exit 1
            ;;
    esac
done

if [ -z "${model_file}" ] || [ -z "${input_raw}" ]; then
    echo -e "${YELLOW}Model file and input raw file are required.${NC}"
    usage
    exit 1
fi

if [ ! -f "${model_file}" ]; then
    echo -e "${YELLOW}Error: Model file '${model_file}' not found.${NC}"
    exit 1
fi

if [ ! -f "${input_raw}" ]; then
    echo -e "${YELLOW}Error: Input raw file '${input_raw}' not found.${NC}"
    exit 1
fi

if [ -z "${name}" ]; then
    filename=$(basename -- "$model_file")
    name="${filename%.*}"
fi

if [ -z "${output}" ]; then
    output="./${name}_profile"
fi
output=$(realpath "$output")
mkdir -p "${output}"

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo -e "${YELLOW}No Android device connected${NC}"
    exit 1
fi

# Create remote directory
remote_dir="/data/local/tmp/${name}"
adb shell "mkdir -p ${remote_dir}"

# Create necessary subdirectories
adb shell "mkdir -p ${remote_dir}/cpu ${remote_dir}/dsp"

# Push QNN libraries and binaries to appropriate directories
echo "Pushing QNN libraries to device..."
# CPU libraries
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnChrometraceProfilingReader.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnCpu.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnDsp.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnDspNetRunExtensions.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnDspV66Stub.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnGpu.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnGpuNetRunExtensions.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHta.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtaNetRunExtensions.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpNetRunExtensions.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpProfilingReader.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV68Stub.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV69Stub.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV73Stub.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV75Stub.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSaver.so ${remote_dir}/cpu/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so ${remote_dir}/cpu/

# DSP library
adb push ${QNN_SDK_ROOT}/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so ${remote_dir}/dsp/

# Push qnn-net-run binary
adb push ${QNN_SDK_ROOT}/bin/aarch64-android/qnn-net-run ${remote_dir}/

# Push model and input
adb push ${model_file} ${remote_dir}/model.bin
adb push ${input_raw} ${remote_dir}/input.raw

# Create input list file
echo "${remote_dir}/input.raw" > "${output}/input_list.txt"
adb push "${output}/input_list.txt" "${remote_dir}/input_list.txt"

# Set up environment on device
adb shell "cd ${remote_dir} && chmod +x qnn-net-run"
adb shell "cd ${remote_dir}/cpu && chmod 755 libQnnHtp.so"
adb shell "cd ${remote_dir}/dsp && chmod 755 libQnnHtpV69Skel.so"

# Export library path and verify libraries
adb shell "cd ${remote_dir} && \
    export LD_LIBRARY_PATH=${remote_dir}/cpu:\$LD_LIBRARY_PATH && \
    ldd ${remote_dir}/qnn-net-run && \
    ldd ${remote_dir}/cpu/libQnnHtp.so"

# Run profiling with performance options and debug output
echo "🚀 Starting model profiling..."
start_time=$(date +%s)

# Create log directory
adb shell "mkdir -p ${remote_dir}/logs"

# Print and run the command
cmd="cd ${remote_dir} && \
    export LD_LIBRARY_PATH=${remote_dir}/cpu:\$LD_LIBRARY_PATH && \
    export ADSP_LIBRARY_PATH=${remote_dir}/dsp && \
    export CDSP_ID=0 && \
    chmod +x ${remote_dir}/qnn-net-run && \
    ${remote_dir}/qnn-net-run \
    --backend ${remote_dir}/cpu/libQnnHtp.so \
    --retrieve_context model.bin \
    --input_list input_list.txt \
    --output_dir ${remote_dir}/output_dir \
    --num_inferences 10 \
    --profiling_level basic \
    --perf_profile high_performance \
    --log_level verbose \
    2>&1"

echo -e "${YELLOW}Executing command:${NC}"
echo "$cmd"
echo

# Execute command and save output
adb shell "$cmd" > "${output}/profiling.log"
cat "${output}/profiling.log"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
avg_time_ms=$((elapsed * 1000 / 1))
echo -e "${GREEN}🎉 Profiling completed! Total time: ${elapsed}s, Average per inference: ${avg_time_ms}ms${NC}"

# Pull the logs
adb pull "${remote_dir}/logs" "${output}/"
adb pull "${remote_dir}/output_dir" "${output}/"

# Parse profiling data locally
echo "📊 Parsing profiling data..."
${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-profile-viewer \
    --input_log "${output}/output_dir/qnn-profiling-data.log" \
    --output_format text \
    > "${output}/profile_summary.txt"

# Display the parsed results
echo -e "${GREEN}Profile Summary:${NC}"
cat "${output}/profile_summary.txt"

# remove the temporary directory on device
adb shell "rm -rf ${remote_dir}"
