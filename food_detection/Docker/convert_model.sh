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

# Function to check if Docker is running
docker_running() {
    docker info >/dev/null 2>&1
}

# Function to check if the Docker image exists
image_exists() {
    docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^ghcr.io/pico-developer/securemr_tool:v0"
}

# Function to pull Docker image from GitHub Container Registry
pull_docker_image() {
    echo "🔄 Pulling Docker image 'ghcr.io/pico-developer/securemr_tool:v0' from GitHub Container Registry..."
    docker pull --platform=linux/amd64 ghcr.io/pico-developer/securemr_tool:v0
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to pull the Docker image."
        exit 1
    fi
}

# Function to convert model inside the Docker container
convert_model() {
    local input_model="$1"
    local custom_io="$2"
    local model_name=$(basename -- "$input_model")
    local model_ext="${model_name##*.}"
    local custom_io_arg=""

    # Prepare custom_io argument if provided
    if [[ -n "$custom_io" ]]; then
        custom_io_arg="-i /app/$custom_io"
    fi

    # Update the docker run command in convert_model function
    case "$model_ext" in
        "onnx" | "tflite" | "pb" | "pth" | "pt")
            echo "🚀 Running $model_ext model conversion inside Docker..."
            docker run --rm --platform=linux/amd64 -v "$(pwd):/app" ghcr.io/pico-developer/securemr_tool:v0 \
                bash -c "source /opt/qnn/2.29.0.241129/bin/envsetup.sh && \
                         source /opt/qnn/2.29.0.241129/qnn229_env/bin/activate && \
                         /opt/scripts/compile.sh -m /app/$input_model $custom_io_arg"
            ;;
        *)
            echo "❌ Unsupported model format: $model_ext"
            echo "Supported formats: .onnx, .tflite, .pb (TensorFlow), .pth/.pt (PyTorch)"
            exit 1
            ;;
    esac

    if [[ $? -eq 0 ]]; then
        echo "✅ Conversion completed: Output saved in the current directory."
    else
        echo "❌ Model conversion failed."
        exit 1
    fi
}

# Check for correct arguments
function usage() {
    echo "Usage: $0 --input <model_file> [--custom_io <custom_io.yml>]"
    exit 1
}

# Update argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            INPUT_MODEL="$2"
            shift 2
            ;;
        --custom_io)
            CUSTOM_IO="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Check required argument
if [[ -z "$INPUT_MODEL" ]]; then
    usage
fi

# Check if custom_io file exists when specified
if [[ -n "$CUSTOM_IO" && ! -f "$CUSTOM_IO" ]]; then
    echo "❌ Error: Custom IO file '$CUSTOM_IO' not found."
    exit 1
fi

# Check if Docker daemon is running
if ! docker_running; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if the input file exists
if [[ ! -f "$INPUT_MODEL" ]]; then
    echo "❌ Error: File '$INPUT_MODEL' not found."
    exit 1
fi

# Check if the Docker image exists; if not, pull it from Docker Hub
if ! image_exists; then
    pull_docker_image
fi

# Run model conversion inside the container
convert_model "$INPUT_MODEL" "$CUSTOM_IO"
