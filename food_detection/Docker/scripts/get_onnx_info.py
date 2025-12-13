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

import onnx
import argparse

parser = argparse.ArgumentParser(description='get onnx info')
parser.add_argument('onnxfile', type=str, help='input onnx file')
args = parser.parse_args()
model = onnx.load(args.onnxfile)
for input in model.graph.input:
    input_name = input.name
    shape = [int(dim.dim_value) for dim in input.type.tensor_type.shape.dim]
    res = f"-d {input_name} {shape}"
    print(res)
