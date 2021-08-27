# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import torch

from models.pmigan_v010 import Model

# 0. Define input model weights path and output model weights path.
src_model_weights_path = os.path.join("weights", "pmigan-v0.1.0.pth")
dst_model_weights_path = os.path.join("weights", "trace_pmigan-v0.1.0.pth")

# 1. PyTorch must define input tensor shape.
channels = 3
height = 32
width = 32
input_shape = [1, channels, height, width]

# 2. Define process device.
device = torch.device("cpu")

# 3. Define PyTorch model.
model = Model().to(device)

# 4. Load weights into model.
state_dict = torch.load(src_model_weights_path, map_location=device)
model.load_state_dict(state_dict)

# 5. Switch the model to eval model.
model.eval()

# 6. An example input you would normally provide to your model's forward() method.
inputs = torch.rand(input_shape, device=device)

# 7. Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_model = torch.jit.trace(model, inputs)

# 8. Save the TorchScript model
traced_script_model.save(dst_model_weights_path)
