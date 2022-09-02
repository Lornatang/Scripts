# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import torch
from model import Discriminator


new_state_dict = Discriminator().state_dict()
old_state_dict = torch.load("D_x2.pth")

new_list = []
old_list = []

for k,v in new_state_dict.items():
    new_list.append(k)


for k,v in old_state_dict["params"].items():
    old_list.append(k)


for i in range(len(new_list)):
    new_state_dict[new_list[i]] = old_state_dict["params"][old_list[i]]

torch.save({"state_dict": new_state_dict}, "Dis_x2.pth.tar")

