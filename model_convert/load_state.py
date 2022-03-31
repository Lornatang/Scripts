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

import csnln
import model

ours_x2_state_dict = model.CSNLN(2).state_dict()
offical_x2_state_dict = csnln.CSNLN(2).state_dict()

ours_key = [
    "conv1.0.weight",
    "conv1.0.bias",
    "conv1.1.weight",
    "conv1.2.weight",
    "conv1.2.bias",
    "conv1.3.weight",
    "self_exemplar_mining.multi_source_projection1.cross_scale_attention.escape",
    "self_exemplar_mining.multi_source_projection1.cross_scale_attention.conv1.0.weight",
    "self_exemplar_mining.multi_source_projection1.cross_scale_attention.conv1.0.bias",
    "self_exemplar_mining.multi_source_projection1.cross_scale_attention.conv1.1.weight",
    "self_exemplar_mining.multi_source_projection1.cross_scale_attention.conv2.0.weight",
    "self_exemplar_mining.multi_source_projection1.cross_scale_attention.conv2.0.bias",
    "self_exemplar_mining.multi_source_projection1.cross_scale_attention.conv2.1.weight",
    "self_exemplar_mining.multi_source_projection1.cross_scale_attention.conv_assembly.0.weight",
    "self_exemplar_mining.multi_source_projection1.cross_scale_attention.conv_assembly.0.bias",
    "self_exemplar_mining.multi_source_projection1.cross_scale_attention.conv_assembly.1.weight",
    "self_exemplar_mining.multi_source_projection1.non_local_attention.conv1.0.weight",
    "self_exemplar_mining.multi_source_projection1.non_local_attention.conv1.0.bias",
    "self_exemplar_mining.multi_source_projection1.non_local_attention.conv1.1.weight",
    "self_exemplar_mining.multi_source_projection1.non_local_attention.conv2.0.weight",
    "self_exemplar_mining.multi_source_projection1.non_local_attention.conv2.0.bias",
    "self_exemplar_mining.multi_source_projection1.non_local_attention.conv2.1.weight",
    "self_exemplar_mining.multi_source_projection1.non_local_attention.conv_assembly.0.weight",
    "self_exemplar_mining.multi_source_projection1.non_local_attention.conv_assembly.0.bias",
    "self_exemplar_mining.multi_source_projection1.non_local_attention.conv_assembly.1.weight",
    "self_exemplar_mining.multi_source_projection1.upsampling.0.weight",
    "self_exemplar_mining.multi_source_projection1.upsampling.0.bias",
    "self_exemplar_mining.multi_source_projection1.upsampling.1.weight",
    "self_exemplar_mining.multi_source_projection1.encoder.rcb.0.weight",
    "self_exemplar_mining.multi_source_projection1.encoder.rcb.0.bias",
    "self_exemplar_mining.multi_source_projection1.encoder.rcb.1.weight",
    "self_exemplar_mining.multi_source_projection1.encoder.rcb.2.weight",
    "self_exemplar_mining.multi_source_projection1.encoder.rcb.2.bias",
    "self_exemplar_mining.down_conv1.0.weight",
    "self_exemplar_mining.down_conv1.0.bias",
    "self_exemplar_mining.down_conv1.1.weight",
    "self_exemplar_mining.down_conv2.0.weight",
    "self_exemplar_mining.down_conv2.0.bias",
    "self_exemplar_mining.down_conv2.1.weight",
    "self_exemplar_mining.diff_encode1.0.weight",
    "self_exemplar_mining.diff_encode1.0.bias",
    "self_exemplar_mining.diff_encode1.1.weight",
    "self_exemplar_mining.conv.0.weight",
    "self_exemplar_mining.conv.0.bias",
    "self_exemplar_mining.conv.1.weight",
    "conv2.weight",
    "conv2.bias",
]

offical_key = [
    "head.0.0.weight",
    "head.0.0.bias",
    "head.0.1.weight",
    "head.1.0.weight",
    "head.1.0.bias",
    "head.1.1.weight",
    "SEM.multi_source_projection.up_attention.escape_NaN",
    "SEM.multi_source_projection.up_attention.conv_match_1.0.weight",
    "SEM.multi_source_projection.up_attention.conv_match_1.0.bias",
    "SEM.multi_source_projection.up_attention.conv_match_1.1.weight",
    "SEM.multi_source_projection.up_attention.conv_match_2.0.weight",
    "SEM.multi_source_projection.up_attention.conv_match_2.0.bias",
    "SEM.multi_source_projection.up_attention.conv_match_2.1.weight",
    "SEM.multi_source_projection.up_attention.conv_assembly.0.weight",
    "SEM.multi_source_projection.up_attention.conv_assembly.0.bias",
    "SEM.multi_source_projection.up_attention.conv_assembly.1.weight",
    "SEM.multi_source_projection.down_attention.conv_match1.0.weight",
    "SEM.multi_source_projection.down_attention.conv_match1.0.bias",
    "SEM.multi_source_projection.down_attention.conv_match1.1.weight",
    "SEM.multi_source_projection.down_attention.conv_match2.0.weight",
    "SEM.multi_source_projection.down_attention.conv_match2.0.bias",
    "SEM.multi_source_projection.down_attention.conv_match2.1.weight",
    "SEM.multi_source_projection.down_attention.conv_assembly.0.weight",
    "SEM.multi_source_projection.down_attention.conv_assembly.0.bias",
    "SEM.multi_source_projection.down_attention.conv_assembly.1.weight",
    "SEM.multi_source_projection.upsample.0.weight",
    "SEM.multi_source_projection.upsample.0.bias",
    "SEM.multi_source_projection.upsample.1.weight",
    "SEM.multi_source_projection.encoder.body.0.weight",
    "SEM.multi_source_projection.encoder.body.0.bias",
    "SEM.multi_source_projection.encoder.body.1.weight",
    "SEM.multi_source_projection.encoder.body.2.weight",
    "SEM.multi_source_projection.encoder.body.2.bias",
    "SEM.down_sample_1.0.weight",
    "SEM.down_sample_1.0.bias",
    "SEM.down_sample_1.1.weight",
    "SEM.down_sample_2.0.weight",
    "SEM.down_sample_2.0.bias",
    "SEM.down_sample_2.1.weight",
    "SEM.error_encode.0.weight",
    "SEM.error_encode.0.bias",
    "SEM.error_encode.1.weight",
    "SEM.post_conv.0.weight",
    "SEM.post_conv.0.bias",
    "SEM.post_conv.1.weight",
    "tail.0.weight",
    "tail.0.bias",
]

for our, offical in zip(ours_key, offical_key):
    ours_x2_state_dict[our] = offical_x2_state_dict[offical]

torch.save(ours_x2_state_dict, "CSNLN_x2-DIV2K.pth.tar")
