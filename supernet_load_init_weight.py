import torch
import time
import torch.nn as nn
import math
from pathlib import Path
from model.supernet_transformer import Vision_TransformerSuper

def load_weight(img_size=240, patch_size=20, embed_dim=256, depth=14, num_heads=4, 
                mlp_ratio=4, qkv_bias=True, drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.,
                gp=True, num_classes=1000, max_relative_position=14, 
                relative_position=True, change_qkv=True, abs_pos=True,
                rank_ratio=0.9, pretrained_output_dir=None, output_path=None):
    # create the modified model
    model_modify = Vision_TransformerSuper(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, 
                                    depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                    drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate,
                                    gp=gp, num_classes=num_classes, max_relative_position=max_relative_position, 
                                    relative_position=relative_position, change_qkv=change_qkv, 
                                    abs_pos=abs_pos, rank_ratio=rank_ratio)
    # print(model_modify)
    # model_modify = deit_tiny_patch16_224(pretrained=False)
    # torch.save(model_modify, "./models/supernet/model_modify.pth")
    
    # get the pretrained model weight
    output_dir_path = Path(pretrained_output_dir)
    pretrained_model_path = output_dir_path / 'supernet-tiny.pth'
    pretext_model = torch.load(pretrained_model_path, map_location='cpu')
    pretext_model_dict = pretext_model["model"]
    # pretext_model_dict = pretext_model.state_dict()
    # print(pretext_model_dict.keys())


    # get the matchable layers
    model_modified_dict = model_modify.state_dict()
    matchable_layer = {k:v for k,v in pretext_model_dict.items() 
                    if (k in model_modified_dict.keys()) 
                    and (k != "patch_embed_super.proj.weight")
                    and (k != "patch_embed_super.proj.bias")
                    and (k != "pos_embed")}

    # modify the unmatchable layers
    unmatchable_layer = {k:v for k,v in pretext_model_dict.items() if not k in model_modified_dict.keys()}
    modified_layer = {}
    token_num = int(round(img_size/patch_size,0) * round(img_size/patch_size,0)) + 1
    patch_embed_super_proj_weight_value = nn.Parameter(torch.Tensor(embed_dim, 3, patch_size, patch_size))
    pos_embed_value = nn.Parameter(torch.Tensor(1, token_num, embed_dim))
    patch_embed_super_proj_bias_value = nn.Parameter(torch.Tensor(embed_dim))
    nn.init.kaiming_uniform_(patch_embed_super_proj_weight_value, a=math.sqrt(5))
    nn.init.kaiming_uniform_(pos_embed_value, a=math.sqrt(5))
    nn.init.uniform_(patch_embed_super_proj_bias_value, 0, 5)
    # nn.init.kaiming_uniform_(patch_embed_super_proj_bias_value, a=math.sqrt(5))
    k_new = "patch_embed_super.proj.weight"
    v_new = patch_embed_super_proj_weight_value
    modified_layer.update({k_new:v_new})
    k_new1 = "pos_embed"
    v_new1 = pos_embed_value
    modified_layer.update({k_new1:v_new1})
    k_new2 = "patch_embed_super.proj.bias"
    v_new2 = patch_embed_super_proj_bias_value
    modified_layer.update({k_new2:v_new2})
    # print(unmatchable_layer.keys())

    # modify the fc layers in the attention
    i = 0
    for k,v in unmatchable_layer.items():
        if k == "blocks." + str(int(i)) + ".attn.qkv.weight":
            v_qkv1,v_qkv2_diag,v_qkv3 = torch.svd_lowrank(v, q = int(rank_ratio*embed_dim))
            k_qkv1 = "blocks." + str(int(i)) + ".attn.qkv1.weight"
            modified_layer.update({k_qkv1:v_qkv3.t()})
            k_qkv2 = "blocks." + str(int(i)) + ".attn.qkv2.weight"
            v_qkv2 = torch.diag(v_qkv2_diag)
            modified_layer.update({k_qkv2:v_qkv2})
            k_qkv3 = "blocks." + str(int(i)) + ".attn.qkv3.weight"
            modified_layer.update({k_qkv3:v_qkv1})
        elif k == "blocks." + str(int(i-0.25)) + ".attn.qkv.bias":
            k_bias = "blocks." + str(int(i-0.25)) + ".attn.qkv3.bias"
            v_bias = v
            modified_layer.update({k_bias:v_bias})
        elif k ==  "blocks." + str(int(i-0.5)) + ".attn.proj.weight":
            v_proj1,v_proj2_diag,v_proj3 = torch.svd_lowrank(v, q = int(rank_ratio*embed_dim))
            k_proj1 = "blocks." + str(int(i-0.5)) + ".attn.proj1.weight"
            modified_layer.update({k_proj1:v_proj3.t()})
            k_proj2 = "blocks." + str(int(i-0.5)) + ".attn.proj2.weight"
            v_fc12 = torch.diag(v_proj2_diag)
            modified_layer.update({k_proj2:v_fc12})
            k_proj3 = "blocks." + str(int(i-0.5)) + ".attn.proj3.weight"
            modified_layer.update({k_proj3:v_proj1})
        elif k == "blocks." + str(int(i-0.75)) + ".attn.proj.bias":
            k_bias = "blocks." + str(int(i-0.75)) + ".attn.proj3.bias"
            v_bias = v
            modified_layer.update({k_bias:v_bias})
        else:
            continue
        i += 0.25

    # modify the fc layers in the mlp
    i = 0
    for k,v in unmatchable_layer.items():
        if k ==  "blocks." + str(int(i)) + ".fc1.weight":
            v_fc11,v_fc12_diag,v_fc13 = torch.svd_lowrank(v, q = int(rank_ratio*embed_dim))
            k_fc11 = "blocks." + str(int(i)) + ".fc11.weight"
            modified_layer.update({k_fc11:v_fc13.t()})
            k_fc12 = "blocks." + str(int(i)) + ".fc12.weight"
            v_fc12 = torch.diag(v_fc12_diag)
            modified_layer.update({k_fc12:v_fc12})
            k_fc13 = "blocks." + str(int(i)) + ".fc13.weight"
            modified_layer.update({k_fc13:v_fc11})

        elif k == "blocks." + str(int(i-0.25)) + ".fc1.bias":
            k_bias = "blocks." + str(int(i-0.25)) + ".fc13.bias"
            v_bias = v
            modified_layer.update({k_bias:v_bias})

        elif k ==  "blocks." + str(int(i-0.5)) + ".fc2.weight":
            v_fc11,v_fc12_diag,v_fc13 = torch.svd_lowrank(v, q = int(rank_ratio*embed_dim))
            k_fc11 = "blocks." + str(int(i-0.5)) + ".fc21.weight"
            modified_layer.update({k_fc11:v_fc13.t()})
            k_fc12 = "blocks." + str(int(i-0.5)) + ".fc22.weight"
            v_fc12 = torch.diag(v_fc12_diag)
            modified_layer.update({k_fc12:v_fc12})
            k_fc13 = "blocks." + str(int(i-0.5)) + ".fc23.weight"
            modified_layer.update({k_fc13:v_fc11})
            
        elif k == "blocks." + str(int(i-0.75)) + ".fc2.bias":
            k_bias = "blocks." + str(int(i-0.75)) + ".fc23.bias"
            v_bias = v
            modified_layer.update({k_bias:v_bias})
        else:
            continue
        i += 0.25


    # print("####",modified_layer.keys())

    # # update weight
    model_modified_dict.update(matchable_layer)
    model_modified_dict.update(modified_layer)
    model_modify.load_state_dict(model_modified_dict)


    # # check
    # a=[]
    # for i in model_modified_dict.keys():
    #     a.append(i)
    # for i in matchable_layer.keys():
    #     a.remove(i)
    # for i in modified_layer.keys():
    #     a.remove(i)
    # print(a)
    # asd

    model_dict = model_modify.state_dict()
    initial_model_path = output_path / 'supernet_tiny_weight.pth'
    torch.save(model_dict, initial_model_path)
    print("Successfully generate pre-trained model")
    time.sleep(3)
    path = initial_model_path
    return path

# load_weight(img_size=240, patch_size=20, embed_dim=256, depth=14, num_heads=4, 
#                 mlp_ratio=4, qkv_bias=True, drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.,
#                 gp=True, num_classes=1000, max_relative_position=14, 
#                 relative_position=True, change_qkv=True, abs_pos=True,
#                 rank_ratio=0.9, output_dir="./result")