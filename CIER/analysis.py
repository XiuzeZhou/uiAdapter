import torch
import random
import os
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from torch.optim import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import pearsonr

from dataloader import *
from utils import *
from model import *
from peft import LoraConfig, get_peft_model, TaskType
from argparse import ArgumentParser

# ==========================================
# 1. Parameter parsing and environment initialization
# ==========================================
parser = ArgumentParser()

parser.add_argument('--devices', default=0, type=int, help='Select which GPU to use with the program.')
parser.add_argument('--batch_size', default=40, type=int)
parser.add_argument('--seed', default=5254, type=int)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--accumulation_steps', default=1, type=int)
parser.add_argument('--rating_weight', default=0.1, type=float, help='regularization on recommendation task')
parser.add_argument('--generate_weight', default=1.0, type=float, help='regularization on generation task')
parser.add_argument('--delta', default=0.2, type=float)
parser.add_argument('--word', default=20, type=int, help='number of words to generate for each sample')
parser.add_argument('--show_train_loss_steps', default=500, type=int, help='number of train steps for display the loss')
parser.add_argument('--id_hidden', default=1024, type=int)
parser.add_argument('--only_eval', action='store_true')

parser.add_argument('--dataset_name', default='ClothingShoesAndJewelry', type=str)
parser.add_argument('--data_dir', default='./data/', type=str)
parser.add_argument('--model_name', default='../autodl-fs/Qwen2.5-7B/', type=str)
parser.add_argument('--ckpt_dir', default='../autodl-tmp/models/qwen/', type=str)
parser.add_argument('--log_dir', default='./log/', type=str)
parser.add_argument('--log_name', default='llama.log', type=str)
parser.add_argument('--output_dir', default='./output/', type=str)
parser.add_argument('--use_moe', action='store_false', help='Use uiLoRA MoE Architecture')
parser.add_argument('--lora_modules', type=int, default=2, help='number of modules for LoRA')
parser.add_argument('--r', type=int, default=4, help='rank for LoRA')
parser.add_argument('--expert_num', type=int, default=4, help='the number of experts')
parser.add_argument('--top_k', type=int, default=2, help='top-k experts')

args = parser.parse_args()

for path in [args.ckpt_dir + args.dataset_name, args.output_dir + args.dataset_name, args.log_dir + args.dataset_name]:
    if not os.path.exists(path):
        os.makedirs(path)

device = f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu"

if 'Yelp' in args.dataset_name:
    user_num, item_num = 27147, 20266
elif 'TripAdvisor' in args.dataset_name:
    user_num, item_num = 9765, 6280
elif 'MoviesAndTV' in args.dataset_name:
    user_num, item_num = 7506, 7360
elif 'ClothingShoesAndJewelry' in args.dataset_name:
    user_num, item_num = 38764, 22919

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# ==========================================
# 2. Data set preprocessing and cache loading
# ==========================================
model_tag = args.model_name.split('/')[-2] if args.model_name.endswith('/') else args.model_name.split('/')[-1]
cache_path = os.path.join(args.data_dir, args.dataset_name, f'dataset_keywords_{model_tag}.pickle')

if not os.path.exists(cache_path):
    dataset = pd.read_pickle(os.path.join(args.data_dir, args.dataset_name, 'reviews.pickle'))
    dataset = pd.DataFrame(dataset)
    
    encoder = LabelEncoder()  
    dataset['user'] = encoder.fit_transform(dataset['user'].tolist()).tolist()
    dataset['item'] = encoder.fit_transform(dataset['item'].tolist()).tolist()

    keywords, keywords_words, text = [], [], []
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    bos_id = tokenizer.bos_token_id
    
    for row in tqdm(dataset['template'], desc="Processing Datasets"):
        kw_tokens = tokenizer(row[0])['input_ids']
        if bos_id is not None and len(kw_tokens) > 0 and kw_tokens[0] == bos_id:
            kw_tokens = kw_tokens[1:]
            
        txt_tokens = tokenizer(row[2])['input_ids']
        if bos_id is not None and len(txt_tokens) > 0 and txt_tokens[0] == bos_id:
            txt_tokens = txt_tokens[1:]
            
        keywords.append(kw_tokens)
        keywords_words.append(row[0])
        text.append(txt_tokens + [eos_id])
        
    dataset['text'] = text
    dataset['keyword'] = keywords
    dataset['keyword_words'] = keywords_words
    dataset = dataset[['user', 'item', 'text', 'keyword', 'keyword_words', 'rating']]
    dataset.to_pickle(cache_path)
else:
    dataset = pd.read_pickle(cache_path)

dataset['rating'] = [int(x-1) for x in dataset['rating'].tolist()]

# Target modules slicing
module_list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
target_modules = module_list[:args.lora_modules]

# ==========================================
# 3. Model initialization and training strategy
# ==========================================
for split_index in ['1']:
    train_dataset, valid_dataset, test_dataset = dataset_split(dataset, split_index, args)
    train_set = MyDataset(train_dataset)
    valid_set = MyDataset(valid_dataset)
    test_set = MyDataset(test_dataset)
    
    collate_train = MyCollater(args.epochs*len(train_dataset)//args.batch_size, args.word, args.delta)
    collate_valid = MyCollater(1, args.word)
    
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_train, shuffle=True, pin_memory=True, num_workers=1)
    valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size, collate_fn=collate_valid, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_valid, shuffle=False)

    # LLaMA / Mistral Base Loading
    model_llm = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    model_llm.gradient_checkpointing_enable()

    if args.use_moe:
        model = uiAdapter(
            user_num=user_num, 
            item_num=item_num, 
            hidden=args.id_hidden, 
            llm_hidden=model_llm.config.hidden_size, 
            tokenizer=tokenizer, 
            model_llm=model_llm, 
            r=args.r, 
            expert_num=args.expert_num, 
            top_k=args.top_k,
            lora_modules=args.lora_modules
        ).to(device)
    else:
        lora_config = LoraConfig(
            r=args.r, lora_alpha=32, target_modules=target_modules,
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
        )
        model_llm = get_peft_model(model_llm, lora_config)
        model_llm.print_trainable_parameters()
        model = MyModel(user_num, item_num, args.id_hidden, model_llm.config.hidden_size, tokenizer).to(device)
        model.model = model_llm
    
    model.generate_weight = args.generate_weight
    model.rating_weight = args.rating_weight

    param = list(filter(lambda p: p.requires_grad==True, model.prompt_encoder.parameters()))
    if args.use_moe:
        param += list(filter(lambda p: p.requires_grad==True, model.f_ui.parameters()))
        
    param2 = filter(lambda p: p.requires_grad==True, model.model.parameters())
    optimizer = AdamW([
         {'params': param, 'lr': args.learning_rate},      # Recommendation and prompt_encoder: 1e-3
         {'params': param2, 'lr': args.learning_rate/10},  # LLM Experts / LoRAs: 1e-4
    ])

    log_name = os.path.join(args.log_dir, args.dataset_name, args.log_name)
    best_loss = 999
    early_stop = 1

    if args.use_moe:
        moe_ckpt_path = os.path.join(args.ckpt_dir, args.dataset_name, f'{split_index}moe_model.pth')
        torch.cuda.empty_cache()
        model.load_state_dict(torch.load(moe_ckpt_path, map_location="cpu"), strict=False)
    else:
        model.model.load_adapter(os.path.join(args.ckpt_dir, args.dataset_name, f'{split_index}model'), 'best_lora')
        model.model.set_adapter("best_lora")
        model.prompt_encoder = torch.load(os.path.join(args.ckpt_dir, args.dataset_name, f'{split_index}ped.bin'), map_location=device, weights_only=False)

# ==========================================
# 4. Correlation & SVD
# ==========================================

def analyze_correlation(model, test_dataloader, device, num_batches=10):
    """
    Modality Analysis (Text vs UI Collaborative) for uiAdapter-based models
    """
    model.eval()
    metrics = {
        'txt_ui': {'cos': [], 'pearson': []}
    }

    # Cosine Correlation
    def get_cos_sim(v1, v2):
        v1 = v1.flatten().to(torch.float32)
        v2 = v2.flatten().to(torch.float32)
        return torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()

    # Pearson Correlation Coefficient
    def get_pearson(v1, v2):
        v1_np = v1.flatten().detach().cpu().to(torch.float32).numpy()
        v2_np = v2.flatten().detach().cpu().to(torch.float32).numpy()
        if np.std(v1_np) == 0 or np.std(v2_np) == 0:
            return 0.0
        corr, _ = pearsonr(v1_np, v2_np)
        return corr if not np.isnan(corr) else 0.0

    print("\n" + "="*60)
    print("Start Modality Analysis (Text vs UI Collaborative)...")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx >= num_batches:
                break
            
            input_ids = batch[0].to(device)
            userid = batch[1].to(device)
            itemid = batch[2].to(device)
            curr_flag = batch[4].to(device)
            rating_inputs = batch[5].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model.get_embedding(
                    input_ids=input_ids, 
                    user_id=userid, 
                    item_id=itemid, 
                    rating=rating_inputs, 
                    curr_flag=curr_flag
                )

            target_layer = None
            for m in model.modules():
                if m.__class__.__name__ == 'SparseLoraLayer' and getattr(m, 'x_ui', None) is not None:
                    target_layer = m
                    break
            
            if target_layer is None:
                continue

            # extract textual Embedding
            if hasattr(model.model, "get_input_embeddings"):
                embeddings_layer = model.model.get_input_embeddings()
            else:
                embeddings_layer = model.model.base_model.get_input_embeddings()
            
            token_embeddings = embeddings_layer(input_ids) # [Batch, Seq_len, Dim]
            if pad_id is not None:
                mask = (input_ids != pad_id).unsqueeze(-1).to(token_embeddings.dtype) # [Batch, Seq_len, 1]
                x_txt = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                x_txt = token_embeddings.mean(dim=1)
            
            x_ui = target_layer.x_ui

            metrics['txt_ui']['cos'].append(get_cos_sim(x_txt, x_ui))
            metrics['txt_ui']['pearson'].append(get_pearson(x_txt, x_ui))

    print(f"{'Modal Pair':<15} | {'Cosine Sim':<15} | {'Pearson Corr':<15}")
    print("-" * 50)
    for key, val in metrics.items():
        if val['cos']:
            avg_cos = np.mean(val['cos'])
            avg_pea = np.mean(val['pearson'])
            print(f"{key:<15} | {avg_cos:^15.4f} | {avg_pea:^15.4f}")
    print("="*60)

    return metrics


def compute_delta_w(lora_A: torch.Tensor, lora_B: torch.Tensor, scaling: float) -> torch.Tensor:
    """"Delta W = B @ A * scaling"""
    lora_A = lora_A.to(torch.float32)
    lora_B = lora_B.to(torch.float32)
    return (lora_B @ lora_A) * scaling


def analyze_svd_of_lora_weights(model: nn.Module, num_singular_values: int = 100, lora_id: int = 0, output_dir: str = './analysis_results'):
    """
    SVD Analysis for uiAdapter-based Weights
    """
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n" + "="*80)
    print("Start SVD Analysis for uiAdapter-based Model ...")
    
    results = {}
    lora_layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'SparseLoraLayer':
            lora_layers.append((name, module))

    if not lora_layers:
        print("Cannot find SparseLoraLayer in models!")
        return

    # Select SparseLoraLayer for analysis
    layer_name, lora_layer = lora_layers[lora_id]
    scaling = lora_layer.scaling
    expert_number = lora_layer.expert_number
    
    print(f"Selected Layer: {layer_name}")
    print(f"Adapter Rank (r): {lora_layer.lora_A_t.shape[0]} | Expert Number: {expert_number}")

    # 1. Textual Branch: Delta W_t
    delta_w_t = compute_delta_w(lora_layer.lora_A_t.data, lora_layer.lora_B_t.data, scaling)
    matrices_to_analyze = {
        'Text_$\Delta$W': delta_w_t
    }

    # 2. UI Branch: Delta W_expert_i
    avg_delta_w_ui_experts = torch.zeros_like(delta_w_t)
    for idx in range(expert_number):
        expert_A = lora_layer.lora_A_ui_experts[idx].data
        expert_B = lora_layer.lora_B_ui_experts[idx].data
        delta_w_exp = compute_delta_w(expert_A, expert_B, scaling)
        avg_delta_w_ui_experts += delta_w_exp
    
    avg_delta_w_ui_experts = avg_delta_w_ui_experts / expert_number

    delta_w_fused = delta_w_t + avg_delta_w_ui_experts
    group1_matrices = {
        'Text_$\\Delta$W': delta_w_t,
        'UI_$\\Delta$W': avg_delta_w_ui_experts
    }
    
    group2_matrices = {
        'Fused_(Text+UI)_$\\Delta$W': delta_w_fused
    }

    r_val = args.r
    rank_checkpoints_group1 = [r_val, r_val * expert_number]
    labels_chk_group1 = [f'$r_t={r_val}$', f'$E*r_{{ui}}={r_val * expert_number}$']
    rank_checkpoints_group2 = [r_val, r_val * expert_number + r_val]
    labels_chk_group2 = [f'$r_t={r_val}$', f'$E*r_{{ui}}+r_t={r_val * expert_number + r_val}$']
    colors = ['blue', 'red']

    def plot_matrix_group(matrices_dict, filename_suffix, title_suffix, rank_checkpoints, labels_chk):
        plt.figure(figsize=(10, 6))
        
        for label, delta_w in matrices_dict.items():
            delta_w_f32 = delta_w.to(torch.float32)
            s = torch.linalg.svdvals(delta_w_f32).cpu().numpy()
            
            num_to_plot = min(len(s), num_singular_values)
            s_plot = s[:num_to_plot]
            indices = np.arange(1, num_to_plot + 1)
            
            variance_ratio = np.sum(s_plot**2) / np.sum(s**2) if np.sum(s**2) > 0 else 0
            
            results[label] = {
                'total_rank': delta_w.shape[0],
                f'top_{num_to_plot}_variance_ratio': variance_ratio
            }
            
            print(f"\n-> SVD Results ({label})")
            print(f"    Shape: {delta_w.shape[0]}x{delta_w.shape[1]}")
            print(f"    First {num_to_plot} singular value ratio: {variance_ratio * 100:.2f}%")
            
            plt.plot(indices, s_plot, marker='.', linestyle='-', markersize=4, label=label)

        plt.xscale('log')
        plt.yscale('log')
        #plt.title(f'Singular Value Spectrum ({title_suffix}) in {layer_name.split(".")[-1]}', fontsize=12)
        plt.xlabel('Singular Value Index $\\log(k)$', fontsize=10)
        plt.ylabel('Singular Value $\\log(\\sigma_k)$', fontsize=10)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc='best')
        
        ymin, ymax = plt.ylim()
        if ymax <= 0:
            ymax = 1.0
            
        for rc, lbl, col in zip(rank_checkpoints, labels_chk, colors):
            plt.axvline(x=rc, color=col, linestyle='--', alpha=0.6, linewidth=1.5)
            plt.text(rc * 1.05, ymax * 0.2, lbl, color=col, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, filename_suffix)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Figures saved to: {plot_path}")

    # Figure One: Text_ΔW 和 UI_ΔW
    plot_matrix_group(group1_matrices, 'svd_spectrum_text_ui.png', 'Text & UI Branches', rank_checkpoints_group1, labels_chk_group1)

    # Figure Two: Fused_(Text+UI)_ΔW
    plot_matrix_group(group2_matrices, 'svd_spectrum_fused.png', 'Fused (Text+UI)', rank_checkpoints_group2, labels_chk_group2)

    print(f"\n" + "="*80)
    print(f"SVD figures saved to: {output_dir}")
    print("="*80)
    
    return results


# ==========================================
# 5. Analysis Module
# ==========================================
if __name__ == "__main__":
    print("\n=== [Start Analysis of Subspace Decoupling and Spectral Extension for CIER] ===")
    # 1. Correlation analysis
    analyze_correlation(model, test_dataloader, device, num_batches=15)
    # 2. SVD Analysis for Spectral Extension
    analyze_svd_of_lora_weights(model, num_singular_values=100, lora_id=0)