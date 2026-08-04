from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import math

class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()

    def forward(self, user, item):  # (batch_size, emsize)
        rating = torch.sum(user * item, 1)  # (batch_size,)
        return rating


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class NeuMF_Predictor(nn.Module):
    def __init__(self, emsize, hidden_size=400, num_layers=0, dropout=0.3):
        super(NeuMF_Predictor, self).__init__()
        self.first_layer = nn.Linear(emsize * 2, hidden_size, dtype=torch.float32)
        self.last_layer = nn.Linear(hidden_size, 1, dtype=torch.float32)
        
        layer = nn.Linear(hidden_size, hidden_size, dtype=torch.float32)
        self.layers = _get_clones(layer, num_layers)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.mf_weight = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        self.mlp_weight = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.first_layer.weight)
        if self.first_layer.bias is not None:
            self.first_layer.bias.data.zero_()
            
        nn.init.normal_(self.last_layer.weight, mean=0.0, std=0.01)
        if self.last_layer.bias is not None:
            self.last_layer.bias.data.zero_()
            
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.zero_()

    def forward(self, p_u, q_i):  
        p_u = p_u.to(torch.float32)
        q_i = q_i.to(torch.float32)
        
        # 1. MF Branch
        mf_output = torch.sum(p_u * q_i, dim=1)
        
        # 2. MLP Branch
        ui_cat = torch.cat([p_u, q_i], dim=1)
        ui_cat = self.dropout(ui_cat)
        hidden = self.sigmoid(self.first_layer(ui_cat))  
        hidden = self.dropout(hidden)
        
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))  
            hidden = self.dropout(hidden)
            
        mlp_output = torch.squeeze(self.last_layer(hidden))  
        
        rating = self.mf_weight * mf_output + self.mlp_weight * mlp_output
        
        return self.sigmoid(rating)


class MLP(nn.Module):
    def __init__(self, emsize, hidden_size=400, num_layers=0, is_sigmoid=False, dtype=torch.bfloat16):
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(emsize, hidden_size, dtype=dtype)
        self.last_layer = nn.Linear(hidden_size, 1, dtype=dtype)
        layer = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.layers = _get_clones(layer, num_layers)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.is_sigmoid = is_sigmoid

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.first_layer.weight, mode='fan_in', nonlinearity='relu')
        if self.first_layer.bias is not None:
            self.first_layer.bias.data.zero_()
            
        nn.init.normal_(self.last_layer.weight, mean=0.0, std=0.001)
        if self.last_layer.bias is not None:
            self.last_layer.bias.data.zero_()
            
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='sigmoid')
            if layer.bias is not None:
                layer.bias.data.zero_()

    def forward(self, x):  
        hidden = self.relu(self.first_layer(x))  
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))  
        rating = torch.squeeze(self.last_layer(hidden))  
        if self.is_sigmoid:
            rating = self.sigmoid(rating)
        return rating


class LoraLayer(nn.Module):
    def __init__(self, base_layer, hidden_size=4096, dtype=torch.bfloat16, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.base_layer = base_layer

        for param in self.base_layer.parameters():
            param.requires_grad = False

        r = kwargs.pop("r", 8)
        lora_alpha = kwargs.pop("lora_alpha", 16)
        lora_dropout = kwargs.pop("lora_dropout", 0.0)
        
        # text LoRA (A_t, B_t)
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A_t = nn.Parameter(torch.randn(r, in_features, dtype=dtype))
        self.lora_B_t = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout)
    
    def forward(self, x):
        # y = x_t * W0 + x_t * B_t * A_t
        lora_t = self.lora_dropout(x) @ self.lora_A_t.transpose(0, 1) @ self.lora_B_t.transpose(0, 1) * self.scaling
        result = self.base_layer(x) + lora_t
        return result
    

class SparseLoraLayer(nn.Module):
    def __init__(self, base_layer, hidden_size=4096, dtype=torch.bfloat16, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.base_layer = base_layer

        for param in self.base_layer.parameters():
            param.requires_grad = False

        r = kwargs.pop("r", 8)
        lora_alpha = kwargs.pop("lora_alpha", 16)
        lora_dropout = kwargs.pop("lora_dropout", 0.0)
        
        # --- MoE Settings ---
        self.expert_number = kwargs.pop("expert_number", 4)  # Expert Number
        self.top_k = kwargs.pop("top_k", 2)                  # Activated Experts
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # 1. Textual Branch
        self.lora_A_t = nn.Parameter(torch.randn(r, in_features, dtype=dtype))
        self.lora_B_t = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))
        
        # 2. User-Item Branch
        self.lora_A_ui_experts = nn.ParameterList([
            nn.Parameter(torch.randn(r, in_features, dtype=dtype)) for _ in range(self.expert_number)
        ])
        self.lora_B_ui_experts = nn.ParameterList([
            nn.Parameter(torch.zeros(out_features, r, dtype=dtype)) for _ in range(self.expert_number)
        ])
        
        # 3. Gating Network of MoE
        self.gate = nn.Linear(in_features, self.expert_number, bias=False, dtype=dtype)
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.x_ui = None
        self.current_routing_probs = None

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Textual Branch
        lora_t = self.lora_dropout(x) @ self.lora_A_t.transpose(0, 1).to(x.dtype) @ self.lora_B_t.transpose(0, 1).to(x.dtype) * self.scaling

        lora_ui = 0
        if self.x_ui is not None and self.base_layer.in_features == self.hidden_size:
            # self.x_ui Shape: (batch_size, hidden_dim)
            current_x_ui = self.x_ui if self.x_ui.size(0) == batch_size else self.x_ui[:batch_size]
            # 1. MoE router
            router_logits = self.gate(current_x_ui.to(x.dtype))
            routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_probs = torch.clamp(routing_probs, min=1e-6, max=1.0)
            self.current_routing_probs = routing_probs
            
            # Select Top-K Experts
            router_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
            router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
            router_weights = router_weights.to(x.dtype)
            
            # Generate Expert Mask: (batch_size, top_k, expert_number)
            expert_mask = F.one_hot(selected_experts, num_classes=self.expert_number)
            # Reshape to: (expert_number, top_k, batch_size)
            expert_mask = expert_mask.permute(2, 1, 0)
            
            final_lora_ui = torch.zeros((batch_size, 1, self.base_layer.out_features), dtype=x.dtype, device=x.device)
            
            # 2. Following the sparse allocation logic of Mistral MoE
            for expert_idx in range(self.expert_number):
                # expert_mask[expert_idx] shape: (top_k, batch_size)
                k_idx, b_idx = torch.where(expert_mask[expert_idx])
                
                if b_idx.numel() == 0:
                    continue  # The current expert was not selected by any sample
                
                current_state = current_x_ui[b_idx].unsqueeze(1)  # Shape: (selected_batch_size, seq_len, in_features)
                # exp_out shape: (selected_batch_size, seq_len, out_features)
                exp_out = current_state @ self.lora_A_ui_experts[expert_idx].transpose(0, 1).to(x.dtype) \
                                        @ self.lora_B_ui_experts[expert_idx].transpose(0, 1).to(x.dtype) * self.scaling
                
                # router_weights[b_idx, k_idx] -> reshape to (selected_batch_size, 1, 1) for Broadcasting
                w = router_weights[b_idx, k_idx].unsqueeze(-1).unsqueeze(-1)
                current_expert_output = (exp_out * w).to(x.dtype)
                final_lora_ui.index_add_(0, b_idx, current_expert_output)
                
            lora_ui = final_lora_ui

        result = self.base_layer(x) + lora_t + lora_ui
        return result


class uiAdapter(nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, k, r, mlp_size, lora_modules, expert_number, top_k, dtype=torch.bfloat16, **kwargs):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            quantization_config=quantization_config,
            torch_dtype=dtype, 
            **kwargs
        )
        #base_model.gradient_checkpointing_enable()
        return cls(base_model, nuser, nitem, k, r, mlp_size, lora_modules, expert_number, top_k, dtype, pretrained_model_name_or_path)
    
    def __init__(self, base_model, nuser, nitem, k, r, mlp_size, lora_modules, expert_number, top_k, dtype, pretrained_model_name_or_path):
        super().__init__()
        self.model = base_model
        self.dtype = dtype
        
        self.user_emb = nn.Embedding(nuser, k, dtype=dtype)
        self.item_emb = nn.Embedding(nitem, k, dtype=dtype)
        emsize = self.model.config.hidden_size
        self.r = r
        self.k = k
        self.expert_number = expert_number
        self.top_k = top_k
        self.f_r = NeuMF_Predictor(emsize=k, hidden_size=mlp_size)  
        self.f_user = nn.Linear(k, k, dtype=dtype)
        self.f_item = nn.Linear(k, k, dtype=dtype)
        self.f_ui = nn.Linear(k * 2, emsize, dtype=dtype)

        # Initialize user/item embeddings
        initrange = 0.1
        self.user_emb.weight.data.uniform_(-initrange, initrange)
        self.item_emb.weight.data.uniform_(-initrange, initrange)
       
        # Dynamically set target_modules based on model type
        model_name_lower = pretrained_model_name_or_path.lower()
        if 'llama' in model_name_lower or 'qwen' in model_name_lower or 'mistral' in model_name_lower or 'gemma' in model_name_lower:
            module_list = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            module_list = ["q_proj", "v_proj", "k_proj", "o_proj", "c_attn", "c_proj"]

        if lora_modules < len(module_list):
            target_modules = module_list[:lora_modules]
        else:
            target_modules = module_list

        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        for name, module in self.model.named_modules():
            is_target_layer = any(t_name in name for t_name in target_modules)
            is_valid_class = isinstance(module, nn.Linear) or module.__class__.__name__ in ['Conv1D', 'Linear']
            
            if is_target_layer and is_valid_class:
                base_layer = module
                
                for param in module.parameters():
                    param.requires_grad = False
                
                in_f = getattr(base_layer, "in_features", getattr(base_layer, "nx", None))
                out_f = getattr(base_layer, "out_features", getattr(base_layer, "nf", None))
                
                if not hasattr(base_layer, "in_features"):
                    base_layer.in_features = in_f
                if not hasattr(base_layer, "out_features"):
                    base_layer.out_features = out_f
                
                # build Sparse MoE layer
                new_layer = SparseLoraLayer(
                    base_layer=base_layer,
                    dtype=self.dtype,
                    r=r,
                    lora_alpha=r, 
                    lora_dropout=0.1, 
                    k=k, 
                    hidden_size=self.model.config.hidden_size,
                    expert_number=self.expert_number,
                    top_k=self.top_k
                )
                
                parts = name.rsplit('.', 1)
                if len(parts) == 1:
                    parent_module = self.model
                    child_name = parts[0]
                else:
                    parent_name, child_name = parts
                    try:
                        parent_module = self.model.get_submodule(parent_name)
                    except AttributeError:
                        parent_module = self.model.base_model.get_submodule(parent_name)

                setattr(parent_module, child_name, new_layer)

        # === 4. Activate Trainable Parameters ===
        for param in self.user_emb.parameters():
            param.requires_grad = True
        for param in self.item_emb.parameters():
            param.requires_grad = True
        for param in self.f_user.parameters(): 
            param.requires_grad = True
        for param in self.f_item.parameters(): 
            param.requires_grad = True
        for param in self.f_ui.parameters(): 
            param.requires_grad = True
        for param in self.f_r.parameters(): 
            param.requires_grad = True

        total_trainable_params = 0
        total_all_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_all_params += num_params
            if param.requires_grad:
                total_trainable_params += num_params

        trainable_ratio = (total_trainable_params / total_all_params) * 100 if total_all_params > 0 else 0
        
        print(f"\n--- Trainable Parameters Summary ({'LLM base'}) ---")
        print(f"Total parameters: {total_all_params:,}")
        print(f"Trainable parameters (LoRA + Embeddings): {total_trainable_params:,}")
        print(f"Trainable ratio: {trainable_ratio:.2f}%")
        print(f"------------------------------------")

    # Add method to propagate Q_ui to all custom LoRA layers
    def set_Q(self, x_ui):
        for module in self.modules():
            if isinstance(module, SparseLoraLayer):
                module.x_ui = x_ui

    #  Wrapper methods for base model functionalities
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens):
        self.model.resize_token_embeddings(new_num_tokens)
    
    def generate(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        item = kwargs.pop('item', None)
        
        if user is not None and item is not None:
            user_emb, item_emb = self.user_emb(user), self.item_emb(item)
            Q_ui = torch.cat([user_emb, item_emb], dim=-1)
            x_ui = self.f_ui(Q_ui)
            self.set_Q(x_ui)
            
        return self.model.generate(*args, **kwargs)

    def get_load_balancing_loss(self):
        """
        Find the MySparseLoraLayer in the model, and calculate their load balancing losses
        """
        total_bal_loss = 0.0
        count = 0
        for module in self.modules():
            if isinstance(module, SparseLoraLayer) and module.current_routing_probs is not None:
                probs = module.current_routing_probs  # shape: (batch_size, expert_number)
                batch_size, num_experts = probs.size()
                
                # 1. Calculate the actual allocation proportion f_m for each expert to be selected as Top-K
                _, topk_indices = torch.topk(probs, module.top_k, dim=-1)
                expert_counts = F.one_hot(topk_indices, num_classes=num_experts).sum(dim=1) # (batch_size, num_experts)
                f_m = expert_counts.float().mean(dim=0) # (num_experts,)
                
                # 2. Calculate the average routing probability P_m for each expert
                P_m = probs.mean(dim=0) # (num_experts,)
                
                # 3. Load Balancing Regularization: E * sum(f_m * P_m)
                bal_loss = num_experts * torch.sum(f_m * P_m)
                total_bal_loss += bal_loss
                count += 1
                
        return total_bal_loss / count if count > 0 else torch.tensor(0.0, device=self.user_emb.weight.device)
    
    def forward(self, input_ids, attention_mask=None, user=None, item=None, text_lens=None, **kwargs):
        device = input_ids.device
        user_emb, item_emb = self.user_emb(user), self.item_emb(item)
        Q_ui = torch.cat([user_emb, item_emb], dim=-1)
        x_ui = self.f_ui(Q_ui)
        self.set_Q(x_ui)

        labels = torch.full_like(input_ids, -100, dtype=torch.int64).to(device)
        if text_lens is not None:
            for i, t_len in enumerate(text_lens):
                labels[i, -t_len:] = input_ids[i, -t_len:]  
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        predicted_rating = self.predict_rating(user, item)
        return outputs, predicted_rating
        
    def predict_rating(self, user, item, llm_state=None):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        p_u = self.f_user(user_emb)
        q_i = self.f_item(item_emb)
        
        predicted_rating = self.f_r(p_u, q_i)
        return predicted_rating


class Concat_LoRA(nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, k, r, mlp_size, lora_modules, dtype=torch.bfloat16, **kwargs):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            quantization_config=quantization_config,
            torch_dtype=dtype, 
            **kwargs
        )
        #base_model.gradient_checkpointing_enable()
        return cls(base_model, nuser, nitem, k, r, mlp_size, lora_modules, dtype, pretrained_model_name_or_path)

    def __init__(self, base_model, nuser, nitem, k, r, mlp_size, lora_modules, dtype, pretrained_model_name_or_path):
        super().__init__()
        self.model = base_model
        self.dtype = dtype
        self.r = r
        self.k = k
        
        # Recommendation Embeddings
        self.user_emb = nn.Embedding(nuser, k, dtype=dtype)
        self.item_emb = nn.Embedding(nitem, k, dtype=dtype)
        self.hidden_size = self.model.config.hidden_size
        
        # Map lays for recommendation embeddings
        self.user_projector = nn.Linear(k, self.hidden_size, dtype=dtype)
        self.item_projector = nn.Linear(k, self.hidden_size, dtype=dtype)
        
        # Rating prediction for user/item embeddings
        self.f_r = NeuMF_Predictor(emsize=k, hidden_size=mlp_size)
        self.f_user = nn.Linear(k, k, dtype=dtype)
        self.f_item = nn.Linear(k, k, dtype=dtype)

        # Initialize weights
        initrange = 0.1
        self.user_emb.weight.data.uniform_(-initrange, initrange)
        self.item_emb.weight.data.uniform_(-initrange, initrange)

        # Trainable Layers of adapter
        model_name_lower = pretrained_model_name_or_path.lower()
        if 'llama' in model_name_lower or 'qwen' in model_name_lower or 'mistral' in model_name_lower or 'gemma' in model_name_lower:
            module_list = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            module_list = ["q_proj", "v_proj", "k_proj", "o_proj", "c_attn", "c_proj"]

        target_modules = module_list[:lora_modules] if lora_modules < len(module_list) else module_list

        # Fraze LLM weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Build LoRA (LoraLayer)
        for name, module in self.model.named_modules():
            is_target_layer = any(t_name in name for t_name in target_modules)
            is_valid_class = isinstance(module, nn.Linear) or module.__class__.__name__ in ['Conv1D', 'Linear']
            
            if is_target_layer and is_valid_class:
                base_layer = module
                
                in_f = getattr(base_layer, "in_features", getattr(base_layer, "nx", None))
                out_f = getattr(base_layer, "out_features", getattr(base_layer, "nf", None))
                if not hasattr(base_layer, "in_features"): base_layer.in_features = in_f
                if not hasattr(base_layer, "out_features"): base_layer.out_features = out_f

                new_layer = LoraLayer(
                    base_layer=base_layer,
                    dtype=self.dtype,
                    r=r,
                    lora_alpha=r,
                    lora_dropout=0.1,
                    hidden_size=self.hidden_size
                )
                
                parts = name.rsplit('.', 1)
                parent_module = self.model.get_submodule(parts[0]) if len(parts) > 1 else self.model
                setattr(parent_module, parts[-1], new_layer)

        # Activate trainable parameters
        for param in self.user_emb.parameters(): param.requires_grad = True
        for param in self.item_emb.parameters(): param.requires_grad = True
        for param in self.user_projector.parameters(): param.requires_grad = True
        for param in self.item_projector.parameters(): param.requires_grad = True
        for param in self.f_user.parameters(): param.requires_grad = True
        for param in self.f_item.parameters(): param.requires_grad = True
        for param in self.f_r.parameters(): param.requires_grad = True

        total_trainable_params = 0
        total_all_params = 0

        for name, param in self.named_parameters():
            num_params = param.numel()
            total_all_params += num_params
            if param.requires_grad:
                total_trainable_params += num_params

        trainable_ratio = (total_trainable_params / total_all_params) * 100 if total_all_params > 0 else 0
        
        print(f"\n--- Trainable Parameters Summary ({'LLM base'}) ---")
        print(f"Total parameters: {total_all_params:,}")
        print(f"Trainable parameters (LoRA + Embeddings): {total_trainable_params:,}")
        print(f"Trainable ratio: {trainable_ratio:.2f}%")
        print(f"------------------------------------")

    def resize_token_embeddings(self, new_num_tokens):
        self.model.resize_token_embeddings(new_num_tokens)
    
    def _get_concat_embeddings(self, user, item, input_ids, attention_mask):
        """
        Left Padding Batchify, adding [User_Token, Item_Token] inserted after [Pad] and before the valid text:
        [Pad...Pad] + [u] + [i] + [Prompt] + <bos> + ...
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        hidden_size = self.hidden_size

        # 1. Extract Embeddings (B, S, H)
        text_tokens = self.model.get_input_embeddings()(input_ids.to(device))

        # 2. User-item tokens: (B, 2, H)
        u_emb = self.user_projector(self.user_emb(user)).unsqueeze(1)  
        i_emb = self.item_projector(self.item_emb(item)).unsqueeze(1)  
        ui_tokens = torch.cat([u_emb, i_emb], dim=1)                 

        # 3. Reconstruct Embedding and mask: (B, 2 + S, H)
        full_embeds = torch.zeros((batch_size, 2 + seq_len, hidden_size), dtype=self.dtype, device=device)
        full_mask = torch.zeros((batch_size, 2 + seq_len), dtype=attention_mask.dtype, device=device)
        for idx in range(batch_size):
            # locate the position of the first 1 in the mask: the exact number of pads on the left side.
            ones_indices = (attention_mask[idx] == 1).nonzero(as_tuple=True)[0]
            num_pads = int(ones_indices[0].item()) if len(ones_indices) > 0 else 0

            # keep padding
            if num_pads > 0:
                full_embeds[idx, :num_pads] = text_tokens[idx, :num_pads]
                
            # insert [User, Item]
            full_embeds[idx, num_pads : num_pads + 2] = ui_tokens[idx]
            full_mask[idx, num_pads : num_pads + 2] = 1
            
            # move valid texts: (Prompt + Review or Prompt)
            full_embeds[idx, num_pads + 2 :] = text_tokens[idx, num_pads:]
            full_mask[idx, num_pads + 2 :] = attention_mask[idx, num_pads:]

        return full_embeds, full_mask

    def forward(self, input_ids, attention_mask=None, user=None, item=None, text_lens=None, **kwargs):
        device = input_ids.device
        
        full_embeds, full_mask = self._get_concat_embeddings(user, item, input_ids, attention_mask)
        labels = torch.full((full_embeds.shape[0], full_embeds.shape[1]), -100, dtype=torch.int64, device=device)
        if text_lens is not None:
            for i, t_len in enumerate(text_lens):
                start_idx_in_input = input_ids.shape[1] - t_len
                start_idx_in_labels = start_idx_in_input + 2
                labels[i, start_idx_in_labels:] = input_ids[i, start_idx_in_input:]

        outputs = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        predicted_rating = self.predict_rating(user, item)
        return outputs, predicted_rating

    def generate(self, *args, **kwargs):
        input_ids = None
        attention_mask = None
        
        args_list = list(args)
        if len(args_list) > 0:
            input_ids = args_list.pop(0)
        if len(args_list) > 0:
            attention_mask = args_list.pop(0)
        
        if input_ids is None: input_ids = kwargs.pop('input_ids', None)
        if attention_mask is None: attention_mask = kwargs.pop('attention_mask', None)
        user = kwargs.pop('user', None)
        item = kwargs.pop('item', None)

        device = self.model.device
        batch_size, seq_len = input_ids.size()

        # 2. Left Padding (B, S, H)
        # w_src shape: [Pad, Pad, ..., Prompt, <bos>]
        w_src = self.model.get_input_embeddings()(input_ids.to(device))    

        # 3. user/item tokens: (B, 2, H)
        u_emb = self.user_projector(self.user_emb(user)).unsqueeze(1)  
        i_emb = self.item_projector(self.item_emb(item)).unsqueeze(1)  
        ui_tokens = torch.cat([u_emb, i_emb], dim=1)                 

        # 4. Reconstruct Embedding and mask: (B, 2 + S, H)
        full_embeds = torch.zeros((batch_size, 2 + seq_len, self.hidden_size), dtype=self.dtype, device=device)
        full_mask = torch.zeros((batch_size, 2 + seq_len), dtype=attention_mask.dtype, device=device)
        for i in range(batch_size):
            # locate the position of Left Padding
            ones_indices = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
            num_pads = int(ones_indices[0].item()) if len(ones_indices) > 0 else 0
            
            # a. keep [Pad] Slice
            if num_pads > 0:
                full_embeds[i, :num_pads] = w_src[i, :num_pads] # full_mask[i, :num_pads] = 0
                
            # b. insert [User_Token, Item_Token]
            full_embeds[i, num_pads : num_pads + 2] = ui_tokens[i]
            full_mask[i, num_pads : num_pads + 2] = 1
            
            # c. Keep [Prompt, <bos>]
            full_embeds[i, num_pads + 2 :] = w_src[i, num_pads:]
            full_mask[i, num_pads + 2 :] = attention_mask[i, num_pads:]

        kwargs['inputs_embeds'] = full_embeds
        kwargs['attention_mask'] = full_mask
        
        return self.model.generate(*tuple(args_list), **kwargs)

    def predict_rating(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        p_u = self.f_user(user_emb)
        q_i = self.f_item(item_emb)
        rating = self.f_r(p_u, q_i)
        return rating