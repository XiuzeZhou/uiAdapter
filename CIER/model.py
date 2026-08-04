import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PromptEncoder(nn.Module):
    def __init__(self, user_num, item_num, tokenizer, hidden=1024, output_hidden=4096):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dropout = nn.Dropout(0.1)
        self.user_embedding = nn.Embedding(user_num, hidden)
        self.item_embedding = nn.Embedding(item_num, hidden)
        self.mlp_u = nn.Sequential(
            torch.nn.Linear(hidden, output_hidden)
        )
        self.mlp_v = nn.Sequential(
            torch.nn.Linear(hidden, output_hidden)
        )
        # Replace original self.instruction and self.verbalizer
        def safe_tokenize(text):
            tokens = tokenizer(text)['input_ids']
            # If the tokenizer automatically adds bos or cls, safely remove the first token
            if len(tokens) > 0 and tokens[0] in [tokenizer.bos_token_id, tokenizer.cls_token_id]:
                return tokens[1:]
            return tokens

        self.instruction = torch.tensor([safe_tokenize('Predict the rating for the given user and item, and generate a corresponding explanation or keyword.')])
        self.hard_prompt1 = torch.tensor([safe_tokenize('The rating given by user')])
        self.hard_prompt2 = torch.tensor([safe_tokenize('to item')])
        self.hard_prompt3 = torch.tensor([safe_tokenize('is ')])
        self.hard_prompt4 = torch.tensor([safe_tokenize('and the corresponding')])
        self.hard_prompt5 = torch.tensor([safe_tokenize('is "')])
        
        self.sub_full_words = torch.tensor(safe_tokenize('keyword explanation'))
        
        self.verbalizer = [tokenizer(str(i), add_special_tokens=False)['input_ids'][-1] for i in range(1, 6)]
        self.prompt_length = 4 + self.instruction.shape[1] + self.hard_prompt1.shape[1] + self.hard_prompt2.shape[1] + self.hard_prompt3.shape[1] + self.hard_prompt4.shape[1] + self.hard_prompt5.shape[1]
        self.rating_index = 1 + self.instruction.shape[1] + self.hard_prompt1.shape[1] + self.hard_prompt2.shape[1] + self.hard_prompt3.shape[1]
    
    def forward(self, user_id=None, item_id=None,
                rating=None,embed_tokens=None,curr_flag=None
               ):
        device = user_id.device
        user_embedding = self.mlp_u(self.user_embedding(user_id)).unsqueeze(1)
        item_embedding = self.mlp_v(self.item_embedding(item_id)).unsqueeze(1)
        if rating == None:
            return torch.cat([embed_tokens(self.instruction.to(device)).repeat(user_embedding.shape[0],1,1),
                              embed_tokens(self.hard_prompt1.to(device)).repeat(user_embedding.shape[0],1,1),
                              user_embedding,
                              embed_tokens(self.hard_prompt2.to(device)).repeat(user_embedding.shape[0],1,1),
                              item_embedding,
                              embed_tokens(self.hard_prompt3.to(device)).repeat(user_embedding.shape[0],1,1),
                              ],dim=-2)
        values = embed_tokens(torch.tensor(self.verbalizer).to(device)).unsqueeze(0).repeat(rating.shape[0],1,1)
        values = (rating.unsqueeze(-1) * values).sum(dim=1)
        if curr_flag == None:
            flag_words = self.sub_full_words.to(device)[[1]].repeat(user_embedding.shape[0],1)
        else:
            flag_words = self.sub_full_words.to(device)[curr_flag].unsqueeze(1)
        return torch.cat([embed_tokens(self.instruction.to(device)).repeat(user_embedding.shape[0],1,1),
                          embed_tokens(self.hard_prompt1.to(device)).repeat(user_embedding.shape[0],1,1),
                          user_embedding,
                          embed_tokens(self.hard_prompt2.to(device)).repeat(user_embedding.shape[0],1,1),
                          item_embedding,
                          embed_tokens(self.hard_prompt3.to(device)).repeat(user_embedding.shape[0],1,1),
                          values.unsqueeze(1),
                          embed_tokens(self.hard_prompt4.to(device)).repeat(user_embedding.shape[0],1,1),
                          embed_tokens(flag_words),
                          embed_tokens(self.hard_prompt5.to(device)).repeat(user_embedding.shape[0],1,1),
                          ],dim=-2)

class MyModel(nn.Module):
    def __init__(self, user_num, item_num,  hidden, llm_hidden, tokenizer):
        super(MyModel, self).__init__()
        self.prompt_encoder = PromptEncoder(user_num, item_num, tokenizer, hidden=hidden, output_hidden=llm_hidden)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()
        self.model = None

        self.generate_weight = 1.0
        self.rating_weight= 0.1
        
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def get_embedding(self, input_ids=None,  user_id=None, item_id=None, rating=None,curr_flag=None):
        if hasattr(self.model, "get_input_embeddings"):
            embeddings = self.model.get_input_embeddings()
        else:
            # Compatible with PEFT
            embeddings = self.model.base_model.get_input_embeddings()
        
        if input_ids == None:
            return self.prompt_encoder(user_id=user_id,item_id=item_id,embed_tokens=embeddings)
        prompt = self.prompt_encoder(user_id=user_id,item_id=item_id,
                                     rating=rating,embed_tokens=embeddings,curr_flag=curr_flag
                                    )
        if input_ids.shape[1]==0:
            return prompt
        inputs_embeds = embeddings(input_ids)
        inputs_embeds = torch.cat([prompt,inputs_embeds],dim=-2)
        return inputs_embeds
    
    def forward(self, input_ids=None, user_id=None, item_id=None,
                rating=None, kv_cache=None
               ):
        #enocde
        if kv_cache == None:
            inputs_embeds = self.get_embedding(input_ids=input_ids, user_id=user_id, item_id=item_id, rating=rating)
            output = self.model(inputs_embeds=inputs_embeds)
        else:
            output = self.model(input_ids = input_ids, past_key_values = kv_cache)
        #decode
        kv_cache = output['past_key_values']
        logits = output['logits'][:,-1,:]
        logits = torch.softmax(logits,dim=1)
        
        return logits, kv_cache
    def rating_predict(self, user_id=None, item_id=None):
        
        inputs_embeds = self.get_embedding(user_id=user_id, item_id=item_id)
        logits = self.model(inputs_embeds=inputs_embeds)['logits']
        #decode
        output = logits[:,self.prompt_encoder.rating_index,:]
        output = output[:,self.prompt_encoder.verbalizer]
        output = torch.softmax(output,dim=1)
        
        return output
    def train_step(self, input_ids, user_id=None, item_id=None, rating=None, curr_flag=None,
                   rating_input=None
                  ):
        #enocde
        inputs_embeds = self.get_embedding(input_ids=input_ids, user_id=user_id, item_id=item_id,
                                              rating=rating_input,curr_flag=curr_flag
                                             )
        logits = self.model(inputs_embeds=inputs_embeds)['logits']
        output = logits[:,self.prompt_encoder.rating_index,:]
        output = output[:,self.prompt_encoder.verbalizer]
        loss = F.cross_entropy(output,rating)*self.rating_weight
        
        #MLM
        logits = logits[:,self.prompt_encoder.prompt_length-1:-1,:]
        targets = input_ids
        y_mask = input_ids.clone()
        y_mask[targets!=0] = 1
        y_mask = y_mask.reshape(-1)
        targets = targets.reshape(-1)
        logits = logits.reshape(-1,logits.shape[-1])
        generate_loss = (self.ce_loss(logits,targets) * y_mask).sum(dim=0) / (y_mask.sum(dim=0))
        loss += self.generate_weight*generate_loss

            
        return loss


# =========================================================================
# uiAdapter MoE Framework
# =========================================================================
class SparseLoraLayer(nn.Module):
    def __init__(self, base_layer, hidden_size=4096, dtype=torch.bfloat16, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        r = kwargs.pop("r", 8)
        lora_alpha = kwargs.pop("lora_alpha", 32)
        lora_dropout = kwargs.pop("lora_dropout", 0.05)
        self.expert_number = kwargs.pop("expert_number", 4)
        self.top_k = kwargs.pop("top_k", 2)
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # 1. Textual Branch
        self.lora_A_t = nn.Parameter(torch.empty(r, in_features, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A_t, a=math.sqrt(5))
        self.lora_B_t = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))
        
        # 2. User-Item Branch
        self.lora_A_ui_experts = nn.ParameterList([nn.Parameter(torch.empty(r, in_features, dtype=dtype)) for _ in range(self.expert_number)])
        for a_exp in self.lora_A_ui_experts:
            nn.init.kaiming_uniform_(a_exp, a=math.sqrt(5))
            
        self.lora_B_ui_experts = nn.ParameterList([nn.Parameter(torch.zeros(out_features, r, dtype=dtype)) for _ in range(self.expert_number)])
        
        # 3. Gating Network of MoE
        self.gate = nn.Linear(in_features, self.expert_number, bias=False, dtype=dtype)
        nn.init.normal_(self.gate.weight, std=0.01)
        
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.x_ui = None
        self.current_routing_probs = None

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        lora_t = self.lora_dropout(x) @ self.lora_A_t.transpose(0, 1).to(x.dtype) @ self.lora_B_t.transpose(0, 1).to(x.dtype) * self.scaling

        lora_ui = 0
        if self.x_ui is not None and self.base_layer.in_features == self.hidden_size:
            current_x_ui = self.x_ui if self.x_ui.size(0) == batch_size else self.x_ui[:batch_size]
            
            router_logits = self.gate(current_x_ui.to(x.dtype))
            routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_probs = torch.clamp(routing_probs, min=1e-6, max=1.0)
            self.current_routing_probs = routing_probs
            
            router_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
            router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
            router_weights = router_weights.to(x.dtype)
            expert_mask = F.one_hot(selected_experts, num_classes=self.expert_number).permute(2, 1, 0)
            
            final_lora_ui = torch.zeros((batch_size, 1, self.base_layer.out_features), dtype=x.dtype, device=x.device)
            for expert_idx in range(self.expert_number):
                k_idx, b_idx = torch.where(expert_mask[expert_idx])
                if b_idx.numel() == 0: continue
                current_state = current_x_ui[b_idx].unsqueeze(1)
                
                exp_out = current_state @ self.lora_A_ui_experts[expert_idx].transpose(0, 1).to(x.dtype) \
                                        @ self.lora_B_ui_experts[expert_idx].transpose(0, 1).to(x.dtype) * self.scaling
                
                w = router_weights[b_idx, k_idx].unsqueeze(-1).unsqueeze(-1)
                final_lora_ui.index_add_(0, b_idx, (exp_out * w).to(x.dtype))
                
            lora_ui = final_lora_ui

        result = self.base_layer(x) + lora_t + lora_ui
        return result


class uiAdapter(MyModel):
    def __init__(self, user_num, item_num, hidden, llm_hidden, tokenizer, model_llm, r=8, lora_alpha=32, expert_num=4, top_k=2, lora_modules=2, bal_reg=0.1):
        super().__init__(user_num, item_num, hidden, llm_hidden, tokenizer)
        self.model = model_llm  # base LLM
        self.bal_reg = bal_reg
        
        self.f_ui = nn.Linear(hidden * 2, llm_hidden, dtype=model_llm.dtype)
        
        # Dynamically set target_modules based on model type
        module_list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules = module_list[:lora_modules]  # ["q_proj", "k_proj"] #["q_proj", "v_proj", "k_proj", "o_proj"]
        for param in self.model.parameters():
            param.requires_grad = False
            
        for name, module in self.model.named_modules():
            if any(t_name in name for t_name in target_modules) and isinstance(module, nn.Linear):
                new_layer = SparseLoraLayer(
                    base_layer=module, dtype=model_llm.dtype, r=r, lora_alpha=lora_alpha, 
                    hidden_size=llm_hidden, expert_number=expert_num, top_k=top_k
                )
                parts = name.rsplit('.', 1)
                parent = self.model.get_submodule(parts[0]) if len(parts) > 1 else self.model
                setattr(parent, parts[-1], new_layer)

    def get_embedding(self, input_ids=None, user_id=None, item_id=None, rating=None, curr_flag=None):
        u_emb = self.prompt_encoder.user_embedding(user_id)
        i_emb = self.prompt_encoder.item_embedding(item_id)
        x_ui = self.f_ui(torch.cat([u_emb, i_emb], dim=-1))
        
        for mod in self.modules():
            if isinstance(mod, SparseLoraLayer):
                mod.x_ui = x_ui
        
        return super().get_embedding(input_ids, user_id, item_id, rating, curr_flag)

    def get_load_balancing_loss(self):
        total_bal_loss, count = 0.0, 0
        for module in self.modules():
            if isinstance(module, SparseLoraLayer) and module.current_routing_probs is not None:
                probs = module.current_routing_probs 
                num_experts = probs.size(1)
                _, topk_indices = torch.topk(probs, module.top_k, dim=-1)
                expert_counts = F.one_hot(topk_indices, num_classes=num_experts).sum(dim=1) 
                f_m = expert_counts.float().mean(dim=0) 
                P_m = probs.mean(dim=0) 
                total_bal_loss += num_experts * torch.sum(f_m * P_m)
                count += 1
        return total_bal_loss / count if count > 0 else torch.tensor(0.0, device=self.f_ui.weight.device)

    # Rewrite the train_step function, integrating the MoE load balancing loss on the official Loss.
    def train_step(self, input_ids, user_id=None, item_id=None, rating=None, curr_flag=None, rating_input=None):
        loss = super().train_step(input_ids, user_id, item_id, rating, curr_flag, rating_input)
        bal_loss = self.get_load_balancing_loss()
        return loss + self.bal_reg * bal_loss