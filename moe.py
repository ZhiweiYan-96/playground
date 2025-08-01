import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, n_embed):
        super(Expert, self).__init__()
        self.linear = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        return self.linear(x)


class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # add noise
        self.noise_linear =nn.Linear(n_embed, num_experts)

    
    def forward(self, mh_output):
        # mh_output [num_token, embed_dim] => [10, 64]
        # logits [num_token, num_experts] => [10, 4]
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        # top_k_logits: [num_token, top_k]
        # indices: [num_token, top_k]
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        # zeros: [num_token, num_experts] => [10, 4]
        zeros = torch.full_like(noisy_logits, float('-inf'))
        # sparse_logits: [num_token, num_experts] => [10, 4]
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        # scatter: zeros[i][indices[i,j]] = top_k_logits[i][j]
        # indices[i,0] => zeros[i][indices[i, 0]] = top_k_logits[i, 0]
        # indices[i,1] => zeros[i][indices[i, 1]] = top_k_logits[i, 1]
        # ...
        # indices[i, top_k-1] => zeros[i][indices[i, top_k-1]] = top_k_logits[i, top_k-1]
        # 将top_k的logits放到对应的indices位置上，其余位置为-inf。类似于mask操作

        # router_output: [num_token, num_experts] => [10, 4]
        # indices: [num_token, top_k] => [10, 2]
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k # 4

    def forward(self, x):
        breakpoint()
        # x [num_token, embed_dim] => [10, 64]
        # 1. 输入进入router得到两个输出
        # gating output: [num_token, num_experts] => [10, 4]
        # indices: [num_token, top_k] => [10, 2]
        # gating_output: 每个token，对每个expert的权重分数，其中低于topk的已经是0了
        # indices: 每个token，对每个expert的前top_k个索引
        gating_output, indices = self.router(x)
        # 2.初始化全零矩阵，后续叠加为最终结果
        final_output = torch.zeros_like(x)

        # 3.展平，即把每个batch拼接到一起，这里对输入x和router后的结果都进行了展平
        # flat_x: [num_token, embed_dim] => [10, 64]
        # flat_gating_output: [num_token, num_experts] => [10, 4]
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # 以每个专家为单位进行操作，即把当前专家处理的所有token都进行加权
        for i, expert in enumerate(self.experts):
            # 4. 对当前的专家(例如专家0)来说，查看其对所有tokens中哪些在前top2
            # expert_mask: [num_token] => [10], 
            expert_mask = (indices == i).any(dim=-1)
            # 5. 展平操作
            # flat_mask: [num_token] => [10]
            flat_mask = expert_mask.view(-1)
            # 如果当前专家是任意一个token的前top2
            if flat_mask.any():
                # 6. 得到该专家对哪几个token起作用后，选取token的维度表示
                # expert_input: [num_token, embed_dim] => [num_token_chosen, embed_dim]
                expert_input = flat_x[flat_mask]
                # 7. 将token输入expert得到输出
                # expert_output: [num_token_chosen, embed_dim] => [num_token_chosen, embed_dim]
                expert_output = expert(expert_input)

                # 8. 计算当前专家对于有作用的token的权重分数
                # gating_scores: [num_token_chosen, 1] => [num_token_chosen, 1]
                # 获取当前专家对有作用的token的权重分数
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                # 9. 将expert输出乘上权重分数
                # [num_token_chosen, embed_dim] * [num_token_chosen, 1] => [num_token_chosen, embed_dim]
                weighted_output = expert_output * gating_scores

                # 10. 循环进行做种的结果叠加
                # 赋值回去
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output




if __name__ == "__main__":

    # Example usage
    n_embed = 64
    num_experts = 4
    top_k = 2
    model = SparseMoE(n_embed, num_experts, top_k)
    input_tensor = torch.randn(10, n_embed)  # Batch of 10 samples
    output = model(input_tensor)
    print(output.shape)  # Should be (10, n_embed)
