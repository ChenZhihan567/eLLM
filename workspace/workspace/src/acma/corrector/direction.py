"""Compute the VA correction direction for posterior adjustment (e.g., neutral(中立), literal(字面), sarcastic(讽刺等)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from acma.config import get_default_config


cfg = get_default_config()


class Expert(nn.Module):
    """The experts are used as different “decision modules” or “perspectives” to handle cases where multimodal information (text + audio/visual) is inconsistent or ambiguous. For example:

        Expert 1 (Neutral/Abstain): When there is no clear conflict or modification needed, it keeps the original text with minimal edits.

        Expert 2 (Literal): When the text and audio-visual cues are consistent, it makes minor literal adjustments for precision.

        Expert 3 (Sarcasm): When the text and audio-visual cues conflict (e.g., irony, sarcasm), it adjusts or even reverses polarity so that the output better reflects the intended meaning.

    Attributes:
        self.fc(torch.nn.modules.linear.Linear): Fully connected layer
    """

    
    def __init__(self, in_features):

        super(Expert, self).__init__()
        hidden_dim = int(in_features * cfg.corrector.hidden_adjuster)
        self.fc1 = nn.Linear(in_features, hidden_dim).to("cuda")
        self.fc2 = nn.Linear(hidden_dim, 1).to("cuda")

    
    def forward(self, x):
        print(type(self.fc1))
        return F.tanh(self.fc2(F.gelu(self.fc1(x))))



class MoERouter(nn.Module):
    def __init__(self, num_experts, in_features, hidden_dim):
        super(MoERouter, self).__init__()
        self.num_experts = num_experts
        self.feature = nn.Linear(in_features, hidden_dim).to("cuda")
        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_dim, num_experts).to("cuda")



    def forward(self, x, tau=1.0):
        x = self.feature(x)  # (batch_size, hidden_dim)
        gate_score = F.softmax(self.gate(x)/tau, dim=-1)  # (batch_size, num_experts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch_size, num_experts, 1)
        # gate_score.unsqueeze(1):(batch_size, 1, num_experts)
        delta = torch.bmm(gate_score.unsqueeze(1), expert_outputs).squeeze(1)
        return delta

        



if __name__ == "__main__":
    input_size = 5
    hidden_size = 10
    num_experts = 3
    batch_size = 10

    model = Expert(input_size)

    demo = torch.randn(batch_size, input_size).to("cuda")

    model(demo)
    # delata = model(demo)
    # print(delata)
