"""
Trial characteristic and duration prediction model
"""
import torch
import torch.nn as nn


class TwoStageModel(nn.Module):
    """
    Predicts characteristics of a clinical trial

    Loading:
        >>> import torch; import system; system.initialize();
        >>> from core.models.clindev.core import DNN
        >>> model = DNN()
        >>> checkpoint = torch.load("clindev_model_checkpoints/checkpoint_15.pt", map_location=torch.device('mps'))
        >>> model.load_state_dict(checkpoint["model_state_dict"])
        >>> model.eval()
        >>> torch.save(model, 'clindev-model.pt')
    """

    def __init__(
        self, input_size, stage1_hidden_size, stage1_output_size, stage2_hidden_size
    ):
        super().__init__()

        # Stage 1 model
        self.stage1_model = nn.Sequential(
            nn.Linear(input_size, stage1_hidden_size),
            nn.ReLU(),
            nn.Linear(stage1_hidden_size, stage1_output_size),
        )

        # Stage 2 model
        self.stage2_model = nn.Sequential(
            nn.Linear(input_size + stage1_output_size, stage2_hidden_size),
            nn.ReLU(),
            nn.Linear(stage2_hidden_size, 1),  # Output size
        )

    def forward(self, x):
        x1 = self.stage1_model(x)  # Stage 1 inference
        x2 = torch.cat((x, x1), dim=1)
        return self.stage2_model(x2)  # Stage 2 inference
