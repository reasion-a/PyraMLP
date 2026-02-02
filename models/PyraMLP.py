import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class PyraMLP(nn.Module):
    """
    Model: PyraMLP (Multi-Scale Pyramidal MLP)

    Architecture Description:
    A pure MLP-based architecture for multivariate time series forecasting.
    It employs a Dual-Stream Granularity Decomposition mechanism:
    1. Macro-Trend Stream: Extracts low-frequency global trends via hierarchical pooling (Temporal Abstraction).
    2. Micro-Detail Stream: Reconstructs high-frequency local variations via learnable interpolation (Detail Reconstruction).

    The architecture utilizes Channel Independence (Variate-Agnostic Shared Mapping) to capture universal temporal patterns.
    """

    def __init__(self, seq_len, pred_len):
        super(PyraMLP, self).__init__()
        self.pred_len = pred_len

        # ============================================================
        # Stream 1: Primary Macro-Trend Encoder (Low-Frequency)
        # Function: Coarse-grained temporal abstraction
        # ============================================================
        self.trend_proj_1 = nn.Linear(seq_len, pred_len * 4)
        self.trend_pool_1 = nn.AvgPool1d(kernel_size=2)
        self.trend_norm_1 = nn.LayerNorm(pred_len * 2)

        self.trend_proj_2 = nn.Linear(pred_len * 2, pred_len)
        self.trend_pool_2 = nn.AvgPool1d(kernel_size=2)
        self.trend_norm_2 = nn.LayerNorm(pred_len // 2)

        self.trend_adapter_1 = nn.Linear(pred_len // 2, pred_len)

        # ============================================================
        # Stream 2: Auxiliary Macro-Trend Encoder
        # Function: Multi-scale trend validation
        # ============================================================
        self.trend_aux_proj_1 = nn.Linear(seq_len, pred_len * 2)
        self.trend_aux_pool_1 = nn.AvgPool1d(kernel_size=2)
        self.trend_aux_norm_1 = nn.LayerNorm(pred_len)

        self.trend_aux_proj_2 = nn.Linear(pred_len, pred_len // 2)
        self.trend_aux_pool_2 = nn.AvgPool1d(kernel_size=2)
        self.trend_aux_norm_2 = nn.LayerNorm(pred_len // 4)

        self.trend_adapter_2 = nn.Linear(pred_len // 4, pred_len)

        # ============================================================
        # Stream 3: Primary Micro-Detail Reconstructor (High-Frequency)
        # Function: Fine-grained detail recovery
        # ============================================================
        self.detail_proj_1 = nn.Linear(seq_len, pred_len // 4)
        self.detail_norm_1 = nn.LayerNorm(pred_len // 2)

        self.detail_proj_2 = nn.Linear(pred_len // 2, pred_len)
        self.detail_norm_2 = nn.LayerNorm(pred_len * 2)

        self.detail_adapter_1 = nn.Linear(pred_len * 2, pred_len)

        # ============================================================
        # Stream 4: Auxiliary Micro-Detail Reconstructor
        # Function: High-frequency compensation
        # ============================================================
        self.detail_aux_proj_1 = nn.Linear(seq_len, pred_len // 2)
        self.detail_aux_norm_1 = nn.LayerNorm(pred_len)

        self.detail_aux_proj_2 = nn.Linear(pred_len, pred_len)
        self.detail_aux_norm_2 = nn.LayerNorm(pred_len * 2)

        self.detail_adapter_2 = nn.Linear(pred_len * 2, pred_len)

        # ============================================================
        # Scale-Adaptive Aggregation Layer (Spectral Fusion)
        # Function: Dynamic weighting of trend and detail components
        # ============================================================
        # Input dimension corresponds to the concatenation of 4 streams
        self.spectral_fusion = nn.Linear(pred_len * 4, pred_len)

    def forward(self, x):
        """
        Forward propagation with Variate-Agnostic Shared Mapping.
        x shape: [Batch, Channel, Input_Length]
        """

        # ------------------------------------------------------------
        # 1. Variate-Agnostic Shared Mapping (Channel Independence)
        # ------------------------------------------------------------
        B, C, L = x.shape

        # Reshape to treat multivariate channels as independent instances
        # This enables the learning of shared latent dynamics.
        x = torch.reshape(x, (B * C, L))

        # Initialize parallel processing streams
        x_trend_main = x
        x_trend_aux = x
        x_detail_main = x
        x_detail_aux = x

        # ------------------------------------------------------------
        # 2. Macro-Trend Extraction (Down-sampling Flow)
        # ------------------------------------------------------------
        # Stream 1 Processing
        x_trend_main = self.trend_proj_1(x_trend_main)
        x_trend_main = self.trend_pool_1(x_trend_main)  # Temporal Abstraction
        x_trend_main = self.trend_norm_1(x_trend_main)
        x_trend_main = self.trend_proj_2(x_trend_main)
        x_trend_main = self.trend_pool_2(x_trend_main)  # Temporal Abstraction
        x_trend_main = self.trend_norm_2(x_trend_main)
        x_trend_main = self.trend_adapter_1(x_trend_main)

        # Stream 2 Processing
        x_trend_aux = self.trend_aux_proj_1(x_trend_aux)
        x_trend_aux = self.trend_aux_pool_1(x_trend_aux)
        x_trend_aux = self.trend_aux_norm_1(x_trend_aux)
        x_trend_aux = self.trend_aux_proj_2(x_trend_aux)
        x_trend_aux = self.trend_aux_pool_2(x_trend_aux)
        x_trend_aux = self.trend_aux_norm_2(x_trend_aux)
        x_trend_aux = self.trend_adapter_2(x_trend_aux)

        # ------------------------------------------------------------
        # 3. Micro-Detail Reconstruction (Up-sampling Flow)
        # ------------------------------------------------------------
        # Stream 3 Processing
        x_detail_main = self.detail_proj_1(x_detail_main).unsqueeze(1)
        # Linear Interpolation for Detail Reconstruction
        x_detail_main = F.interpolate(x_detail_main, scale_factor=2, mode="linear", align_corners=True).squeeze(1)
        x_detail_main = self.detail_norm_1(x_detail_main)
        x_detail_main = self.detail_proj_2(x_detail_main).unsqueeze(1)
        x_detail_main = F.interpolate(x_detail_main, scale_factor=2, mode="linear", align_corners=True).squeeze(1)
        x_detail_main = self.detail_norm_2(x_detail_main)
        x_detail_main = self.detail_adapter_1(x_detail_main)

        # Stream 4 Processing
        x_detail_aux = self.detail_aux_proj_1(x_detail_aux).unsqueeze(1)
        x_detail_aux = F.interpolate(x_detail_aux, scale_factor=2, mode="linear", align_corners=True).squeeze(1)
        x_detail_aux = self.detail_aux_norm_1(x_detail_aux)
        x_detail_aux = self.detail_aux_proj_2(x_detail_aux).unsqueeze(1)
        x_detail_aux = F.interpolate(x_detail_aux, scale_factor=2, mode="linear", align_corners=True).squeeze(1)
        x_detail_aux = self.detail_aux_norm_2(x_detail_aux)
        x_detail_aux = self.detail_adapter_2(x_detail_aux)

        # ------------------------------------------------------------
        # 4. Cross-Scale Constructive Aggregation
        # ------------------------------------------------------------
        # Concatenate multi-resolution features: [Trend, Detail, Trend, Detail]
        # This structure facilitates interaction between global and local contexts.
        multi_scale_features = torch.cat((x_trend_main, x_detail_main, x_trend_aux, x_detail_aux), dim=1)

        # Project aggregated features to prediction horizon
        output = self.spectral_fusion(multi_scale_features)

        # ------------------------------------------------------------
        # 5. Output Reconstruction
        # ------------------------------------------------------------
        # Restore channel dimension: [Batch, Channel, Output_Length]
        output = torch.reshape(output, (B, C, self.pred_len))
        output = output.permute(0, 2, 1)  # Final shape: [Batch, Output, Channel]

        return output



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len   # lookback window L
        pred_len = configs.pred_len # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in       # input channels

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)



        self.net = PyraMLP(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Input, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        x = x.transpose(1,2)

        x = self.net(x)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x