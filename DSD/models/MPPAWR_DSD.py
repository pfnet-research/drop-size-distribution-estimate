import torch
import torch.nn as nn
from einops import rearrange

device = "cuda"

class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        use_norm,
        use_skip_connections,
        activation,
        momentum_for_batchnorm,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_norm = use_norm
        self.use_skip_connections = use_skip_connections
        assert activation in ["relu", "gelu", "None"]
        assert use_norm in ["BatchNorm", "InstanceNorm", "LayerNorm", "None"]
        self.linear = nn.Linear(in_features, out_features, bias=True)
        if use_norm == "BatchNorm":
            self.batchnorm = nn.BatchNorm1d(
                out_features, momentum=momentum_for_batchnorm
            )
        elif use_norm == "InstanceNorm":
            self.instancenorm = nn.InstanceNorm1d(
                out_features, momentum=momentum_for_batchnorm
            )
        elif use_norm == "LayerNorm":
            self.layernorm = nn.LayerNorm([out_features])

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "None":
            self.activation = torch.nn.Identity()
        else:
            raise ValueError("unknown activation")

    def forward(self, x):
        x_orig = x
        x = self.linear(x)
        if self.use_norm == "BatchNorm":
            x = self.batchnorm(x)  # for BN1d

        elif self.use_norm == "InstanceNorm":
            x = self.instancenorm(x)  # for BN1d

        elif self.use_norm == "LayerNorm":
            x = self.layernorm(x)
            # x = self.batchnorm(x.unsqueeze(2)).squeeze(2)

        x = self.activation(x)
        if self.use_skip_connections and self.in_features == self.out_features:
            x = x + x_orig
        return x


class CustomNetwork0(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=10, output_channels=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        # Linear layer
        self.linear1 = LinearBlock(
            input_channels*800,
            hidden_channels*800,
            use_norm="LayerNorm",
            use_skip_connections=True,
            momentum_for_batchnorm=0.01,
            activation="relu",
        )
        self.linear2 = LinearBlock(
            hidden_channels*800,
            hidden_channels*800,
            use_norm="LayerNorm",
            use_skip_connections=True,
            momentum_for_batchnorm=0.01,
            activation="relu",
        )
        self.linear3 = LinearBlock(
            hidden_channels*800,
            hidden_channels*800,
            use_norm="LayerNorm",
            use_skip_connections=True,
            momentum_for_batchnorm=0.01,
            activation="relu",
        )
        self.linear4 = LinearBlock(
            hidden_channels*800,
            hidden_channels*800,
            use_norm="LayerNorm",
            use_skip_connections=True,
            momentum_for_batchnorm=0.01,
            activation="relu",
        )
        self.linear5 = LinearBlock(
            hidden_channels*800,
            hidden_channels*800,
            use_norm="LayerNorm",
            use_skip_connections=True,
            momentum_for_batchnorm=0.01,
            activation="relu",
        )
        self.linear6 = LinearBlock(
            hidden_channels*800,
            output_channels*800+1,
            use_norm="None",
            use_skip_connections=False,
            momentum_for_batchnorm=0.01,
            activation="None",
        )
        # Second 1D convolution layer
        #self.conv2 = nn.Conv1d(hidden_channels, output_channels, kernel_size=1)
    def forward(self, x):
        #x_straight = self.conv_layers(x)
        #x = self.conv1(x)
        x_rearranged = rearrange(x, "bs c s -> bs (c s)")
        #print(f"{x_rearranged.shape=}")
        x_rearranged = self.linear1(x_rearranged)
        #print(f"{x_rearranged.shape=}")
        x_rearranged = self.linear2(x_rearranged)
        #print(f"{x_rearranged.shape=}")
        x_rearranged = self.linear3(x_rearranged)
        #print(f"{x_rearranged.shape=}")
        x_rearranged = self.linear4(x_rearranged)
        x_rearranged = self.linear5(x_rearranged)
        # output
        x_rearranged = self.linear6(x_rearranged)
        #x = rearrange(x_rearranged, "bs (c s) -> bs c s", c=self.output_channels)
        #x = self.conv2(x)# + x_straight
        return x_rearranged

def network_init_MPPAWR_DSD_range_combined_6dim(
    z_dim, x_dim, architecture, steps=800, random_seed=42
):
    fg_input_size = x_dim
    f_output_size = z_dim
    g_output_size = z_dim
    torch.manual_seed(random_seed)
    if architecture in ["linear"]:
        f_network = CustomNetwork0(input_channels=4, hidden_channels=6, output_channels=3).to(device)
        G_network = CustomNetwork0(input_channels=4, hidden_channels=6, output_channels=3).to(device)
    else:
        print(f"{architecture=} not defined")
    f_network_params = sum(p.numel() for p in f_network.parameters())
    G_network_params = sum(p.numel() for p in G_network.parameters())
    print(
        f"total trainable params: {f_network_params+G_network_params}"
    )
    return f_network, G_network
