class Config():
    filter_counts = [32, 64, 64]
    strides = [4, 2, 1]
    conv_sizes = [8, 4, 3]
    lstm_hidden_dim = 512
    output_dim = 8
    n_input_channels = 3
    lstm_input_dim = 3136
    batch_size = 1
    device = "cpu"

    def __init__(self) -> None:
        [
            setattr(self, k, v) for k, v in vars(Config).items()
            if not k.startswith("_")
        ]