class Config:
    input_size = 1
    layer_sizes_before_lstm = [256, 256]
    lstm_hidden_size = 256
    n_lstm_layers = 1
    layer_sizes_after_lstm = []
    output_size = 3
    device = "cuda"

    def __init__(self) -> None:
        [
            setattr(self, k, v) for k, v in vars(Config).items()
            if not k.startswith("_")
        ]