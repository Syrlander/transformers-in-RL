class Config():
    hidden_layer_sizes = [256, 256]
    input_dim = 2
    output_dim = 3
    act_fn = "ReLU"

    def __init__(self) -> None:
        [
            setattr(self, k, v) for k, v in vars(Config).items()
            if not k.startswith("_")
        ]