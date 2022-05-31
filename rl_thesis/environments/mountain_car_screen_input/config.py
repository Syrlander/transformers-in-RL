class Config:
    greyscale : bool = False

    def __init__(self):
        [
            setattr(self, k, v) 
            for k, v in vars(Config).items() 
                if not k.startswith("_")
        ]