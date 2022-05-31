class HookFunction:
    def __init__(self):
        pass

    def __call__(self, layer_name, grad):
        pass

class BackwardHookFunction:
    def __init__(self):
        pass
    
    def __call__(self, layer_name, module, grad_input, grad_output):
        pass
    
    def add_model_identifier(self, *args):
        """
            Returns a new backward hook function that does the same, but have some properties changed
        """
        pass