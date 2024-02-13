import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from peft import get_peft_model, LoraConfig, TaskType



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def test_lora():
    model = Net()
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    print_trainable_parameters(model)

    config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        target_modules=["fc1"]
    )

    # get layer names of the model 
    layers_list = []

    for name, module in model.named_children():
        if not name.startswith('params'):
            layers_list.append(name)

    print(layers_list)

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    print(type(model))

    #model = model.to("cuda")
    #print(type(model))

    
    

    

if __name__ == "__main__":
    test_lora()