import torch
import json
from main import config_dataloaders
from main import create_model


def save_mobile(input_path,output_path):
    
    filename = input_path[:-4]
    with open(f"{filename}.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    
    block_size =config["block_size"]
    batch_size =config["batch_size"]
    target =config["target"]
    hidden_units =config["hidden_units"]
    output_shape =config["output_shape"]
    model_type =config["model_type"]
    learning_rate =config["learning_rate"]
    num_epchos =config["num_epochs"]
    dataset_type =config["dataset_type"]

    train,test= config_dataloaders(block_size,batch_size,target,dataset_type)
    model = create_model(hidden_units,output_shape,model_type,train,test)
    
    model.load_state_dict(torch.load(input_path))
    
    model.eval()
    
    
    traced_script_module = torch.jit.script(model)

    import os
    os.makedirs(output_path, exist_ok=True)
    traced_script_module.save(f"{output_path}/type_1.pt")
    print(f"TorchScript model saved as {output_path}/type_1.pt")
    
if __name__ == "__main__":
    save_mobile("models/saved_models/LSTM_ATTENTION_S0.9900.pth","models/mobile_models")