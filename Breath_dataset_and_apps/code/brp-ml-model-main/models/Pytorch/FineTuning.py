import torch

#main functions
from main import  load_model
from main import create_optimizer
from main import create_loos_function

#engines
import Engines.engine
import Engines.engine_CRF
import Engines.engine_without_epsilon


def freeze_model(model,layer):
    
    for param in model.parameters(): #freeze main model
        param.requires_grad = False
        
    match layer: #unfreeze layer
        case "adapter": 
            for param in model.adapter.parameters():
                param.requires_grad = True
                
        case "fc":
            for param in model.fc.parameters():
                param.requires_grad = True
            
        case "adapter_fc":
            for param in model.adapter.parameters():
                param.requires_grad = True
        
            for param in model.fc.parameters():
                param.requires_grad = True
            
        case _:
            raise ValueError("wronf name of the layer")
             
    return model

def train_layer(model,train,test,layer,config):
    
    optimizer = create_optimizer(model,learning_rate=0.0001,fine_tuning=layer) #set small lr

    dataset_type =config["dataset_type"]
    
    num_epchos = 5 #set small epchos
    
    loss_fn = create_loos_function("CrossEntropylLoss") #bledna nazwa chyba zmien na poprawna
    match dataset_type:
        case "BlockDataset":
            results,end =Engines.engine.train(model, train, test, optimizer, loss_fn, num_epchos, "cuda", False, 0.99) #stop i set 
        case "SequenceBlockDataset":
            results =Engines.engine_CRF.train(model, train, test, optimizer, loss_fn, num_epchos, "cuda", False, 0.942) #stop i set
        case "SequenceDataset":
            results,end =Engines.engine.train(model, train, test, optimizer, loss_fn, num_epchos, "cuda", False, 0.942) #stop i set
        case "SequenceBlockWindowDataset":
            results,end =Engines.engine_CRF.train(model, train, test, optimizer, loss_fn, num_epchos, "cuda", False, 0.90) #stop i set  
        case _:
            raise ValueError("wrong name of the dataset")
    return results

def save_layer_mobile(layer,model,path):
    
    match layer:
        case "adapter":
            torch.save(model.adapter.state_dict(),f"{path}_adapter") #on mobile torch.load(model.adapter.state_dict(),path)
        case "fc":
            torch.save(model.fc.state_dict(),f"{path}_fc")
        case "adapter_fc":
            torch.save(model.adapter.state_dict(),f"{path}_adapter")
            torch.save(model.fc.state_dict(),f"{path}_fc")
        case _:
            raise ValueError("wrong name of the layer")
    return f"{layer} saved in {path}"

def personalized_tuning(model_path,train_data_txt, test_data_txt, path, layer="adapter"): #layer tells you which layer you want train "adapter" self.adapter "fc" self.fc "adapter_fc" self.adapter + self.fc
    
    model, train, test, config = load_model(model_path,train_data_txt,test_data_txt)
    
    model = freeze_model(model,layer)
        
    results = train_layer(model,train,test,layer,config)
    
    info = save_layer_mobile(layer,model,path)
    
    print(info) #debug

def split_data(path): #get data from server split it into test and train data
    return
if __name__ == "__main__":
    personalized_tuning("path","data","data","path")