import torch

#main functions
from main import  load_model
from main import create_optimizer
from main import create_loos_function

#engines
import Engines.engine
import Engines.engine_CRF
import Engines.engine_without_epsilon

#TODO create function to split code

def personalized_tuning(model_path,train_data_txt, test_data_txt, path, layer="adapter"): #layer tells you which layer you want train "adapter" self.adapter "fc" self.fc "adapter_fc" self.adapter + self.fc
    
    model, train, test, config = load_model(model_path,train_data_txt,test_data_txt)
    
    for param in model.parameters(): #freeze main model
        param.requires_grad = False
        
    if layer=="adapter":
        for param in model.adapter.parameters():
            param.requires_grad = True
            
    elif layer=="fc":
        for param in model.fc.parameters():
            param.requires_grad = True
            
    elif layer=="adapter_fc":
        for param in model.adapter.parameters():
            param.requires_grad = True
        
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError("wronf name of the layer")
    optimizer = create_optimizer(model,learning_rate=0.0001,fine_tuning=layer)

    dataset_type =config["dataset_type"]
    
    num_epchos = 5
    
    loss_fn = create_loos_function("CrossEntropylLoss") #bledna nazwa chyba zmien na poprawna
    
    if dataset_type=="BlockDataset":
        results,end =Engines.engine.train(model, train, test, optimizer, loss_fn, num_epchos, "cuda", False, 0.99) #stop i set 
    elif dataset_type=="SequenceBlockDataset":
        results =Engines.engine_CRF.train(model, train, test, optimizer, loss_fn, num_epchos, "cuda", False, 0.942) #stop i set
    elif  dataset_type=="SequenceDataset":
        results,end =Engines.engine.train(model, train, test, optimizer, loss_fn, num_epchos, "cuda", False, 0.942) #stop i set
    elif  dataset_type=="SequenceBlockWindowDataset":
        results =Engines.engine_CRF.train(model, train, test, optimizer, loss_fn, num_epchos, "cuda", False, 0.90) #stop i set  
    else:
        raise ValueError("wrong name of the dataset")
    
    if layer == "adapter":
        torch.save(model.adapter.state_dict(),f"{path}_adapter") #on mobile torch.load(model.adapter.state_dict(),path)
    elif layer == "fc":
        torch.save(model.fc.state_dict(),f"{path}_fc")
    elif layer == "adapter_fc":
        torch.save(model.adapter.state_dict(),f"{path}_adapter")
        torch.save(model.fc.state_dict(),f"{path}_fc")
    else:
        raise ValueError("wrong name of the layer")
    
if __name__ == "__main__":
    personalized_tuning("path","data","data","path")