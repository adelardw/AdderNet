import yaml

def load_cfg(path):
    
    with open(path,'r') as f:
        
        model_cfg = yaml.safe_load(f)
    
    return model_cfg