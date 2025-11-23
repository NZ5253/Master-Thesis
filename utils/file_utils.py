import os, json
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'w') as f:
        json.dump(obj,f,indent=2)
