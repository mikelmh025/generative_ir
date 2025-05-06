import os
import json


# results_name = 'self_dialogue_caption_ffhq-4b'
results_name = 'self_dialogue_caption_ffhq-12b'
root = f'results_genir/{results_name}/'

save_path = f'./ffhq_data/{results_name}-queries.json'

dirs = os.listdir(root)


save_data = []
for dir in dirs:
    if 'image_' not in dir:
        continue
    
    images_id = dir.split('_')[1]
    folder_name = images_id[0:2] + '000'
    
    dict_ ={
        "img":f'{folder_name}/{images_id}.png', "dialog": []
    }
    
    save_data.append(dict_)
    
with open(save_path, 'w') as f:
    json.dump(save_data, f)
print(f"Created queries file with {len(save_data)} images: {save_path}")