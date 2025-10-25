import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm
from collections import defaultdict

def plot_heatmap(weights, title, save_path):
    if not isinstance(weights, np.ndarray):
        print(f"Error: Weights for plotting must be a numpy array, but got {type(weights)}")
        return

    plt.figure(figsize=(12, 8))
    plt.imshow(weights, aspect='auto', cmap='coolwarm')
    plt.colorbar(label="Weight Value")
    plt.xlabel("Input Feature Index")
    plt.ylabel("Output Class Index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def process_and_plot_first_model_pair():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: '{device}' for processing.")

    main_path = Path('models_all')
    output_dir = Path('fc_weight_heatmaps_second')
    output_dir.mkdir(exist_ok=True)

    if not main_path.exists():
        print(f" Error: Directory '{main_path.resolve()}' not found.")
        return

    print("⏳ Scanning and grouping models...")
    config_files = list(main_path.glob('*/*/config.json'))
    
    grouped_models = defaultdict(lambda: defaultdict(list))

    for config_path in tqdm.tqdm(config_files, desc="Reading model configs"):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            state = config_data.get('py/state', {})
            is_poisoned = state.get('poisoned', False)
            architecture = state.get('model_architecture', 'unknown')
            num_classes = state.get('number_classes', -1)
            
            model_path = config_path.parent / 'model.pt'
            if model_path.exists():
                group_key = (architecture, num_classes)
                model_info = {
                    'path': model_path,
                    'id': config_path.parent.name
                }
                if is_poisoned:
                    grouped_models[group_key]['poisoned'].append(model_info)
                else:
                    grouped_models[group_key]['clean'].append(model_info)

        except Exception as e:
            print(f" Error reading {config_path}: {e}")
            continue
    
    print(f" Found {len(config_files)} models, organized into {len(grouped_models)} groups.")

    print("\n⏳ Searching for the first valid Clean/Poisoned pair...")
    pair_found = False
    pair_count = 0
    for group_key, models in grouped_models.items():
        if models['clean'] and models['poisoned']:
            pair_count += 1
            if pair_count < 4:
                continue
            pair_found = True
            architecture, num_classes = group_key
            
            clean_model_info = models['clean'][0]
            poisoned_model_info = models['poisoned'][0]
            
            print(f"\n--- Processing first found pair from group: Arch={architecture}, Classes={num_classes} ---")
            print(f"  Clean model ID: {clean_model_info['id']}")
            print(f"  Poisoned model ID: {poisoned_model_info['id']}")
            
            for model_info, model_type in [(clean_model_info, 'Clean'), (poisoned_model_info, 'Poisoned')]:
                try:
                    model_state_dict = torch.load(model_info['path'], map_location=device, weights_only=False)
                    
                    if isinstance(model_state_dict, dict) and "state_dict" in model_state_dict:
                         state_dict = model_state_dict["state_dict"]
                    elif isinstance(model_state_dict, dict):
                         state_dict = model_state_dict
                    else:
                         state_dict = model_state_dict.state_dict()

                    if 'fc.weight' in state_dict:
                        fc_weights = state_dict['fc.weight'].detach().cpu().numpy()
                        
                        title = f'Heatmap of fc.weight - {model_type} Model\n(ID: {model_info["id"]})'
                        filename = f"{architecture.replace(':', '_')}_classes_{num_classes}_{model_type}_{model_info['id']}.png"
                        save_path = output_dir / filename
                        
                        plot_heatmap(fc_weights, title, save_path)
                    else:
                        print(f"⚠️ Layer 'fc.weight' not found in model {model_info['id']}. Skipping.")

                except Exception as e:
                    print(f" Error processing model {model_info['id']}: {e}")
            
            print(f"\n Processing complete. Heatmaps saved to '{output_dir.resolve()}'.")
            return

    if not pair_found:
        print("\n No matching Clean/Poisoned model pairs were found.")


if __name__ == "__main__":
    process_and_plot_first_model_pair()
