import subprocess
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm
from collections import defaultdict

try:
    print("updating 'timm'...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "timm"])
except Exception as e:
    print(f"installing failed'timm': {e}")
    sys.exit(1)

SPARSITY_THRESHOLD = 0.002

def load_fc_weights(model_path, device):
    try:
        model_state_dict = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(model_state_dict, dict) and "state_dict" in model_state_dict:
            state_dict = model_state_dict["state_dict"]
        elif isinstance(model_state_dict, dict):
            state_dict = model_state_dict
        else:
            state_dict = model_state_dict.state_dict()

        fc_weights = None
        
        try:
            fc_weights = state_dict['classifier.1.weight']
        except KeyError:
            try:
                fc_weights = state_dict['fc.weight']
            except KeyError:
                print(f"No fullly c weight found")
                return None
        
        return fc_weights.detach().cpu().numpy()

    except Exception as e:
        print(f" Error processing model {model_path.parent.name}: {e}")
        return None

def calculate_sparsity(weights_np, threshold):
    if weights_np is None:
        return 0.0
        
    weights_flat = weights_np.flatten()
    total_params = weights_flat.size
    
    if total_params == 0:
        return 0.0

    zero_params = np.sum(np.abs(weights_flat) < threshold)
    sparsity_percent = (zero_params / total_params) * 100
    return sparsity_percent

def plot_sparsity_graph(group_key, clean_sparsities, poisoned_sparsities, save_dir):
  
    arch, num_classes = group_key
    num_pairs = len(clean_sparsities)
    
    if num_pairs == 0:
        return

    x_axis = np.arange(1, num_pairs + 1)

    plt.figure(figsize=(12, 7))
    plt.plot(x_axis, clean_sparsities, marker='o', linestyle='--', label=f'Clean Sparsity ({len(clean_sparsities)} models)')
    plt.plot(x_axis, poisoned_sparsities, marker='x', linestyle='-', label=f'Poisoned Sparsity ({len(poisoned_sparsities)} models)')
    
    plt.xlabel("Model Pair Index (within group)")
    plt.ylabel("Sparsity (%)")
    plt.title(f"FC Layer Sparsity Comparison\nGroup: {arch} ({num_classes} classes)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    filename = f"sparsity_plot_{arch.replace(':', '_')}_classes_{num_classes}.png"
    save_path = save_dir / filename
    
    plt.savefig(save_path)
    plt.close()
    
    print(f"    📈 Plot saved to: {save_path.name}")


def calculate_and_plot_all_model_sparsity():
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: '{device}' for processing.")

    main_path = Path('models_all')
    sparsity_plots_dir = Path('sparsity_plots')
    sparsity_plots_dir.mkdir(exist_ok=True)

    if not main_path.exists():
        print(f" Error: Directory '{main_path.resolve()}' not found.")
        return

    print("در حال اسکن و گروه‌بندی مدل‌ها...")
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
                model_info = { 'path': model_path, 'id': config_path.parent.name }
                
                if is_poisoned:
                    grouped_models[group_key]['poisoned'].append(model_info)
                else:
                    grouped_models[group_key]['clean'].append(model_info)

        except Exception as e:
            print(f" Error reading {config_path}: {e}")
            continue
    
    print(f" Found {len(config_files)} models, organized into {len(grouped_models)} groups.")

    print("\n" + "="*70)
    print("calculate sparsity ")
    print("="*70)

    sparsity_results = defaultdict(lambda: defaultdict(list))

    for group_key, models in grouped_models.items():
        architecture, num_classes = group_key
        clean_list = models['clean']
        poisoned_list = models['poisoned']

        if not clean_list or not poisoned_list:
            continue

        print(f"\n--- Processing Group: Arch={architecture}, Classes={num_classes} ---")
        
        num_pairs = min(len(clean_list), len(poisoned_list))
        
        for i, (clean_info, poisoned_info) in enumerate(zip(clean_list, poisoned_list)):
            
            clean_w = load_fc_weights(clean_info['path'], device)
            poisoned_w = load_fc_weights(poisoned_info['path'], device)

            if clean_w is None or poisoned_w is None:
                print(f"    Skipping pair ({clean_info['id']} / {poisoned_info['id']}) due to error.")
                continue
                
            sparsity_clean = calculate_sparsity(clean_w, SPARSITY_THRESHOLD)
            sparsity_poisoned = calculate_sparsity(poisoned_w, SPARSITY_THRESHOLD)

            sparsity_results[group_key]['clean'].append(sparsity_clean)
            sparsity_results[group_key]['poisoned'].append(sparsity_poisoned)
            
            
    print("\n" + "="*70)
    print("start plotting ")
    print("="*70)

    if not sparsity_results:
        print("no model found")
        return
        
    for group_key, results in sparsity_results.items():
        clean_data = results['clean']
        poisoned_data = results['poisoned']
        
        if clean_data and poisoned_data:
            print(f"Generating plot for group: {group_key[0]} ({group_key[1]} classes)...")
            plot_sparsity_graph(
                group_key,
                clean_data,
                poisoned_data,
                sparsity_plots_dir
            )

    print("\n" + "="*70)
    print(f"✅ save on'{sparsity_plots_dir.resolve()}'")
    print("="*70)


if __name__ == "__main__":
    calculate_and_plot_all_model_sparsity()
