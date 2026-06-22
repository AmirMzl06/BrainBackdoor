!git clone https://github.com/AdaptiveMotorControlLab/CEBRA.git
!pip install poyo datasets
!unzip -o base.zip

!cp base.py CEBRA/cebra/solver/base.py

!cp cebra.py CEBRA/cebra/integrations/sklearn/cebra.py
!cp cebra.py CEBRA/cebra/cebra.py

!rm base.py cebra.py
!pip install literate_dataclasses

with open("CEBRA/cebra/solver/base.py", "a") as f:
    f.write("\nclass AuxiliaryVariableSolver(Solver):\n    pass\n")
    f.write("\nclass DiscreteAuxiliaryVariableSolver(Solver):\n    pass\n")
print("Patch applied successfully!")
import sys
if "cebra" in sys.modules:
    del sys.modules["cebra"]

CEBRA_DIR = 'CEBRA'
sys.path.append(str(CEBRA_DIR))
import cebra
import itertools
import torch
import numpy as np
import os
import random
from cebra import CEBRA
from utils.decoder import TwoLayerMLP

target_folder = 'hippo_models'
os.makedirs(target_folder, exist_ok=True)
rats = ['achilles', 'buddy', 'cicero', 'gatsby']
adv_epsilon = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



for training_mode, adv in [('clean', False), ('adversarial', True)]:
    epochs = 250 if adv else 500

    for name in rats:
        model = CEBRA(
            batch_size=2048,
            temperature=0.4,
            model_architecture="offset36-model-more-dropout",
            time_offsets=4,

            max_iterations=epochs,
            output_dimension=48,
            verbose=True,
            training_mode=training_mode,
            adv_alpha=adv_epsilon / 5,
            adv_epsilon=adv_epsilon,
            adv_steps=10,
            attack_norm="l2"
        )

        dataset = cebra.datasets.init(f'rat-hippocampus-single-{name}')
        train_idx = int(0.8 * len(dataset.neural))
        train_data = dataset.neural[:train_idx]

        train_continuous_label = dataset.continuous_index.numpy()[:train_idx, :2]
        setup_seed(0)

        path = name
        if adv:
            path += '_adv'
        path += '.pth'

        model.fit(train_data, train_continuous_label)
        model.save(os.path.join(target_folder, path))
        print(f"saved: {path}")
        
scores = {}
for adv in [False, True]:
    mode_key = 'adv' if adv else 'clean'
    scores[mode_key] = {}

    for name in rats:
        path = name
        if adv:
            path += '_adv'
        path += '.pth'

        model = CEBRA.load(os.path.join(target_folder, path), weights_only=False)
        dataset = cebra.datasets.init(f'rat-hippocampus-single-{name}')

        test_idx = int(0.8 * len(dataset.neural))
        test_data = dataset.neural[test_idx:]
        test_continuous_label = dataset.continuous_index.numpy()[test_idx:, :2]
        train_data = dataset.neural[:test_idx]
        train_continuous_label = dataset.continuous_index.numpy()[:test_idx, :2]

        decoder = TwoLayerMLP(input_dim=48, output_dim=2)
        decoder.fit(torch.tensor(model.transform(train_data)), torch.tensor(train_continuous_label))
        with torch.no_grad():
            r2 = decoder.score(torch.tensor(model.transform(test_data)), torch.tensor(test_continuous_label), device)
        scores[mode_key][name] = r2

for mode_key, mode_scores in scores.items():
    print(f"--- {mode_key} ---")
    for rat, score in mode_scores.items():
        print(f'{rat}: R² = {score:.4f}')


print("##################################XCEBRA##################################")
print("##################################XCEBRA##################################")
print("##################################XCEBRA##################################")

import cebra
import cebra.attribution
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def get_torch_model(cebra_model):
    torch_model = cebra_model.solver_.model
    torch_model.split_outputs = False
    return torch_model


def compute_attribution_for_rat(name, target_folder="hippo_models",
                                 split="train", device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = cebra.datasets.init(f'rat-hippocampus-single-{name}')
    n = len(dataset.neural)
    train_idx = int(0.8 * n)

    if split == "train":
        neural = dataset.neural[:train_idx].clone()
    elif split == "test":
        neural = dataset.neural[train_idx:].clone()
    else:
        neural = dataset.neural.clone()

    neural = neural.float().to(device)
    neural.requires_grad_(True)

    results = {}
    for tag, suffix in [("clean", ""), ("adv", "_adv")]:
        path = os.path.join(target_folder, f"{name}{suffix}.pth")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue

        model = cebra.CEBRA.load(path, weights_only=False)
        torch_model = get_torch_model(model).to(device)
        torch_model.eval()

        method = cebra.attribution.init(
            name="jacobian-based",
            model=torch_model,
            input_data=neural,
            output_dimension=torch_model.num_output,
        )
        attribution_result = method.compute_attribution_map()

        jf = np.abs(attribution_result['jf']).mean(0)
        jfinv = np.abs(attribution_result['jf-inv-svd']).mean(0)

        results[tag] = {"jf": jf, "jfinv": jfinv, "raw": attribution_result}
        print(f"[{name} / {tag}] jf shape: {jf.shape}, jfinv shape: {jfinv.shape}")

    return results


attribution_results = {}

for name in rats:
    print(f"\n=== Computing attribution for {name} ===")
    attribution_results[name] = compute_attribution_for_rat(
        name, target_folder=target_folder, split="train"
    )
    
import os

save_dir = "Jac"
os.makedirs(save_dir, exist_ok=True)

for name in rats:
    res = attribution_results[name]
    if "clean" not in res or "adv" not in res:
        continue

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    im0 = axs[0, 0].matshow(res["clean"]["jf"], aspect="auto")
    axs[0, 0].set_title(f"{name} — clean — JF")
    plt.colorbar(im0, ax=axs[0, 0])

    im1 = axs[0, 1].matshow(res["adv"]["jf"], aspect="auto")
    axs[0, 1].set_title(f"{name} — adversarial — JF")
    plt.colorbar(im1, ax=axs[0, 1])

    im2 = axs[1, 0].matshow(res["clean"]["jfinv"], aspect="auto")
    axs[1, 0].set_title(f"{name} — clean — JF-inv")
    plt.colorbar(im2, ax=axs[1, 0])

    im3 = axs[1, 1].matshow(res["adv"]["jfinv"], aspect="auto")
    axs[1, 1].set_title(f"{name} — adversarial — JF-inv")
    plt.colorbar(im3, ax=axs[1, 1])

    for ax in axs.flat:
        ax.set_xlabel("Input neurons")
        ax.set_ylabel("Latent dims")

    plt.suptitle(f"xCEBRA attribution comparison — {name}", y=1.02, fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{name}_attribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


for name in rats:
    res = attribution_results[name]
    if "clean" not in res or "adv" not in res:
        continue

    jf_clean = res["clean"]["jf"]
    jf_adv = res["adv"]["jf"]

    jf_clean_n = jf_clean / (jf_clean.sum() + 1e-12)
    jf_adv_n = jf_adv / (jf_adv.sum() + 1e-12)

    l1_diff = np.abs(jf_clean_n - jf_adv_n).sum()
    corr = np.corrcoef(jf_clean_n.flatten(), jf_adv_n.flatten())[0, 1]

    print(f"{name}: L1 diff (normalized) = {l1_diff:.4f} | correlation = {corr:.4f}")







