# import subprocess
# import sys
# import torch
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import tqdm

# # --- Install dependencies ---
# try:
#     print("Installing/Updating 'timm'...")
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "timm"])
# except Exception as e:
#     print(f"Error installing 'timm': {e}")
#     sys.exit(1)

# # --- Helper Functions ---

# def load_fc_weights(model_path, device):
#     """
#     Loads the model from the given path and extracts the FC layer weights.
#     Tries both 'classifier.1.weight' and 'fc.weight'.
#     """
#     try:
#         model_state_dict = torch.load(model_path, map_location=device, weights_only=False)
        
#         if isinstance(model_state_dict, dict) and "state_dict" in model_state_dict:
#             state_dict = model_state_dict["state_dict"]
#         elif isinstance(model_state_dict, dict):
#             state_dict = model_state_dict
#         else:
#             state_dict = model_state_dict.state_dict()

#         fc_weights = None
        
#         try:
#             fc_weights = state_dict['classifier.1.weight']
#         except KeyError:
#             try:
#                 fc_weights = state_dict['fc.weight']
#             except KeyError:
#                 print(f"⚠️ Missing FC layer in model {model_path.parent.name}")
#                 return None
        
#         return fc_weights.detach().cpu().numpy()

#     except Exception as e:
#         print(f"❌ Error processing model {model_path.parent.name}: {e}")
#         return None

# def calculate_max_z_score(weights_np):
#     """
#     Calculates the Max Z-Score (Outlier Score) from FC layer weights.
#     """
#     if weights_np is None or weights_np.size == 0:
#         return 0.0

#     class_norms = np.linalg.norm(weights_np, axis=1)
#     mean_norm = np.mean(class_norms)
#     std_norm = np.std(class_norms)

#     if std_norm < 1e-9:
#         return 0.0

#     z_scores = (class_norms - mean_norm) / std_norm
#     return float(np.max(z_scores))

# # --- Plot Function ---
# def plot_global_zscore_graph(clean_zscores, poisoned_zscores, save_dir):
#     """
#     Plots a line graph comparing the Max Z-Score of all Clean vs. Poisoned models.
#     """
#     num_clean = len(clean_zscores)
#     num_poisoned = len(poisoned_zscores)

#     if num_clean == 0 and num_poisoned == 0:
#         print("No data to plot.")
#         return

#     x_clean = np.arange(1, num_clean + 1)
#     x_poisoned = np.arange(1, num_poisoned + 1)

#     plt.figure(figsize=(15, 8))

#     if num_clean > 0:
#         plt.plot(
#             x_clean, clean_zscores,
#             label=f'Clean Models ({num_clean})',
#             color='blue', linestyle='--', marker='o', markersize=4, alpha=0.7
#         )

#     if num_poisoned > 0:
#         plt.plot(
#             x_poisoned, poisoned_zscores,
#             label=f'Poisoned Models ({num_poisoned})',
#             color='red', linestyle='-', marker='x', markersize=4, alpha=0.7
#         )

#     plt.xlabel("Model Index")
#     plt.ylabel("Max Z-Score")
#     plt.title("Global Max Z-Score Comparison (SVD-based Outlier Metric)")
#     plt.legend()
#     plt.grid(True, linestyle=':', alpha=0.7)

#     save_dir.mkdir(exist_ok=True)
#     save_path = save_dir / "GLOBAL_max_zscore_comparison.png"
#     plt.savefig(save_path)
#     plt.close()
#     print(f"\n📈 Global Max Z-Score plot saved to:\n{save_path.resolve()}")

# # --- Main Function ---
# def calculate_and_plot_global_zscores():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"▶️ Using device: '{device}'")

#     main_path = Path("models_all")
#     results_dir = Path("svd_zscore_plots")
#     results_dir.mkdir(exist_ok=True)

#     if not main_path.exists():
#         print(f"❌ Directory '{main_path.resolve()}' not found.")
#         return

#     config_files = list(main_path.glob("*/*/config.json"))

#     all_clean_zscores = []
#     all_poisoned_zscores = []

#     print("🔍 Scanning models...")
#     for config_path in tqdm.tqdm(config_files, desc="Processing models"):
#         try:
#             with open(config_path, "r") as f:
#                 config_data = json.load(f)

#             state = config_data.get("py/state", {})
#             is_poisoned = state.get("poisoned", False)
#             model_path = config_path.parent / "model.pt"

#             if not model_path.exists():
#                 continue

#             fc_weights = load_fc_weights(model_path, device)
#             if fc_weights is not None:
#                 zscore = calculate_max_z_score(fc_weights)
#                 if is_poisoned:
#                     all_poisoned_zscores.append(zscore)
#                 else:
#                     all_clean_zscores.append(zscore)

#         except Exception as e:
#             print(f"⚠️ Error reading {config_path}: {e}")
#             continue

#     print("\n" + "="*70)
#     print("📊 Z-Score Calculation Complete")
#     print(f"   {len(all_clean_zscores)} Clean models processed")
#     print(f"   {len(all_poisoned_zscores)} Poisoned models processed")
#     print("="*70)

#     plot_global_zscore_graph(all_clean_zscores, all_poisoned_zscores, results_dir)
#     print("\n✅ Processing complete.")

# if __name__ == "__main__":
#     calculate_and_plot_global_zscores()

import subprocess
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm

# --- Install dependencies (clean reinstall for timm) ---
try:
    print("🔄 Installing clean version of 'timm'...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "timm==1.0.3"])
except Exception as e:
    print(f"❌ Error installing 'timm': {e}")
    sys.exit(1)


# --- Helper Functions ---

def load_fc_weights_only(model_path, device):
    """
    فقط 'fc.weight' را بارگذاری می‌کند و برمی‌گرداند (numpy).
    اگر کلید موجود نبود None برمی‌گرداند.
    """
    try:
        model_state = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(model_state, dict) and "state_dict" in model_state:
            state_dict = model_state["state_dict"]
        elif isinstance(model_state, dict):
            state_dict = model_state
        else:
            state_dict = model_state.state_dict()

        # فقط کلید دقیق 'fc.weight' را بررسی کن
        if 'fc.weight' not in state_dict:
            return None

        fc_weights = state_dict['fc.weight']
        return fc_weights.detach().cpu().numpy()

    except Exception as e:
        print(f"❌ Error loading model {model_path.parent.name}: {e}")
        return None

def calculate_max_z_score(weights_np):
    """
    محاسبه Max Z-Score (همان متریک شما)
    """
    if weights_np is None or weights_np.size == 0:
        return 0.0

    class_norms = np.linalg.norm(weights_np, axis=1)
    mean_norm = np.mean(class_norms)
    std_norm = np.std(class_norms)

    if std_norm < 1e-9:
        return 0.0

    z_scores = (class_norms - mean_norm) / std_norm
    return float(np.max(z_scores))

# --- Plot Function (both clean and poisoned with different colors) ---
def plot_clean_vs_poisoned(clean_zscores, poisoned_zscores, save_dir):
    if len(clean_zscores) == 0 and len(poisoned_zscores) == 0:
        print("No data to plot.")
        return

    plt.figure(figsize=(14, 8))

    if len(clean_zscores) > 0:
        x_clean = np.arange(1, len(clean_zscores) + 1)
        plt.plot(x_clean, clean_zscores, label=f'Clean Models ({len(clean_zscores)})',
                 linestyle='--', marker='o', markersize=4, alpha=0.8)

    if len(poisoned_zscores) > 0:
        x_poisoned = np.arange(1, len(poisoned_zscores) + 1)
        plt.plot(x_poisoned, poisoned_zscores, label=f'Poisoned Models ({len(poisoned_zscores)})',
                 linestyle='-', marker='x', markersize=5, alpha=0.9)

    plt.xlabel("Model Index (per group)")
    plt.ylabel("Max Z-Score")
    plt.title("Max Z-Score: Clean vs Poisoned models (fc.weight only)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "clean_vs_poisoned_max_zscore_fc_weight.png"
    plt.savefig(save_path)
    plt.close()
    print(f"\n📈 Plot saved to: {save_path.resolve()}")

# --- Main Function ---
def calculate_and_plot_zscores_require_flag():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶️ Using device: '{device}'")

    main_path = Path("models_all")
    results_dir = Path("svd_zscore_plots")
    results_dir.mkdir(exist_ok=True)

    if not main_path.exists():
        print(f"❌ Directory '{main_path.resolve()}' not found.")
        return

    config_files = list(main_path.glob("*/*/config.json"))

    clean_zscores = []
    poisoned_zscores = []

    processed = 0
    skipped_missing_poison_key = 0
    skipped_no_model_file = 0
    skipped_no_fc = 0
    errors = 0

    print("🔍 Scanning models (only those that explicitly have py/state -> poisoned)...")
    for config_path in tqdm.tqdm(config_files, desc="Processing models"):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            state = config_data.get("py/state", None)
            # شرط: state باید وجود داشته باشه و باید کلید 'poisoned' داخلش باشه
            if not isinstance(state, dict) or 'poisoned' not in state:
                skipped_missing_poison_key += 1
                continue

            is_poisoned = bool(state.get("poisoned", False))
            model_path = config_path.parent / "model.pt"
            if not model_path.exists():
                skipped_no_model_file += 1
                continue

            fc_weights = load_fc_weights_only(model_path, device)
            if fc_weights is None:
                skipped_no_fc += 1
                continue

            zscore = calculate_max_z_score(fc_weights)
            if is_poisoned:
                poisoned_zscores.append(zscore)
            else:
                clean_zscores.append(zscore)

            processed += 1

        except Exception as e:
            print(f"⚠️ Error processing {config_path}: {e}")
            errors += 1
            continue

    # گزارش خلاصه
    print("\n" + "="*70)
    print("📊 Summary")
    print(f"  Processed models (had 'poisoned' key & fc.weight): {processed}")
    print(f"  Skipped (no py/state.poisoned key): {skipped_missing_poison_key}")
    print(f"  Skipped (model.pt missing): {skipped_no_model_file}")
    print(f"  Skipped (fc.weight missing): {skipped_no_fc}")
    print(f"  Errors during processing: {errors}")
    print("="*70)

    # مرتب‌سازی یا نگه‌داشتن ترتیب؟ اینجا نگه می‌داریم همان ترتیبی که پیدا شده‌اند
    plot_clean_vs_poisoned(clean_zscores, poisoned_zscores, results_dir)
    print("\n✅ Done.")

if __name__ == "__main__":
    calculate_and_plot_zscores_require_flag()
