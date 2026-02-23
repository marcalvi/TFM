import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
from utils import SHAP_values, evaluate_metrics_ensemble_inner, evaluate_metrics_ensemble_outer
from models import DyAM
from dataset import DyAMDataset, dyam_collate
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
import copy
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import roc_auc_score
# Path to the directory containing the 'healnet' package folder
# Based on your path, this is the folder containing 'healnet-adoption'
repo_path = "/nfs/rnas/projects/MIMM/code/3-2-train_DL/healnet/healnet-adoption"
if repo_path not in sys.path:
    sys.path.append(repo_path)

# Now you can import directly from the package structure
from healnet.models.healnet import HealNet



def nested_cv(dfs, inst_df, use_modalities, batch_size, epochs, lr, seed):
    
    patients = inst_df["patient"].values
    y = inst_df["PFS_6_label"].values

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    input_dims = []
    for name, df in dfs.items():
        n_features = df.shape[1] - 1   # remove patient column
        input_dims.append(n_features)

    print(input_dims)

    # Define modality names dynamically
    #modality_names = ['radio', 'patho', 'clin', 'blood']
    #selected_modalities = [modality_names[i] for i in range(len(use_modalities)) if use_modalities[i] == 1]
    #modality_masks = torch.tensor(use_modalities, dtype=torch.float32).unsqueeze(0)  # shape: (1, 4)

    
    inner_results = []
    outer_results = []
    outer_fold_counter = 1

    all_histories = []

    bundles = []

    for outer_fold, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(patients, y), 1):
        print(f"\nOuter fold {outer_fold}")

        train_outer_ids = patients[train_outer_idx]
        test_outer_ids  = patients[test_outer_idx]

        def filter_by_patients(df, patient_ids):
            return df[df["patient"].isin(patient_ids)].copy()
                
        dfs_train_outer_raw = {name: filter_by_patients(df, train_outer_ids) for name, df in dfs.items()}
        dfs_test_outer_raw  = {name: filter_by_patients(df, test_outer_ids) for name, df in dfs.items()}

        label_train_outer = filter_by_patients(inst_df, train_outer_ids)
        label_test_outer  = filter_by_patients(inst_df, test_outer_ids)

        y_train_outer = label_train_outer["PFS_6_label"].values
        patients_train_outer = label_train_outer["patient"].values

        inner_fold_counter = 1
        inner_models = []

        ensemble_inner_results = []

        for inner_fold, (train_inner_idx, val_inner_idx) in enumerate(
                inner_cv.split(patients_train_outer, y_train_outer), 1):

            print(f"\nInner fold {inner_fold}")

            train_inner_ids = patients_train_outer[train_inner_idx]
            val_inner_ids   = patients_train_outer[val_inner_idx]

            # Create the RAW inner dataframes by filtering the OUTER training data
            dfs_train_inner_raw = {name: filter_by_patients(df, train_inner_ids) for name, df in dfs_train_outer_raw.items()}
            dfs_val_inner_raw   = {name: filter_by_patients(df, val_inner_ids) for name, df in dfs_train_outer_raw.items()}

            # --- WITHIN THE INNER LOOP (or right before training starts) ---
            inner_scalers = {}
            dfs_train_inner_scaled = {}
            dfs_val_inner_scaled = {}

            for name in dfs.keys():
                # 1. Get the raw splits for this specific fold
                df_tr = dfs_train_inner_raw[name].copy()
                df_val = dfs_val_inner_raw[name].copy()
                
                # 2. Identify feature columns
                feats = [c for c in df_tr.columns if c != "patient"]
                
                # 3. Fit scaler ONLY on the inner training set
                scaler = StandardScaler()
                if len(df_tr) > 0:
                    df_tr[feats] = scaler.fit_transform(df_tr[feats])
                    
                    # 4. Transform inner validation (and later the outer test) using TRAIN stats
                    if len(df_val) > 0:
                        df_val[feats] = scaler.transform(df_val[feats])
                        
                dfs_train_inner_scaled[name] = df_tr
                dfs_val_inner_scaled[name] = df_val
                inner_scalers[name] = scaler

            label_train_inner = filter_by_patients(label_train_outer, train_inner_ids)
            label_val_inner   = filter_by_patients(label_train_outer, val_inner_ids)

            train_dataset = DyAMDataset(dfs=dfs_train_inner_scaled, label_df=label_train_inner, label_col="PFS_6_label")
            val_dataset = DyAMDataset(dfs=dfs_val_inner_scaled, label_df=label_val_inner, label_col="PFS_6_label")

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dyam_collate, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dyam_collate)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, history, model_state = train_function_inner(train_loader, val_loader, device, input_dims, epochs, lr, outer_fold_counter, inner_fold_counter)
            all_histories.extend(history)
            inner_models.append(copy.deepcopy(model))
            # Store the model AND its specific scalers as a pair
            ensemble_inner_results.append({
                'inner_fold': inner_fold_counter,
                'model_state': model_state,
                'scalers': inner_scalers  # The scalers from THIS inner fold
            })

            # Evaluate model (committee so far)
            metrics = evaluate_metrics_ensemble_inner(inner_models, val_loader, device)
            metrics.update({'Fold': f'outer_{outer_fold_counter}_inner_{inner_fold_counter}'})
            inner_results.append(metrics)
            inner_fold_counter += 1  

        # --- OUTER EVALUATION ---
        print(f"Running ensemble outer fold {outer_fold_counter}")

        # Save the entire ensemble for this outer fold
        bundle = {
            'outer_fold': outer_fold_counter,
            'ensemble': ensemble_inner_results,
            'input_dims': input_dims
        }
        bundles.append(bundle)

        metrics = evaluate_metrics_ensemble_outer(
            ensemble_list=ensemble_inner_results, 
            raw_test_dfs=dfs_test_outer_raw, 
            label_test_df=label_test_outer,
            device=device,
            batch_size=batch_size,
            input_dims=input_dims
            )


        #test_dataset = DyAMDataset(dfs=dfs_test_outer, label_df=label_test_outer, label_col="PFS_6_label")
        #val_loader_outer = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=dyam_collate)

        #metrics = evaluate_metrics_ensemble(inner_models, val_loader_outer, device)
        metrics.update({'Fold': f'outer_{outer_fold_counter}'})
        outer_results.append(metrics)

        outer_fold_counter += 1
    return pd.DataFrame(inner_results), pd.DataFrame(outer_results), pd.DataFrame(all_histories), bundles



def train_function_inner(train_loader, val_loader, device, input_dims, epochs, lr, outer_idx, inner_idx):
    model = DyAM(input_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_model_state = None
    early_stop = 0
    patience = 10
    
    # List to store metrics for each epoch
    history = []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss, train_preds, train_targets = 0.0, [], []

        for Xs, y, _ in train_loader:
            Xs = [x.to(device) if x is not None else None for x in Xs]
            y = y.to(device)

            optimizer.zero_grad()
            outputs, _, _, _ = model(Xs)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()

            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * (0.01 / ((1 + epoch)**0.55))
                    param.grad.add_(noise)

            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.extend(y.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_auc = roc_auc_score(train_targets, train_preds) if len(set(train_targets)) > 1 else 0.5

        # --- Validation Phase ---
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for Xs, y, _ in val_loader:
                Xs = [x.to(device) if x is not None else None for x in Xs]
                y = y.to(device)
                outputs, _, _, _ = model(Xs)
                
                loss = criterion(outputs.squeeze(), y)
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds) if len(set(val_targets)) > 1 else 0.5
        scheduler.step(avg_val_loss)

        # Record metrics
        epoch_data = {
            'outer_fold': outer_idx,
            'inner_fold': inner_idx,
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_auc': train_auc,
            'val_auc': val_auc
        }
        history.append(epoch_data)
        
        print(f"E{epoch+1} | T-Loss: {avg_train_loss:.3f} | V-Loss: {avg_val_loss:.3f} | T-AUC: {train_auc:.2f} | V-AUC: {val_auc:.2f}")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            print(f"  ---> Validation loss improved: {best_val_loss:.4f} to {avg_val_loss:.4f}.")
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop = 0  # Reset patience counter
        else:
            early_stop += 1
            
            if early_stop >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # Load best weights before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model, history, best_model_state

