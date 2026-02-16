import numpy as np
import pandas as pd
from scipy.stats import chi2
import shap
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, average_precision_score, accuracy_score
import json
import torch
from torch.utils.data import DataLoader
from models import DyAM
from dataset import DyAMDataset, dyam_collate



def SHAP_values(best_classifier, data_set):
    explainer = shap.Explainer(best_classifier, data_set)
    shap_values = explainer(data_set)
    global_importance = np.mean(np.abs(shap_values.values), axis = 0)

    return global_importance

def evaluate_metrics(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_preds_proba = []
    all_true = []
    all_patients = []

    all_attention_weights = []  # To store attention weights from validation
    with torch.no_grad():
        for data in val_loader:
            modality_tensors, y_val = data  # Assuming your DataLoader returns a tuple (modality tensors, targets, patient ids)
            
            # Ensure modality_tensors is a list of tensors
            modality_tensors = [tensor.to(device) if tensor is not None else None for tensor in modality_tensors]
            
            # Forward pass with the model, which should now return attention weights too
            outputs, attn_share, risk_scores = model(modality_tensors)
            
            # Convert logits to probabilities
            y_pred_proba = torch.sigmoid(outputs).cpu().numpy()

            # Convert probabilities to binary predictions
            y_pred_binary = (y_pred_proba >= 0.5).astype(int)

            # Store predictions, ground truth, and attention weights
            all_preds_proba.extend(y_pred_proba)
            all_preds.extend(y_pred_binary)
            all_true.extend(y_val.numpy())
            #all_patients.extend(patient_ids.numpy())  # Collect patient identifiers
            all_attention_weights.extend(list(attn_share.cpu().numpy()))  # Store attention weights

    # Convert lists to numpy arrays for metric calculations
    y_true = np.array(all_true)
    y_pred_binary = np.array(all_preds)
    y_pred_proba = np.array(all_preds_proba)
    #patients = np.array(all_patients)

    # Compute metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    aucpr = average_precision_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred_binary)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    sen = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (Recall)
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision (Positive Predictive Value)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    mcc = matthews_corrcoef(y_true, y_pred_binary)

    # Prepare results
    results = {
        'AUC': auc,
        'AUCPR': aucpr,
        'ACC': acc,
        'SEN': sen,
        'SP': sp,
        'PPV': ppv,
        'NPV': npv,
        'MCC': mcc,
        'Predicted_Probas': y_pred_proba.tolist(),
        'Binary_Preds': y_pred_binary.tolist(),
        #'Patients': list(patients),  # Include patient identifiers
        'Attention weights': all_attention_weights # Include attention weights
    }

    return results

class NumpyEncoder(json.JSONEncoder):  #we need this for xgboost results. Error "object of type float32 is not JSON serializable"
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)
    
def HosmerLemeshow(data, predictions, Q=10):
    """Hosmer-Lemeshow goodness of fit test
    Parameters
    ----------
    data : Dataframe
        Dataframe containing :
            - the effective of each group n
            - for each group, the number of cases where c = 1, n1
    predictions : Dataframe
        Dataframe containing :
            - for each group, the predicted proportion of c = 1 according to a logistic regression, p1
    Q : int, optional
        The number of groups
    Returns
    -------
    result : the result of the test, including Chi2-HL statistics and p-value 
    """

    data.loc[:, "expected_prop"] = predictions.p1
    data = data.sort_values(by="expected_prop", ascending=False)

    categories, bins = pd.cut(
        data["expected_prop"],
        np.percentile(data["expected_prop"], np.linspace(0, 100, Q + 1)),
        labels=False,
        include_lowest=True,
        retbins=True,
    )

    meanprobs1 = np.zeros(Q)
    expevents1 = np.zeros(Q)
    obsevents1 = np.zeros(Q)
    meanprobs0 = np.zeros(Q)
    expevents0 = np.zeros(Q)
    obsevents0 = np.zeros(Q)

    y0 = data.n - data.n1
    y1 = data.n1
    n = data.n
    expected_prop = predictions.p1
    for i in range(Q):

        meanprobs1[i] = np.mean(expected_prop[categories == i])
        expevents1[i] = np.sum(n[categories == i]) * np.array(meanprobs1[i])
        obsevents1[i] = np.sum(y1[categories == i])
        meanprobs0[i] = np.mean(1 - expected_prop[categories == i])
        expevents0[i] = np.sum(n[categories == i]) * np.array(meanprobs0[i])
        obsevents0[i] = np.sum(y0[categories == i])

    data1 = {"meanprobs1": meanprobs1, "meanprobs0": meanprobs0}
    data2 = {"expevents1": expevents1, "expevents0": expevents0}
    data3 = {"obsevents1": obsevents1, "obsevents0": obsevents0}
    m = pd.DataFrame(data1)
    e = pd.DataFrame(data2)
    o = pd.DataFrame(data3)

    chisq_value = sum(sum((np.array(o) - np.array(e)) ** 2 / np.array(e)))
    pvalue = 1 - chi2.cdf(chisq_value, Q - 2)

    result = pd.DataFrame(
        [[Q - 2, chisq_value.round(2), pvalue.round(2)]],
        columns=["df", "Chi2", "p - value"],
    )
    return result



def evaluate_metrics_ensemble_inner(inner_models, val_loader, device):
    model_ensemble_probas = []
    model_ensemble_attentions = []
    model_ensemble_shares = []
    model_ensemble_risks = []
    y_true = []
    all_pids = []

    for idx, model in enumerate(inner_models):
        model.eval()
        current_model_probas = []
        current_model_attentions = []
        current_model_shares = []
        current_model_risks = []

        with torch.no_grad():
            for data in val_loader:
                modality_tensors, y_val, pids = data
                modality_tensors = [t.to(device) if t is not None else None for t in modality_tensors]
                
                # Unpack the 4 items: output, attention_weights (a), risks (R), and attention_share (alpha)
                # Note: Adjust order if your forward returns alpha second and risks third
                outputs, a_weights, a_shares, risks = model(modality_tensors)
                
                current_model_probas.extend(torch.sigmoid(outputs).cpu().numpy())
                current_model_attentions.extend(a_weights.cpu().numpy())
                current_model_shares.extend(a_shares.cpu().numpy())
                current_model_risks.extend(risks.cpu().numpy())
                
                if idx == 0:
                    y_true.extend(y_val.cpu().numpy())
                    all_pids.extend(pids)

        model_ensemble_probas.append(np.array(current_model_probas))
        model_ensemble_attentions.append(np.array(current_model_attentions))
        model_ensemble_shares.append(np.array(current_model_shares))
        model_ensemble_risks.append(np.array(current_model_risks))

    y_pred_proba = np.mean(model_ensemble_probas, axis=0)
    avg_attention_weights = np.mean(model_ensemble_attentions, axis=0)
    avg_shares = np.mean(model_ensemble_shares, axis=0)
    avg_risks = np.mean(model_ensemble_risks, axis=0)
    
    y_true = np.array(y_true)
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    results = {
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'AUCPR': average_precision_score(y_true, y_pred_proba),
        'ACC': accuracy_score(y_true, y_pred_binary),
        'SEN': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'SP': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'MCC': matthews_corrcoef(y_true, y_pred_binary),
        'Predicted_Probas': y_pred_proba.flatten().tolist(),
        'Avg_Attention_Weights': avg_attention_weights.tolist(),
        'Avg_Attention_Shares': avg_shares.tolist(),
        'Avg_Risks': avg_risks.tolist(),
        'True_Labels': y_true.tolist(),
        'Patient_IDs': all_pids 
    }
    
    results.update({
        'Global_Attention_Mean': np.mean(avg_attention_weights, axis=0).tolist(),
        'Global_Attention_Std': np.std(avg_attention_weights, axis=0).tolist(),
        'Global_Share_Mean': np.mean(avg_shares, axis=0).tolist(),
        'Global_Share_Std': np.std(avg_shares, axis=0).tolist(),
        'Global_Risk_Mean': np.mean(avg_risks, axis=0).tolist()
    })

    return results

def evaluate_metrics_ensemble_outer(ensemble_list, raw_test_dfs, label_test_df, device, batch_size, input_dims):
    model_ensemble_probas = []
    model_ensemble_attentions = []
    model_ensemble_shares = []
    model_ensemble_risks = []
    y_true = []
    all_pids = []

    model = DyAM(input_dims).to(device)

    for idx, member in enumerate(ensemble_list):
        # A. Load weights for this specific inner model
        model.load_state_dict(member['model_state'])
        model.eval()

        # B. Scale the raw test data using THIS model's specific scalers
        scaled_test_dfs = {}
        for name, scaler in member['scalers'].items():
            df_copy = raw_test_dfs[name].copy()
            feats = [c for c in df_copy.columns if c != "patient"]
            if len(df_copy) > 0:
                df_copy[feats] = scaler.transform(df_copy[feats])
            scaled_test_dfs[name] = df_copy

        # C. Create a temporary loader for this scaled version
        temp_dataset = DyAMDataset(dfs=scaled_test_dfs, label_df=label_test_df, label_col="PFS_6_label")
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, collate_fn=dyam_collate)

        current_model_probas = []
        current_model_attentions = []
        current_model_shares = []
        current_model_risks = []

        with torch.no_grad():
            for data in temp_loader:
                modality_tensors, y_val, pids = data
                modality_tensors = [t.to(device) if t is not None else None for t in modality_tensors]
                
                outputs, a_weights, a_shares, risks = model(modality_tensors)
                current_model_probas.extend(torch.sigmoid(outputs).cpu().numpy())
                current_model_attentions.extend(a_weights.cpu().numpy())
                current_model_shares.extend(a_shares.cpu().numpy())
                current_model_risks.extend(risks.cpu().numpy())
                if idx == 0: # Only need labels and IDs once
                    y_true.extend(y_val.cpu().numpy())
                    all_pids.extend(pids)

        model_ensemble_probas.append(np.array(current_model_probas))
        model_ensemble_attentions.append(np.array(current_model_attentions))
        model_ensemble_shares.append(np.array(current_model_shares))
        model_ensemble_risks.append(np.array(current_model_risks))

    y_pred_proba = np.mean(model_ensemble_probas, axis=0)
    avg_attention_weights = np.mean(model_ensemble_attentions, axis=0)
    avg_shares = np.mean(model_ensemble_shares, axis=0)
    avg_risks = np.mean(model_ensemble_risks, axis=0)
    
    y_true = np.array(y_true)
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    results = {
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'AUCPR': average_precision_score(y_true, y_pred_proba),
        'ACC': accuracy_score(y_true, y_pred_binary),
        'SEN': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'SP': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'MCC': matthews_corrcoef(y_true, y_pred_binary),
        'Predicted_Probas': y_pred_proba.flatten().tolist(),
        'Avg_Attention_Weights': avg_attention_weights.tolist(),
        'Avg_Attention_Shares': avg_shares.tolist(),
        'Avg_Risks': avg_risks.tolist(),
        'True_Labels': y_true.tolist(),
        'Patient_IDs': all_pids 
    }
    
    results.update({
        'Global_Attention_Mean': np.mean(avg_attention_weights, axis=0).tolist(),
        'Global_Attention_Std': np.std(avg_attention_weights, axis=0).tolist(),
        'Global_Share_Mean': np.mean(avg_shares, axis=0).tolist(),
        'Global_Share_Std': np.std(avg_shares, axis=0).tolist(),
        'Global_Risk_Mean': np.mean(avg_risks, axis=0).tolist()
    })

    return results

