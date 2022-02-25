### Utility functions to find the maximum treatment effect phenotype and mean differential survival
import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# def predict_treatment_phenotype_proba(model, x, a):
#     """
#     Using the trained CMHE model, find the probability that an individual 
#     belongs to each treatment phenotype.
#     """
    
#     # Find the probability that an individual belongs to each treatment phenotype
#     zeta_probs = torch.exp(model.torch_model[0](
#             torch.from_numpy(x), torch.from_numpy(a))[0]).sum(dim=1).detach().numpy()
   
#     return zeta_probs

def find_max_treatment_effect_phenotype(g, zeta_probs, factual_outcomes):
    """
    Find the group with the maximum treatement effect phenotype
    """
    mean_differential_survival = np.zeros(zeta_probs.shape[1]) # Area under treatment phenotype group
    outcomes_train, interventions_train = factual_outcomes 

    # Assign each individual to their treatment phenotype group
    for gr in range(g): # For each treatment phenotype group
        # Probability of belonging the the g^th treatment phenotype
        zeta_probs_g = zeta_probs[:, gr] 
        # Consider only those individuals who are in the top 75 percentiles in this phenotype
        z_mask = zeta_probs_g>np.quantile(zeta_probs_g, 0.75) 

        mean_differential_survival[gr] = find_mean_differential_survival(
            outcomes_train.loc[z_mask], interventions_train.loc[z_mask]) 

    return np.nanargmax(mean_differential_survival)

def find_mean_differential_survival(outcomes, interventions):
    """
    Given outcomes and interventions, find the maximum restricted mean survival time
    """
    from lifelines import KaplanMeierFitter

    treated_km = KaplanMeierFitter().fit(outcomes['uncensored time treated'].values, np.ones(len(outcomes)).astype(bool))
    control_km = KaplanMeierFitter().fit(outcomes['uncensored time control'].values, np.ones(len(outcomes)).astype(bool))

    unique_times = treated_km.survival_function_.index.values.tolist() + control_km.survival_function_.index.values.tolist()  
    unique_times = np.unique(unique_times)

    treated_km = treated_km.predict(unique_times, interpolate=True)
    control_km = control_km.predict(unique_times, interpolate=True)

    mean_differential_survival = np.trapz(y=(treated_km.values - control_km.values),
                                        x=unique_times)

    return mean_differential_survival

def plot_phenotypes_roc(outcomes, zeta_probs):
    zeta = outcomes['Zeta']

    y_true = zeta == 0

    fpr, tpr, thresholds = roc_curve(y_true, zeta_probs)
    auc = roc_auc_score(y_true, zeta_probs) 

    plt.figure(figsize=(8,6))

    plt.plot(fpr, tpr, label="AUC:"+str(round(auc,3)))
    plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), ls='--', color='k')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14) 

    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.xscale('log')
    plt.show()