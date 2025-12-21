#!/usr/bin/env python

#################################################################################################
# Import module
#################################################################################################

#sys.path.append('..')  # To import from GP.kernels and property_predition.data_utils

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm 

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
import os, shutil
import random
import math
import time
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np

import crossover as co

#################################################################################################
# Main function
#################################################################################################

def main():

    start_time = time.time()

    # Define a random seed generator
    g = torch.Generator()
    g.manual_seed(88848)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load initial set 
    database = pd.read_csv("initpool.dat")[["SMILES"]]
    batch_size = 32
    
    # Load saved model
    gp_model = GaussianProcessRegression(kernel=Tanimoto_score, noise=1e-5)
    checkpoint = torch.load('gp_model.pth', map_location=device)
    
    gp_model.load_state_dict(checkpoint['model_state_dict'])
    gp_model.forward(checkpoint['x_train'], checkpoint['y_train'])

    scaler_mean = checkpoint['scaler_mean']
    scaler_std = checkpoint['scaler_std']
    
    gp_model.to(device)
    gp_model.eval()

    ############################
    ### parameters for GB-GA ###
    # GA parameters
    nstep = 10
    target_pool = 400
    target_value = 10
    
    # set range of number of heavy atoms in molecule
    co.average_size = 30
    co.size_stdev = 40
    co.string_type = 'SMILES'
    
    # gaussian noise range for selection
    gau_sigma = 0.001

    # mutation probability
    mu_prob = 0.3
    
    ############################

    # load reactions for mutations 
    with open("mutate_reaction.dat", "r") as f:
        lines = f.readlines()

    rxn_list = []
    for line in lines:
        rxn_list.append(line.rstrip())

    # initial genetic pool generation
    GenPool = []
    for elements in database["SMILES"]:
        GenPool.append(elements)

    # initialize log
    print("#cycle  cut/mean/std  total_pool(from selection+from crossover&mutation)  number_of_crossover  number_of_mutation", flush=True)
    
    # main generation cycle
    n0_time = start_time
    for istep in range(nstep):
        GenPool = GB_GA(gp_model, scaler_std, scaler_mean, batch_size, GenPool, istep, target_value, gau_sigma, target_pool, rxn_list, mu_prob, g, device) 

        n1_time = time.time()
        time_cost = n1_time - n0_time
        print(f"istep: {time_cost:.1f} seconds", flush=True)
        n0_time = n1_time

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.1f} seconds\n")

#################################################################################################
# Calculation functions
#################################################################################################

# convert SMILES strings to Morgan fingerprint
def from_SMILES_to_fp(smiles, radius=3, n_bits=2048, fp_type='morgan'):
    mol = Chem.MolFromSmiles(smiles)
    if mol :
        if fp_type == 'morgan':
            return torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol,radius, n_bits), dtype=torch.float32)
        if fp_type == 'MACCS':
            return torch.tensor(AllChem.GetMACCSKeysFingerprint(mol), dtype=torch.float32)
    return None

# gaussian process regression model by PyTorch
class GaussianProcessRegression(nn.Module) :
    def __init__(self, kernel, noise=1e-5) :
        super().__init__()
        self.kernel = kernel
        self.log_noise = nn.Parameter(torch.tensor(noise).log())
        self.K_inv = None

    # K: Training covariance matrix / K_c: Cross covariance matrix / K_inv: Inverse covariance matrix
    def forward(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
        K = self.kernel(x_train, x_train) + self.log_noise.exp() * torch.eye(len(x_train), device=x_train.device)
        self.K_inv = torch.inverse(K)

    def predict(self, x_test):
        K_c = self.kernel(self.x_train, x_test)
        pred_mean = torch.matmul(K_c.T, torch.matmul(self.K_inv, self.y_train))	
        return pred_mean

    def log_marginal_likelihood(self):
        K = self.kernel(self.x_train, self.x_train) + self.log_noise.exp() * torch.eye(len(self.x_train), device=self.x_train.device)
        
        L = torch.cholesky(K)
        alpha = torch.cholesky_solve(self.y_train.unsqueeze(1),L)
        log_likelihood = -0.5 * torch.sum(self.y_train.unsqueeze(1) * alpha)
        log_likelihood -= torch.sum(torch.log(torch.diagonal(L)))
        log_likelihood -= 0.5 * len(self.y_train) * torch.log(torch.tensor(2.0 * math.pi, device=self.y_train.device))
        
        return log_likelihood

    def optimize_lml(self, lr=0.01, max_iter=100):
        optimizer = optim.Adam([self.log_noise], lr=lr)
        for i in range(max_iter):
            optimizer.zero_grad()
            loss = -self.log_marginal_likelihood()
            loss.backward()
            optimizer.step()
        
            if i % 10 == 0:
                print(f"Step {i}: LML = {-loss.item():.3f}, Noise = {self.log_noise.exp().item():.5f}")

# Split dataset into training and test sets
def split_dataset(x, y, train_ratio, generator) :
    total_size = len(x)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    
    dataset = torch.utils.data.TensorDataset(x,y)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size,test_size], generator=generator)
    
    x_train, y_train = zip(*train_set)
    x_test, y_test = zip(*test_set)
    
    x_train = torch.stack(x_train)
    y_train = torch.stack(y_train)
    x_test = torch.stack(x_test)
    y_test = torch.stack(y_test)
    
    print(f'# of training set: {x_train.shape[0]} / # of test set: {x_test.shape[0]}')
    return x_train, y_train, x_test, y_test

# main GB_GA cycle
def GB_GA(gp_model, scaler_std, scaler_mean, batch_size, GenPool, istep, target_value, gau_sigma, target_pool, rxn_list, mu_prob, generator, device):
    
    # GPR model prediction
    fp_ligand = torch.stack([from_SMILES_to_fp(smiles, fp_type='morgan') for smiles in GenPool])
    y_pred_scaled = gp_model.predict(fp_ligand)
    y_pred = y_pred_scaled * scaler_std + scaler_mean

    # set criteria for cutting limit of fitness value 
    sort_a = []  # array for cutting criteria
    score = []

    for imol in range(len(y_pred)):
        gau_target = target_value + torch.normal(mean=0.0, std=gau_sigma, size=(1,), generator=generator).item()
        diff = abs(gau_target - y_pred[imol].item())
        score.append(diff)
        sort_a.append(diff)
    
    sort_a.sort()
    cut = sort_a[int(len(sort_a) * 0.2)] if sort_a else 0.0

    x_parents1 = []  # array for selected parent molecules
    x_parents2 = []  # array for offspring from crossover & mutation
    over_cut = f"High-score molecules based on the cutoff : {cut:9.5f}\n"
    less_cut = f"\nLow-score molecules based on the cutoff : {cut:9.5f}\n"
    mol_list = []  # list for analysis of high-score molecules
    
    #### Selection ####

    # for step 1, selection is not required
    if istep == 0:
        x_parents1 = GenPool
    else:
        for imol in range(len(GenPool)):
            mol = Chem.MolFromSmiles(GenPool[imol])
            rand = random.random()

            # selection based on cutting criteria
            if score[imol] < cut:
                x_parents1.append(GenPool[imol])
                over_cut += f"{imol:8.0f} th molecule: {GenPool[imol]} , {y_pred[imol].item():9.5f}\n"
                mol_list.append(y_pred[imol].item())
            else:
                less_cut += f"{imol:8.0f} th molecule: {GenPool[imol]} , {y_pred[imol].item():9.5f}\n"
                # Select 20% in 80% of undesired molecules
                if rand >= 0.8:
                    x_parents1.append(GenPool[imol])
        # save step information
        with open(f"steps/INDEX_{istep}.dat", "w") as f:
            f.write(over_cut)
            f.write(less_cut)
    
    mol_tensor = torch.tensor(mol_list, dtype=torch.float)
    mean_val = mol_tensor.mean().item()
    std_val = mol_tensor.std().item()
   
    #### Crossover & Mutation ####
    ncross = 0  # Crossover count
    nmut = 0    # Mutation count
    need_offspring = target_pool - len(x_parents1)

    while ncross < need_offspring:
        x_tmp1 = random.choice(x_parents1)
        x_tmp2 = random.choice(x_parents1)
    
        mol1 = Chem.MolFromSmiles(x_tmp1)
        mol2 = Chem.MolFromSmiles(x_tmp2)
    
        # crossover
        child = co.crossover(mol1, mol2)
        if child is None:
            continue
    
        l_mut = False
        # mutation 
        if random.random() > mu_prob:
            l_mut = True

            # add all possible mutations into mutate_candidates
            mutate_candidates = []
            for reaction in rxn_list:
                rxn = AllChem.ReactionFromSmarts(reaction)
                mutate_mols = rxn.RunReactants((child,))
                for mols in mutate_mols:
                    mutate_candidates.append(mols[0])
            if len(mutate_candidates) == 0:
                continue

            # random selection from mutation candidates
            mutated_child = np.random.choice(mutate_candidates)
            child = mutated_child

        # scaffold screening
        scaffold_smiles1 = "O=C(N)c1noc(c2c([OH])[cH]c([OH])c(C([CH3])[CH3])[cH]2)c1Nc3ccccc3"
        core_mol1 = Chem.MolFromSmarts(scaffold_smiles1)
        scaffold_smiles2 = "O=C(N)c1n[nH]c(c2c([OH])[cH]c([OH])c(C([CH3])[CH3])[cH]2)c1Nc3ccccc3"
        core_mol2 = Chem.MolFromSmarts(scaffold_smiles2)
        scaffold_smiles3 = "O=C(N)c1noc(c2c([OH])[cH]c([OH])c(C([CH3])[CH3])[cH]2)c1Nc3ccccc3"
        core_mol3 = Chem.MolFromSmarts(scaffold_smiles3)
        scaffold_smiles4 = "O=C(N)c1n[nH]c(c2c([OH])[cH]c([OH])c(C([CH3])[CH3])[cH]2)c1Nc3ccccc3"
        core_mol4 = Chem.MolFromSmarts(scaffold_smiles4)

        child = Chem.MolFromSmiles(Chem.MolToSmiles(child), sanitize=True)

        if child is None:
            continue

        # if the molecule does not have scaffold structure, reject it
        if (child.HasSubstructMatch(core_mol1)==False and child.HasSubstructMatch(core_mol2)==False and  child.HasSubstructMatch(core_mol3)==False and child.HasSubstructMatch(core_mol4)==False):
            continue

        cano_child = Chem.MolToSmiles(child, True)
        cano_child_fp = AllChem.GetMorganFingerprintAsBitVect(child, 3, 2048)  # child, radius, bit
        
        # if the molecular size is too large, reject it
        if (child.GetNumAtoms() > 41):
             continue

        # if there is similar molecule in the genetic pool, reject it
        l_simcheck = False
        for molecules in x_parents1:
            mol_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(molecules), 3, 2048)
            sim = DataStructs.TanimotoSimilarity(cano_child_fp, mol_fp) 
            if sim > (0.8 - istep*0.01):
                l_simcheck = True
                continue

        for molecules in x_parents2:
            mol_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(molecules), 3, 2048)
            sim = DataStructs.TanimotoSimilarity(cano_child_fp, mol_fp) 
            if sim > (0.8 - istep*0.01):
                l_simcheck = True
                continue
        if l_simcheck == True:
            continue

        ncross += 1
        if l_mut:
            nmut += 1
        x_parents2.append(cano_child)
    
    # remove duplicate molecules from the pool
    GenPool = x_parents1 + x_parents2
    GenPool_canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), True) for smi in GenPool]
    
    GenPool_remove_rep = []
    for elements in GenPool_canonical:
        if elements not in GenPool_remove_rep:
            GenPool_remove_rep.append(elements)
    
    repetition = len(GenPool) - len(GenPool_remove_rep)
    GenPool = GenPool_remove_rep
   
    # log for current step 
    if istep != 0:
        print(f"{(istep+1):6.0f}  {cut:10.5f} / {mean_val:8.5f} / {std_val:8.5f}  {len(GenPool):6.0f} ({len(x_parents1):6.0f} + {len(x_parents2):6.0f}) {ncross:6.0f} {nmut:6.0f}", flush=True)
    
    return GenPool

# function to convert data to PyTorch tensors
def get_tensor_data(data):
    """
    Convert data to PyTorch tensors.
    
    Parameters:
        data (DataFrame): Input data containing 'input_ids', 'attention_mask', and 'pIC50' columns.
    
    Returns:
        TensorDataset containing input_ids, attention_mask, and labels tensors.
    """
    input_ids_tensor = torch.tensor(data["input_ids"].tolist(), dtype=torch.long)
    attention_mask_tensor = torch.tensor(data["attention_mask"].tolist(), dtype=torch.long)

    return TensorDataset(input_ids_tensor, attention_mask_tensor)

# define Tanimoto kernel
def Tanimoto_score(x1, x2) :
    device = x1.device
    x2 = x2.to(device)
    
    dot_product = torch.matmul(x1, x2.T)
    norm_x1 = torch.sum(x1**2, dim=1).reshape(-1,1)
    norm_x2 = torch.sum(x2**2, dim=1).reshape(1,-1)
    return dot_product / (norm_x1 + norm_x2 - dot_product)


#################################################################################################
# Why python using this? :P
#################################################################################################

if __name__ == "__main__":
    # Initialize output directory
    path = "./steps"
    if (os.path.exists(path)):
        if (os.path.exists(path + "_old")):
            shutil.rmtree(path + "_old")
        shutil.move(path, path + "_old")
    os.makedirs(path)

    main()

