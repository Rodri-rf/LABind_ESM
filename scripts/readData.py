import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import nn_config
data_class = nn_config['pdb_class']

class readData(Dataset):  # For training
    def __init__(self, name_list, proj_dir, lig_dict, true_file, mode):
        self.label_dict = self._read(true_file, skew=1)
        if name_list is not None:
            self.name_list = name_list
        self.proj_dir = proj_dir
        self.lig_dict = lig_dict
        self.mode = mode
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        try:
            name, lig = self.name_list[idx]
            
            # Load DSSP
            dssp = self.Normalize(
                np.load(f'{self.proj_dir}/{data_class}_dssp/{self.mode}/{name}.npy'),
                nn_config[f'dssp_max_repr'],
                nn_config[f'dssp_min_repr']
            )
            
            # Load ESM (from esm3B directory)
            esm = self.Normalize(
                np.load(f'{self.proj_dir}/esm3B/{name}.npy'),
                nn_config[f'esm2_max_repr'],
                nn_config[f'esm2_min_repr']
            )
            
            # Check shape compatibility BEFORE concatenation
            if dssp.shape[0] != esm.shape[0]:
                print(f"Shape mismatch for {name}:")
                print(f"  DSSP: {dssp.shape}")
                print(f"  ESM: {esm.shape}")
                # Try to fix by truncating to minimum length
                min_len = min(dssp.shape[0], esm.shape[0])
                dssp = dssp[:min_len]
                esm = esm[:min_len]
                print(f"  Truncated to length: {min_len}")
            
            # Concatenate
            feature = np.concatenate([dssp, esm], axis=1)
            
            # Load other data
            ligand = self.Normalize(
                self.lig_dict[lig],
                nn_config[f'ion_max_repr'],
                nn_config[f'ion_min_repr']
            )
            xyz = np.load(f'{self.proj_dir}/{data_class}_pos/{self.mode}/{name}.npy')
            y_true = np.asarray(list(self.label_dict[(name, lig)]), dtype=int)
            
            # Final consistency check
            if not (feature.shape[0] == xyz.shape[0] == y_true.shape[0]):
                print(f"Length mismatch after concatenation for {name}:")
                print(f"  feature: {feature.shape[0]}")
                print(f"  xyz: {xyz.shape[0]}")
                print(f"  y_true: {y_true.shape[0]}")
                # Truncate all to minimum
                min_len = min(feature.shape[0], xyz.shape[0], y_true.shape[0])
                feature = feature[:min_len]
                xyz = xyz[:min_len]
                y_true = y_true[:min_len]
                print(f"  Truncated all to: {min_len}")
            
            return feature, ligand, xyz, y_true
            
        except Exception as e:
            print(f"ERROR loading {self.name_list[idx]}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def Normalize(self, arr, max_value, min_value):
        scalar = max_value - min_value
        scalar[scalar == 0] = 1
        return (arr - min_value) / scalar
    
    def collate_fn(self, batch):
        # CRITICAL FIX: Filter out None entries
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            print("WARNING: Empty batch after filtering None entries")
            return None
        
        features, ligands, xyzs, y_trues = zip(*batch)
        maxlen = max(1500, max([f.shape[0] for f in features]))
        
        batch_feat = []
        batch_ligand = []
        batch_xyz = []
        batch_mask = []
        batch_y_true = []
        
        for idx in range(len(batch)):
            batch_feat.append(self._padding(features[idx], maxlen))
            batch_ligand.append(torch.tensor(ligands[idx], dtype=torch.float))
            batch_xyz.append(self._padding(xyzs[idx], maxlen))
            
            mask = np.zeros(maxlen)
            mask[:features[idx].shape[0]] = 1
            batch_mask.append(torch.tensor(mask, dtype=torch.long))
            
            pad_y = np.zeros(maxlen)
            pad_y[:y_trues[idx].shape[0]] = y_trues[idx]
            batch_y_true.append(torch.tensor(pad_y, dtype=torch.float))
        
        return (torch.stack(batch_feat), torch.stack(batch_ligand),
                torch.stack(batch_xyz), torch.stack(batch_mask),
                torch.stack(batch_y_true))
    
    def _padding(self, arr, maxlen=1500):
        padded = np.zeros((maxlen, *arr.shape[1:]), dtype=np.float32)
        padded[:arr.shape[0]] = arr
        res = torch.tensor(padded, dtype=torch.float)
        return res
    
    def _read(self, file_name, skew=0):
        lab_dict = {}
        with open(file_name, 'r') as file:
            content = file.readlines()
            lens = len(content)
            for idx in range(lens)[::2 + skew]:
                name = content[idx].replace('>', '').replace('\n', '')
                id, lig = name.split(' ')[0], name.split(' ')[1]
                lab = content[idx + 1 + skew].replace('\n', '')
                lab_dict[(id, lig)] = lab
        return lab_dict


class LoadData(Dataset):  # For testing
    def __init__(self, name_list, proj_dir, lig_dict, repr_dict):
        self.name_list = name_list
        self.proj_dir = proj_dir
        self.repr_dict = repr_dict
        self.lig_dict = lig_dict
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        try:
            name, lig = self.name_list[idx]
            
            # Load DSSP
            dssp = self.Normalize(
                np.load(f'{self.proj_dir}/dssp/{name}.npy'),
                self.repr_dict['dssp_max_repr'],
                self.repr_dict['dssp_min_repr']
            )
            
            # Load ESM
            esm = self.Normalize(
                np.load(f'{self.proj_dir}/esm2/{name}.npy'),
                self.repr_dict['esm2_max_repr'],
                self.repr_dict['esm2_min_repr']
            )
            
            # Check shape compatibility
            if dssp.shape[0] != esm.shape[0]:
                print(f"Shape mismatch for {name}:")
                print(f"  DSSP: {dssp.shape}, ESM: {esm.shape}")
                min_len = min(dssp.shape[0], esm.shape[0])
                dssp = dssp[:min_len]
                esm = esm[:min_len]
                print(f"  Truncated to: {min_len}")
            
            feature = np.concatenate([dssp, esm], axis=1)
            xyz = np.load(os.path.join(self.proj_dir, 'pos', name + '.npy'))
            ligand = self.Normalize(
                self.lig_dict[lig],
                self.repr_dict['ion_max_repr'],
                self.repr_dict['ion_min_repr']
            )
            
            return name, lig, feature, ligand, xyz
            
        except Exception as e:
            print(f"ERROR loading {self.name_list[idx]}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def Normalize(self, arr, max_value, min_value):
        scalar = max_value - min_value
        scalar[scalar == 0] = 1
        return (arr - min_value) / scalar
    
    def collate_fn(self, batch):
        # Filter out None entries
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            print("WARNING: Empty batch after filtering None entries")
            return None
        
        names, ligs, features, ligands, xyzs = zip(*batch)
        maxlen = 1500
        
        batch_rfeat = []
        batch_ligand = []
        batch_xyz = []
        batch_mask = []
        
        for idx in range(len(batch)):
            batch_rfeat.append(self._padding(features[idx], maxlen))
            batch_ligand.append(torch.tensor(ligands[idx], dtype=torch.float))
            batch_xyz.append(self._padding(xyzs[idx], maxlen))
            
            mask = np.zeros(maxlen)
            mask[:features[idx].shape[0]] = 1
            batch_mask.append(torch.tensor(mask, dtype=torch.long))
        
        return (names, ligs, torch.stack(batch_rfeat), torch.stack(batch_ligand),
                torch.stack(batch_xyz), torch.stack(batch_mask))
    
    def _padding(self, arr, maxlen=1500):
        padded = np.zeros((maxlen, *arr.shape[1:]), dtype=np.float32)
        padded[:arr.shape[0]] = arr
        res = torch.tensor(padded, dtype=torch.float)
        return res