# predict_hackathon.py
import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional

import yaml
from hackathon_api import Datapoint, Protein, SmallMolecule

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import warnings

class ProteinLigandFeatureExtractor:
    """Extract physical features from protein-ligand complexes for pose prediction"""
    
    def __init__(self, pdb_file: str):
        self.pdb_file = pdb_file
        self.protein_atoms = []
        self.ligand_atoms = []
        self.protein_residues = []
        
    def parse_pdb(self):
        """Parse PDB file to extract protein and ligand atoms"""
        with open(self.pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_info = self._parse_atom_line(line, is_protein=True)
                    self.protein_atoms.append(atom_info)
                    
                    # Track unique residues
                    res_id = (atom_info['chain'], atom_info['residue'], atom_info['res_num'])
                    if res_id not in [r['id'] for r in self.protein_residues]:
                        self.protein_residues.append({
                            'id': res_id,
                            'chain': atom_info['chain'],
                            'residue': atom_info['residue'],
                            'res_num': atom_info['res_num']
                        })
                        
                elif line.startswith('HETATM'):
                    atom_info = self._parse_atom_line(line, is_protein=False)
                    self.ligand_atoms.append(atom_info)
        
    def _parse_atom_line(self, line: str, is_protein: bool) -> Dict:
        """Parse ATOM or HETATM line"""
        return {
            'atom_num': int(line[6:11].strip()),
            'atom_name': line[12:16].strip(),
            'residue': line[17:20].strip(),
            'chain': line[21].strip(),
            'res_num': int(line[22:26].strip()),
            'x': float(line[30:38].strip()),
            'y': float(line[38:46].strip()),
            'z': float(line[46:54].strip()),
            'element': line[76:78].strip() if len(line) > 76 else line[12:16].strip()[0],
            'is_protein': is_protein
        }
    
    def calculate_distance(self, atom1: Dict, atom2: Dict) -> float:
        """Calculate Euclidean distance between two atoms"""
        return np.sqrt(
            (atom1['x'] - atom2['x'])**2 +
            (atom1['y'] - atom2['y'])**2 +
            (atom1['z'] - atom2['z'])**2
        )
    
    def extract_distance_features(self) -> Dict:
        """Extract distance-based features"""
        features = {
            'min_distance': float('inf'),
            'mean_distance': 0,
            'distances_under_4A': 0,
            'distances_under_5A': 0,
            'distances_under_6A': 0,
        }
        
        distances = []
        for lig_atom in self.ligand_atoms:
            for prot_atom in self.protein_atoms:
                dist = self.calculate_distance(lig_atom, prot_atom)
                distances.append(dist)
                
                if dist < features['min_distance']:
                    features['min_distance'] = dist
                    
                if dist < 4.0:
                    features['distances_under_4A'] += 1
                if dist < 5.0:
                    features['distances_under_5A'] += 1
                if dist < 6.0:
                    features['distances_under_6A'] += 1
        
        if distances:
            features['mean_distance'] = np.mean(distances)
            features['std_distance'] = np.std(distances)
            features['median_distance'] = np.median(distances)
        
        return features
    
    def extract_contact_features(self, cutoff: float = 4.5) -> Dict:
        """Extract contact features within cutoff distance"""
        features = {
            'total_contacts': 0,
            'contacts_per_ligand_atom': 0,
            'contacting_residues': set(),
            'hydrophobic_contacts': 0,
            'polar_contacts': 0,
        }
        
        hydrophobic_residues = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'}
        polar_residues = {'SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN'}
        
        for lig_atom in self.ligand_atoms:
            for prot_atom in self.protein_atoms:
                dist = self.calculate_distance(lig_atom, prot_atom)
                
                if dist < cutoff:
                    features['total_contacts'] += 1
                    features['contacting_residues'].add(
                        (prot_atom['chain'], prot_atom['res_num'], prot_atom['residue'])
                    )
                    
                    if prot_atom['residue'] in hydrophobic_residues:
                        features['hydrophobic_contacts'] += 1
                    elif prot_atom['residue'] in polar_residues:
                        features['polar_contacts'] += 1
        
        features['contacts_per_ligand_atom'] = (
            features['total_contacts'] / len(self.ligand_atoms) 
            if self.ligand_atoms else 0
        )
        features['num_contacting_residues'] = len(features['contacting_residues'])
        
        return features
    
    def extract_hydrogen_bond_features(self) -> Dict:
        """Estimate potential hydrogen bonds"""
        features = {
            'potential_hbonds': 0,
        }
        
        donor_elements = {'N', 'O'}
        acceptor_elements = {'N', 'O'}
        hbond_cutoff = 3.5
        
        for lig_atom in self.ligand_atoms:
            if lig_atom['element'] in donor_elements or lig_atom['element'] in acceptor_elements:
                for prot_atom in self.protein_atoms:
                    if prot_atom['element'] in donor_elements or prot_atom['element'] in acceptor_elements:
                        dist = self.calculate_distance(lig_atom, prot_atom)
                        
                        if dist < hbond_cutoff:
                            features['potential_hbonds'] += 1
        
        return features
    
    def extract_geometric_features(self) -> Dict:
        """Extract geometric features of the binding site"""
        features = {}
        
        # Ligand center of mass
        lig_coords = np.array([[a['x'], a['y'], a['z']] for a in self.ligand_atoms])
        lig_com = np.mean(lig_coords, axis=0)
        
        # Find binding site residues (within 8Å of ligand COM)
        binding_site_atoms = []
        for prot_atom in self.protein_atoms:
            prot_coord = np.array([prot_atom['x'], prot_atom['y'], prot_atom['z']])
            dist = np.linalg.norm(prot_coord - lig_com)
            if dist < 8.0:
                binding_site_atoms.append(prot_atom)
        
        if binding_site_atoms:
            binding_coords = np.array([[a['x'], a['y'], a['z']] for a in binding_site_atoms])
            binding_com = np.mean(binding_coords, axis=0)
            
            features['ligand_binding_site_distance'] = np.linalg.norm(lig_com - binding_com)
            features['binding_site_size'] = len(binding_site_atoms)
            
            features['pocket_span_x'] = np.max(binding_coords[:, 0]) - np.min(binding_coords[:, 0])
            features['pocket_span_y'] = np.max(binding_coords[:, 1]) - np.min(binding_coords[:, 1])
            features['pocket_span_z'] = np.max(binding_coords[:, 2]) - np.min(binding_coords[:, 2])
            features['pocket_volume_approx'] = (
                features['pocket_span_x'] * 
                features['pocket_span_y'] * 
                features['pocket_span_z']
            )
        
        # Ligand size features
        features['ligand_span_x'] = np.max(lig_coords[:, 0]) - np.min(lig_coords[:, 0])
        features['ligand_span_y'] = np.max(lig_coords[:, 1]) - np.min(lig_coords[:, 1])
        features['ligand_span_z'] = np.max(lig_coords[:, 2]) - np.min(lig_coords[:, 2])
        features['ligand_volume_approx'] = (
            features['ligand_span_x'] * 
            features['ligand_span_y'] * 
            features['ligand_span_z']
        )
        
        return features
    
    def extract_atom_type_features(self) -> Dict:
        """Extract features based on atom types"""
        features = {
            'ligand_heavy_atoms': len(self.ligand_atoms),
            'ligand_carbon_count': 0,
            'ligand_nitrogen_count': 0,
            'ligand_oxygen_count': 0,
            'ligand_sulfur_count': 0,
            'ligand_other_count': 0,
        }
        
        for atom in self.ligand_atoms:
            elem = atom['element'].upper()
            if elem == 'C':
                features['ligand_carbon_count'] += 1
            elif elem == 'N':
                features['ligand_nitrogen_count'] += 1
            elif elem == 'O':
                features['ligand_oxygen_count'] += 1
            elif elem == 'S':
                features['ligand_sulfur_count'] += 1
            else:
                features['ligand_other_count'] += 1
        
        return features
    
    def extract_all_features(self, verbose: bool = False) -> Dict:
        """Extract all features"""
        self.parse_pdb()
        
        features = {}
        
        features.update(self.extract_distance_features())
        contact_features = self.extract_contact_features()
        contact_features.pop('contacting_residues', None)  # Remove set for serialization
        features.update(contact_features)
        
        hbond_features = self.extract_hydrogen_bond_features()
        features['potential_hbonds'] = hbond_features['potential_hbonds']
        
        features.update(self.extract_geometric_features())
        features.update(self.extract_atom_type_features())
        
        if verbose:
            print("\n=== Feature Summary ===")
            for key, value in features.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
        
        return features

# ---------------------------------------------------------------------------
# ---- Participants should modify these four functions ----------------------
# ---------------------------------------------------------------------------

def prepare_protein_complex(datapoint_id: str, proteins: List[Protein], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein complex prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        proteins: List of protein sequences to predict as a complex
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `proteins`` will contain 3 chains
    # H,L: heavy and light chain of the Fv or Fab region
    # A: the antigen
    #
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME], 
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict

    # Example: predict 5 structures
    cli_args = ["--diffusion_samples", "5"]
    return [(input_dict, cli_args)]


def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligands: list[SmallMolecule], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein-ligand prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        protein: The protein sequence
        ligands: A list of a single small molecule ligand object 
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `protein` is a single-chain target protein sequence with id A
    # `ligands` contains a single small molecule ligand object with unknown binding sites
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME], 
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict

    # Example: predict 5 structures
    # cli_args = ["--diffusion_samples", "5"]
    # return [(input_dict, cli_args)]

    # Get the ligand ID from the input_dict
    ligand_id = None
    for seq_entry in input_dict["sequences"]:
        if "ligand" in seq_entry:
            ligand_id = seq_entry["ligand"]["id"]
            break
    
    # Add affinity properties at the top level of input_dict
    if ligand_id:
        input_dict["properties"] = [
            {
                "affinity": {
                    "binder": ligand_id
                }
            }
        ]
    
    # Generate multiple diverse predictions with different sampling parameters
    configs = []
    
    # Config 1: Default
    cli_args_1 = ["--diffusion_samples", "50", "--seed", "42", "--write_embeddings"]
    configs.append((input_dict.copy(), cli_args_1))
    # Config 2: change --step_scale to 1.8(default is 1.5) This is "temperature" knob for diversity in diffusion
    cli_args_2 = ["--diffusion_samples", "50", "--step_scale", "1.8", "--seed", "42", "--write_embeddings"]
    # configs.append((input_dict.copy(), cli_args_2))
    # Config 3: use physical potetial
    cli_args_3 = ["--diffusion_samples", "50", "--use_potentials", "--seed", "42", "--write_embeddings"]
    # configs.append((input_dict.copy(), cli_args_3))
    # Config 4: --sampling_steps Default 200, increase for better convergence
    cli_args_4 = ["--diffusion_samples", "50", "--sampling_steps", "300", "--seed", "42", "--write_embeddings"]
    # configs.append((input_dict.copy(), cli_args_4))
    # Config 5: change --step_scale to 2.0 (default is 1.5)
    cli_args_5 = ["--diffusion_samples", "50", "--step_scale", "2.0", "--seed", "42", "--write_embeddings"]
    # configs.append((input_dict.copy(), cli_args_5))
    cli_args_6 = ["--diffusion_samples", "50", "--step_scale", "1.0", "--seed", "42", "--write_embeddings"]
    # configs.append((input_dict.copy(), cli_args_6))
    cli_args_7 = ["--diffusion_samples", "50", "--step_scale", "2.0", "--seed", "42", "--write_embeddings"]
    # configs.append((input_dict.copy(), cli_args_7))
    cli_args_8 = ["--diffusion_samples", "50", "--step_scale", "2.0", "--seed", "42", "--write_embeddings"]
    # configs.append((input_dict.copy(), cli_args_8))
    cli_args_9 = ["--diffusion_samples", "50", "--recycling_steps", "5", "--seed", "42", "--write_embeddings"]
    # configs.append((input_dict.copy(), cli_args_9))
    cli_args_10 = ["--diffusion_samples", "50", "--step_scale", "2.0", "--recycling_steps", "5", "--seed", "42", "--write_embeddings"]
    # configs.append((input_dict.copy(), cli_args_10))
    cli_args_11 = ["--diffusion_samples", "50", "--step_scale", "2.0", "--recycling_steps", "10", "--seed", "42", "--write_embeddings"]
    # configs.append((input_dict.copy(), cli_args_11))
    cli_args_12 = ["--diffusion_samples", "50", "--step_scale", "2.0", "--recycling_steps", "4", "--seed", "42", "--write_embeddings"]
    # configs.append((input_dict.copy(), cli_args_12))

    # for efficiency: --max_parallel_samples (default is none, depends on GPU memory)

    return configs

def post_process_protein_complex(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Return ranked model files for protein complex submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
    Returns: 
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    all_pdbs = []
    for prediction_dir in prediction_dirs:
        config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
        all_pdbs.extend(config_pdbs)

    all_confidence_scores = {}
    for prediction_dir in prediction_dirs:
        config_confidence_scores = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.json"))
        

    # Sort all PDBs and return their paths
    all_pdbs = sorted(all_pdbs)
    return all_pdbs

def post_process_protein_ligand(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Return ranked model files for protein-ligand submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
    Returns: 
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    # all_pdbs = []
    # for prediction_dir in prediction_dirs:
    #     config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
    #     all_pdbs.extend(config_pdbs)

    # all_scores = {}
    # for prediction_dir in prediction_dirs:
    #     config_confidence_scores = sorted(prediction_dir.glob(f"confidence_{datapoint.datapoint_id}_config_*_model_*.json"))
        
    #     for json_file in config_confidence_scores:
    #         with open(json_file, 'r') as f:
    #             data = json.load(f)
            
    #         # Process top-level scalar metrics
    #         for key in ['confidence_score', 'ptm', 'iptm', 'ligand_iptm', 
    #                     'protein_iptm', 'complex_plddt', 'complex_iplddt', 
    #                     'complex_pde', 'complex_ipde']:
    #             if key in data:
    #                 if key not in all_scores:
    #                     all_scores[key] = []
    #                 all_scores[key].append(data[key])
            
    #         # Process chains_ptm (flatten to chains_ptm_0, chains_ptm_1, etc.)
    #         if 'chains_ptm' in data:
    #             for chain_id, value in data['chains_ptm'].items():
    #                 key_name = f'chains_ptm_{chain_id}'
    #                 if key_name not in all_scores:
    #                     all_scores[key_name] = []
    #                 all_scores[key_name].append(value)
            
    #         # Process pair_chains_iptm (flatten to pair_chains_iptm_0_0, pair_chains_iptm_0_1, etc.)
    #         if 'pair_chains_iptm' in data:
    #             for chain_i, inner_dict in data['pair_chains_iptm'].items():
    #                 for chain_j, value in inner_dict.items():
    #                     key_name = f'pair_chains_iptm_{chain_i}_{chain_j}'
    #                     if key_name not in all_scores:
    #                         all_scores[key_name] = []
    #                     all_scores[key_name].append(value)
    # print(all_scores)

    """
    Return ranked model files for protein-ligand submission with comprehensive feature analysis.
    
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
    
    Returns:
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    all_pdbs = []
    for prediction_dir in prediction_dirs:
        config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
        all_pdbs.extend(config_pdbs)
    
    print(f"Found {len(all_pdbs)} PDB files")
    
    # Collect confidence scores
    all_scores = defaultdict(list)
    pdb_to_scores = {}
    
    for prediction_dir in prediction_dirs:
        config_confidence_scores = sorted(prediction_dir.glob(f"confidence_{datapoint.datapoint_id}_config_*_model_*.json"))
        
        for json_file in config_confidence_scores:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract model identifier from filename
            model_id = json_file.stem.replace('confidence_', '')
            
            scores_dict = {}
            
            # Process top-level scalar metrics
            for key in ['confidence_score', 'ptm', 'iptm', 'ligand_iptm',
                       'protein_iptm', 'complex_plddt', 'complex_iplddt',
                       'complex_pde', 'complex_ipde']:
                if key in data:
                    scores_dict[key] = data[key]
                    all_scores[key].append(data[key])
            
            # Process chains_ptm
            if 'chains_ptm' in data:
                for chain_id, value in data['chains_ptm'].items():
                    key_name = f'chains_ptm_{chain_id}'
                    scores_dict[key_name] = value
                    all_scores[key_name].append(value)
            
            # Process pair_chains_iptm
            if 'pair_chains_iptm' in data:
                for chain_i, inner_dict in data['pair_chains_iptm'].items():
                    for chain_j, value in inner_dict.items():
                        key_name = f'pair_chains_iptm_{chain_i}_{chain_j}'
                        scores_dict[key_name] = value
                        all_scores[key_name].append(value)
            
            pdb_to_scores[model_id] = scores_dict

    # Extract structural features from each PDB
    print("\nExtracting structural features from PDB files...")
    all_data = []
    
    for pdb_path in all_pdbs:
        print(f"Processing {pdb_path.name}...")
        
        # Extract model identifier
        model_id = pdb_path.stem
        
        # Initialize feature extractor
        try:
            extractor = ProteinLigandFeatureExtractor(str(pdb_path))
            structural_features = extractor.extract_all_features(verbose=False)
        except Exception as e:
            print(f"Warning: Could not extract features from {pdb_path.name}: {e}")
            structural_features = {}
        
        # Combine confidence scores and structural features
        row_data = {
            'pdb_file': pdb_path.name,
            'pdb_path': str(pdb_path),
            'model_id': model_id
        }
        
        # Add confidence scores
        if model_id in pdb_to_scores:
            row_data.update(pdb_to_scores[model_id])
        
        # Add structural features
        row_data.update(structural_features)
        
        all_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save DataFrame to CSV
    output_csv = prediction_dirs[0].parent / f"{datapoint.datapoint_id}_analysis.csv"
    df.to_csv(output_csv, index=False)
   
    # Sort all PDBs and return their paths
    all_pdbs = sorted(all_pdbs)
    return all_pdbs

# -----------------------------------------------------------------------------
# ---- End of participant section ---------------------------------------------
# -----------------------------------------------------------------------------


DEFAULT_OUT_DIR = Path("predictions")
DEFAULT_SUBMISSION_DIR = Path("submission")
DEFAULT_INPUTS_DIR = Path("inputs")

ap = argparse.ArgumentParser(
    description="Hackathon scaffold for Boltz predictions",
    epilog="Examples:\n"
            "  Single datapoint: python predict_hackathon.py --input-json examples/specs/example_protein_ligand.json --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate\n"
            "  Multiple datapoints: python predict_hackathon.py --input-jsonl examples/test_dataset.jsonl --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

input_group = ap.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input-json", type=str,
                        help="Path to JSON datapoint for a single datapoint")
input_group.add_argument("--input-jsonl", type=str,
                        help="Path to JSONL file with multiple datapoint definitions")

ap.add_argument("--msa-dir", type=Path,
                help="Directory containing MSA files (for computing relative paths in YAML)")
ap.add_argument("--submission-dir", type=Path, required=False, default=DEFAULT_SUBMISSION_DIR,
                help="Directory to place final submissions")
ap.add_argument("--intermediate-dir", type=Path, required=False, default=Path("hackathon_intermediate"),
                help="Directory to place generated input YAML files and predictions")
ap.add_argument("--group-id", type=str, required=False, default=None,
                help="Group ID to set for submission directory (sets group rw access if specified)")
ap.add_argument("--result-folder", type=Path, required=False, default=None,
                help="Directory to save evaluation results. If set, will automatically run evaluation after predictions.")


# after existing ap.add_argument(...) lines:
ap.add_argument(
    "--pred-root",
    type=Path,
    required=False,
    default=None,  # will default to args.intermediate_dir / "predictions" later
    help="Root directory where existing Boltz prediction folders are stored. "
         "If not provided, defaults to <intermediate-dir>/predictions",
)


args = ap.parse_args()

def _prefill_input_dict(datapoint_id: str, proteins: Iterable[Protein], ligands: Optional[list[SmallMolecule]] = None, msa_dir: Optional[Path] = None) -> dict:
    """
    Prepare input dict for Boltz YAML.
    """
    seqs = []
    for p in proteins:
        if msa_dir and p.msa:
            if Path(p.msa).is_absolute():
                msa_full_path = Path(p.msa)
            else:
                msa_full_path = msa_dir / p.msa
            try:
                msa_relative_path = os.path.relpath(msa_full_path, Path.cwd())
            except ValueError:
                msa_relative_path = str(msa_full_path)
        else:
            msa_relative_path = p.msa
        entry = {
            "protein": {
                "id": p.id,
                "sequence": p.sequence,
                "msa": msa_relative_path
            }
        }
        seqs.append(entry)
    if ligands:
        def _format_ligand(ligand: SmallMolecule) -> dict:
            output =  {
                "ligand": {
                    "id": ligand.id,
                    "smiles": ligand.smiles
                }
            }
            return output
        
        for ligand in ligands:
            seqs.append(_format_ligand(ligand))
    doc = {
        "version": 1,
        "sequences": seqs,
    }
    return doc

def _run_boltz_and_collect(datapoint) -> None:
    """
    New flow: prepare input dict, write yaml, run boltz, post-process, copy submissions.
    """
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint.datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    # Prepare input dict and CLI args
    base_input_dict = _prefill_input_dict(datapoint.datapoint_id, datapoint.proteins, datapoint.ligands, args.msa_dir)

    if datapoint.task_type == "protein_complex":
        configs = prepare_protein_complex(datapoint.datapoint_id, datapoint.proteins, base_input_dict, args.msa_dir)
    elif datapoint.task_type == "protein_ligand":
        configs = prepare_protein_ligand(datapoint.datapoint_id, datapoint.proteins[0], datapoint.ligands, base_input_dict, args.msa_dir)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    # Run boltz for each configuration
    all_input_dicts = []
    all_cli_args = []
    all_pred_subfolders = []
    
    input_dir = args.intermediate_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    for config_idx, (input_dict, cli_args) in enumerate(configs):
        # Write input YAML with config index suffix
        yaml_path = input_dir / f"{datapoint.datapoint_id}_config_{config_idx}.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(input_dict, f, sort_keys=False)


        # Compute prediction subfolder for this config
        pred_subfolder = out_dir / f"boltz_results_{datapoint.datapoint_id}_config_{config_idx}" / "predictions" / f"{datapoint.datapoint_id}_config_{config_idx}"
        
        all_input_dicts.append(input_dict)
        all_cli_args.append(cli_args)
        all_pred_subfolders.append(pred_subfolder)

    # Post-process and copy submissions
    if datapoint.task_type == "protein_complex":
        ranked_files = post_process_protein_complex(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    elif datapoint.task_type == "protein_ligand":
        ranked_files = post_process_protein_ligand(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    if not ranked_files:
        raise FileNotFoundError(f"No model files found for {datapoint.datapoint_id}")

    for i, file_path in enumerate(ranked_files[:]):
        target = subdir / (f"model_{i}.pdb" if file_path.suffix == ".pdb" else f"model_{i}{file_path.suffix}")
        shutil.copy2(file_path, target)
        print(f"Saved: {target}")

    if args.group_id:
        try:
            subprocess.run(["chgrp", "-R", args.group_id, str(subdir)], check=True)
            subprocess.run(["chmod", "-R", "g+rw", str(subdir)], check=True)
        except Exception as e:
            print(f"WARNING: Failed to set group ownership or permissions: {e}")

def _load_datapoint(path: Path):
    """Load JSON datapoint file."""
    with open(path) as f:
        return Datapoint.from_json(f.read())


def _process_jsonl(jsonl_path: str, msa_dir: Optional[Path] = None):
    """Process multiple datapoints from a JSONL file."""
    print(f"Processing JSONL file: {jsonl_path}")

    for line_num, line in enumerate(Path(jsonl_path).read_text().splitlines(), 1):
        if not line.strip():
            continue

        print(f"\n--- Processing line {line_num} ---")

        try:
            datapoint = Datapoint.from_json(line)
            _run_boltz_and_collect(datapoint)

        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON on line {line_num}: {e}")
            continue
        except Exception as e:
            print(f"ERROR: Failed to process datapoint on line {line_num}: {e}")
            raise e
            continue

def _process_json(json_path: str, msa_dir: Optional[Path] = None):
    """Process a single datapoint from a JSON file."""
    print(f"Processing JSON file: {json_path}")

    try:
        datapoint = _load_datapoint(Path(json_path))
        _run_boltz_and_collect(datapoint)
    except Exception as e:
        print(f"ERROR: Failed to process datapoint: {e}")
        raise


    """Main entry point for the hackathon scaffold."""
    # Determine task type from first datapoint for evaluation
    task_type = None
    input_file = None
    
    if args.input_json:
        input_file = args.input_json
        _process_json(args.input_json, args.msa_dir)
        # Get task type from the single datapoint
        try:
            datapoint = _load_datapoint(Path(args.input_json))
            task_type = datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    elif args.input_jsonl:
        input_file = args.input_jsonl
        _process_jsonl(args.input_jsonl, args.msa_dir)
        # Get task type from first datapoint in JSONL
        try:
            with open(args.input_jsonl) as f:
                first_line = f.readline().strip()
                if first_line:
                    first_datapoint = Datapoint.from_json(first_line)
                    task_type = first_datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
 


def _discover_prediction_dirs(pred_root: Path, datapoint_id: str) -> List[Path]:
    """
    Find all directories that hold model_* files for a datapoint, e.g.:
      pred_root/boltz_results_<ID>_config_X/predictions/<ID>_config_X
    Returns a list of those leaf directories.
    """
    dirs: List[Path] = []
    # Strategy 1: use the expected layout
    for config_dir in pred_root.glob(f"boltz_results_{datapoint_id}_config_*"):
        preds_base = config_dir / "predictions"
        # inside predictions, find the subdir that starts with "<id>_config_"
        for leaf in preds_base.glob(f"{datapoint_id}_config_*"):
            if leaf.is_dir():
                dirs.append(leaf)

    # Strategy 2 (fallback): if above pattern misses anything, look for any parent dir
    # that contains <id>_config_*_model_*.pdb and use its parent as a prediction dir.
    if not dirs:
        for pdb_path in pred_root.rglob(f"{datapoint_id}_config_*_model_*.pdb"):
            parent = pdb_path.parent
            if parent not in dirs:
                dirs.append(parent)

    return sorted(dirs)


def _extract_from_existing(datapoint) -> None:
    """
    Extract features from already-generated predictions for one datapoint.
    DOES NOT RUN BOLTZ. DOES NOT RUN EVALUATION.
    """
    # where predictions are
    pred_root = args.pred_root or (args.intermediate_dir / "predictions")
    pred_root = pred_root.resolve()

    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction root not found: {pred_root}")

    # Discover prediction leaf dirs that contain PDB/JSON for this datapoint
    prediction_dirs = _discover_prediction_dirs(pred_root, datapoint.datapoint_id)
    if not prediction_dirs:
        print(f"[WARN] No prediction folders found for {datapoint.datapoint_id} under {pred_root}")
        return

    # We only need input_dicts / cli_args_list to satisfy the signature; they are unused in your extractor
    input_dicts: List[dict[str, Any]] = []
    cli_args_list: List[list[str]] = []

    # Call your existing post_process function — it already extracts features and writes CSV
    if datapoint.task_type == "protein_ligand":
        _ = post_process_protein_ligand(datapoint, input_dicts, cli_args_list, prediction_dirs)
    elif datapoint.task_type == "protein_complex":
        _ = post_process_protein_complex(datapoint, input_dicts, cli_args_list, prediction_dirs)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")






def main():
    """Extract-only entry point: read datapoints, scan pred folders, write CSV."""
    # Determine pred root default
    if args.pred_root is None:
        args.pred_root = (args.intermediate_dir / "predictions")

    # Process a single JSON datapoint
    if args.input_json:
        print(f"[extract-only] Loading datapoint from {args.input_json}")
        datapoint = _load_datapoint(Path(args.input_json))
        _extract_from_existing(datapoint)
        return

    # Process a JSONL of datapoints (multiple)
    if args.input_jsonl:
        print(f"[extract-only] Loading datapoints from {args.input_jsonl}")
        for line_num, line in enumerate(Path(args.input_jsonl).read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                datapoint = Datapoint.from_json(line)
                _extract_from_existing(datapoint)
            except Exception as e:
                print(f"ERROR at line {line_num}: {e}")
        return





if __name__ == "__main__":
    main()

