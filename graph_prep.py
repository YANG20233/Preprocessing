import pandas as pd
import numpy as np
import torch

import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader

from rdkit.Chem import AllChem, rdchem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit import Chem, DataStructs

from sklearn.preprocessing import OneHotEncoder

def find_max_atomic_number(df):
    max_atomic_number = 0
    for mol in df['RD_MOL']:
        if mol:
            for atom in mol.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                if atomic_num > max_atomic_number:
                    max_atomic_number = atomic_num
    return max_atomic_number

# Prepare the encoder dynamically based on the maximum atomic number
def prepare_encoder(df):
    max_atomic_number = find_max_atomic_number(df)
    observed_atomic_numbers = list(range(1, max_atomic_number + 1))  # Atomic numbers from 1 to max

    # Convert observed atomic numbers to a numpy array for OneHotEncoder
    observed_atomic_numbers = np.array(observed_atomic_numbers).reshape(-1, 1)

    # Initialize and fit OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(observed_atomic_numbers)  # Fit encoder only on possible atomic numbers
    return encoder

# Extended feature encoding for bonds
def bond_to_feature(bond):
    bond_type = bond.GetBondType()
    is_conjugated = bond.GetIsConjugated()
    is_in_ring = bond.IsInRing()
    return [
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC),
        int(is_conjugated),
        int(is_in_ring)
    ]

# Convert an RDKit molecule to a PyTorch Geometric graph with extended features
def molecule_to_graph(molecule, encoder, dili_risk_label):
    atoms = molecule.GetAtoms()
    bonds = molecule.GetBonds()

    if len(atoms) == 0:
        return None  # Handle molecules with no atoms

    # Convert atoms to node features with additional features
    atom_features = []
    for atom in atoms:
        atom_feature = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetHybridization(),
            atom.GetTotalNumHs(),
            atom.GetIsAromatic()
        ]
        atom_features.append(atom_feature)

    # Keep only atomic numbers for encoding, add other features later
    atom_features = np.array(atom_features)[:, 0].reshape(-1, 1)
    node_features = encoder.transform(atom_features).toarray()  # One-hot encode atomic numbers
    node_features = np.hstack([node_features, np.array(atom_features)[:, 1:]])  # Add other atom features to the node features

    edge_index = []
    edge_attr = []

    # Convert bonds to edges and edge attributes with additional features
    for bond in bonds:
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[start, end], [end, start]]  # Add both directions
        edge_attr += [bond_to_feature(bond), bond_to_feature(bond)]  # Add bond features

    # Convert lists to tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Add self-loops to the graph
    self_loops = torch.arange(node_features.size(0)).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    # Create default attributes for self-loops (all zeros)
    self_loop_attr = torch.zeros(node_features.size(0), edge_attr.size(1))
    edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

    # Ensure the number of edges and edge attributes match
    assert edge_index.size(1) == edge_attr.size(0), "Number of edges in edge_index and edge_attr should match"

    # Add graph-level features
    graph_features = torch.tensor([molecule.GetNumAtoms(), molecule.GetNumBonds()], dtype=torch.float).unsqueeze(0)

    # Add the DILI risk label
    dili_risk_label = torch.tensor([dili_risk_label], dtype=torch.long)

    # Create a PyTorch Geometric Data object
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=dili_risk_label, graph_features=graph_features)

# Prepare the encoder
encoder = prepare_encoder(df)

graphs = []
for mol, dili_risk in zip(df['RD_MOL'], df['DILI_Classification']):
    if mol:
        try:
            graph = molecule_to_graph(mol, encoder, dili_risk) 
            if graph:
                graphs.append(graph)
        except Exception as e:
            print(f"Error processing molecule: {e}")

for i, data in enumerate(graphs):
    print(f"Graph {i+1}:")
    
    # Verify node features
    num_nodes, num_node_features = data.x.size()
    print(f"  Number of nodes: {num_nodes}, Node feature size: {num_node_features}")
    
    # Verify edge index
    edge_index_shape = data.edge_index.size()
    print(f"  Edge index shape: {edge_index_shape}")
    assert edge_index_shape[0] == 2, "Edge index should have shape [2, num_edges]"
    
    # Verify edge attributes
    num_edges, num_edge_features = data.edge_attr.size()
    print(f"  Number of edges: {num_edges}, Edge feature size: {num_edge_features}")
    assert edge_index_shape[1] == num_edges, "Number of edges in edge_index and edge_attr should match"
    
    # Verify target variable (y)
    print(f"  Target label shape: {data.y.size()}")
    
    # Verify graph-level features
    print(f"  Graph-level features shape: {data.graph_features.size()}")
    
    print("\n")
