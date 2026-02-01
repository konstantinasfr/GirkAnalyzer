import json
import numpy as np
import pandas as pd

# def gat1508_numbering(residue_id):
#     if residue_id<=324:
#         return residue_id+54,0
#     elif residue_id<=649:
#         return residue_id-324+54,1
#     elif residue_id<=794:
#         return residue_id-(324+325)+54,2
#     elif residue_id<=1300:
#         return residue_id-(324+325+326)+43,3
        
# def convert_to_pdb_numbering(residue_id, channel_type):
#     """
#     Converts a residue ID to a PDB-style numbering.
#     """
#     if channel_type == "G4":
#         residues_per_chain = 325
#         offset = 49
#     elif channel_type == "G2" or channel_type == "G2_FD":
#         residues_per_chain = 328
#         offset = 54
#     elif channel_type == "G12":
#         residues_per_chain = 325
#         offset = 53
#     elif channel_type == "G12_gat":
#         pdb_number, chain_number = gat1508_numbering(residue_id)


#     amino_acid_names = {152:"E",
#                        184:"N",
#                        141:"E",
#                        173:"D",
#                         170:"S",
#                         181:"S",
#                         177:"S",
#                         166:"S"
#                        }
#     if channel_type == "G2_FD":
#             amino_acid_names = {152:"E",
#                        184:"N",
#                        141:"E",
#                        173:"D",
#                        }
            
#     if residue_id != "SF":
#         if channel_type != "G12_gat":
#             residue_id = int(residue_id)
#             chain_number = int(residue_id)//residues_per_chain
#             if channel_type == "G2" or channel_type == "G2_FD":
#                 chain_dict = {0:"A", 1:"B", 2:"C", 3:"D"}
#             elif channel_type == "G12":
#                 chain_dict = {0:"D", 1:"C", 2:"B", 3:"A"}
#             pdb_number = residue_id-residues_per_chain*chain_number+offset
#         elif channel_type == "G12_gat":
#             chain_dict = {0:"A", 1:"B", 2:"C", 3:"D"}

#         if channel_type == "G12" and residue_id<=325:
#             pdb_number = residue_id+42
#         if channel_type == "G2_FD" and pdb_number==184 and chain_number==0:
#             return "D184.A"
#         if pdb_number not in amino_acid_names:
#             return f"{pdb_number}.{chain_dict[chain_number]}"
#         else:
#             return f"{amino_acid_names[pdb_number]}{pdb_number}.{chain_dict[chain_number]}"
#     else:
#         return "SF"

def get_first_ion_id_part(ion_id):
    return ion_id.split('_')[0]


def convert_to_pdb_numbering(residue_id, channel_type):
    """Converts a residue ID to PDB-style numbering."""
    if channel_type == "G2":
        glu_mapping = {98: "152.A", 426: "152.C", 754: "152.B", 1082: "152.D"}
        asn_mapping = {130: "184.A", 458: "184.C", 786: "184.B", 1114: "184.D"}
        if residue_id in glu_mapping:
            return glu_mapping[residue_id]
        if residue_id in asn_mapping:
            return asn_mapping[residue_id]
    elif channel_type == "G12" or channel_type == "G12_GAT" :
        glu_mapping = {422: "152.A", 98: "152.B", 747: "152.C", 1073: "141.D"}
        asn_asp_mapping = {454: "184.A", 130: "184.B", 779: "184.C", 1105: "173.D"}
        if residue_id in glu_mapping:
            return glu_mapping[residue_id]
        if residue_id in asn_asp_mapping:
            return asn_asp_mapping[residue_id]
    elif channel_type == "G12_ML":
        glu_mapping = {749: "152.A", 1074: "152.B", 424: "152.C", 99: "141.D"}
        asn_asp_mapping = {781: "184.A", 1106: "184.B", 456: "184.C", 131: "173.D"}
        if residue_id in glu_mapping:
            return glu_mapping[residue_id]
        if residue_id in asn_asp_mapping:
            return asn_asp_mapping[residue_id]
    return str(residue_id)
    