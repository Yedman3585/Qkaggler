import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple

dtype = jnp.float32

class FCIDumpFileContent:
    num_orbitals: int
    one_body_terms: List[Tuple[List[int], float]]
    two_body_terms: List[Tuple[List[int], float]]

    def __init__(self, num_orbitals, one_body_terms, two_body_terms):
        self.num_orbitals = num_orbitals
        self.one_body_terms = one_body_terms
        self.two_body_terms = two_body_terms
    
    @classmethod
    def from_file(self, file_name):
        with open(file_name, "r") as f:
            lines = [str.strip() for str in f.readlines()]

        assert lines[0].startswith("&FCI")

        parse_header = True
        header = ""
        header_values = {}
        self.one_body_terms = []
        self.two_body_terms = []
        self.ecore = 0.

        for line in lines:
            if parse_header:
                # File header might not have key-value pairs in separate lines,
                # so accumulate the whole header first and then parse.
                header += line + " "
                if line.endswith("&END"):
                    # Strip "&FCI" and "&END"
                    rest_header = header[4:-4]

                    # Find each key-value pair based on the "=" sign
                    while (ind := rest_header.find("=")) > -1:
                        key = rest_header[0:ind].strip()
                        rest_header = rest_header[ind + 1 :]
                        # Figure out which part of the rest is value: until
                        # either whitespace or (if key is not ORBSYM) comma
                        ind = rest_header.find("," if key != "ORBSYM" else " ")
                        value = rest_header[0:ind].strip()
                        rest_header = rest_header[ind + 1 :]
                        if key == "ORBSYM":
                            value = value[:-1]
                        header_values[key] = value
                    parse_header = False
            else:
                tokens = line.split()
                coefficient = float(tokens[0])
                indices = [int(str) - 1 for str in tokens[1:] if str != "0"]

                if len(indices) == 2:
                    indices.sort()
                    self.one_body_terms.append((indices, coefficient))
                elif len(indices) == 4:
                    # conversion from Mulliken to Dirac
                    i = indices[0]
                    j = indices[2]
                    k = indices[3]
                    l = indices[1]

                    symmetries = [
                        [i, j, k, l],
                        [j, i, l, k],
                        [k, l, i, j],
                        [l, k, j, i],
                        [i, k, j, l],
                        [k, i, l, j],
                        [j, l, i, k],
                        [l, j, k, i],
                    ]
                    symmetries.sort()
                    self.two_body_terms.append((symmetries[0], coefficient))
                elif len(indices) == 0:
                    self.ecore = coefficient

        self.num_orbitals = int(header_values["NORB"])
        self.num_electrons = int(header_values["NELEC"])

        return self

def form_indices(orbitals):
    tril = jnp.tril_indices(orbitals)
    tril_diag = (tril[0]==tril[1]).nonzero()[0]
    tril_shift = jnp.array([tril_diag.repeat(orbitals), jnp.tile(tril_diag,orbitals)])
    return tril, tril_diag, tril_shift

def H2_full_from_lower(H2_lower):
    orbitals = int(-0.5 + np.sqrt(0.25 + 2.0 * float(H2_lower.shape[0])))
    if int(orbitals * (orbitals + 1) / 2) != H2_lower.shape[1]:
        raise Exception(f"H2_lower {H2_lower.shape} dimensions not all equal or computed orbitals {orbitals} incorrect.")

    tril, _, _ = form_indices(orbitals)
    
    sym = np.zeros((orbitals*(orbitals+1)//2,orbitals,orbitals))
    sym[:,tril[0],tril[1]] = np.array(H2_lower)
    sym[:,tril[1],tril[0]] = np.array(H2_lower)
    H2_full = np.zeros((orbitals,orbitals,orbitals,orbitals))
    H2_full[tril[0],tril[1],:,:] = sym
    H2_full[tril[1],tril[0],:,:] = sym 
    return H2_full

def H2_lower_from_full(H2_full):
    orbitals = H2_full.shape[0]
    if np.sum(np.square(np.array(H2_full.shape) - orbitals)) > 0:
        raise Exception(f"H2_full {H2_full.shape} dimensions not all equal")
    tril, _, _ = form_indices(orbitals)
    H2_lower = H2_full[:,:,tril[0],tril[1]][tril[0],tril[1],:] 
    return H2_lower


def Spinful_ref(ref, electrons_0 = None):
    n = ref['orbitals']
    tril, tril_diag, tril_shift = form_indices(2 * n)
    ref_out = {k:v for k,v in ref.items()}
    ref_out.update({'tril': tril, 
                    'tril_diag': tril_diag, 
                    'tril_shift': tril_shift})
    ref_out['electrons_0'] = electrons_0 if electrons_0 is not None else int(ref['electrons'] / 2)
    ref_out['electrons_1'] = ref['electrons'] - ref_out['electrons_0']
    ref_out['spin_squared'] = 0
    h2_full = H2_full_from_lower(ref['H2_lower'])
    ref_out['H2_spinful_lower'] = H2_lower_from_full((np.pad(h2_full,((0,n),(0,n),(0,n),(0,n))) 
                                   + np.pad(h2_full,((0,n),(0,n),(n,0),(n,0)))
                                   + np.pad(h2_full,((n,0),(n,0),(0,n),(0,n)))
                                   + np.pad(h2_full,((n,0),(n,0),(n,0),(n,0))))
                                )
    return ref_out

def fci_from_fcidump(filename):
    FCIDump = FCIDumpFileContent.from_file(filename)
    electrons, orbitals, one_body_terms, two_body_terms = FCIDump.num_electrons, FCIDump.num_orbitals, FCIDump.one_body_terms, FCIDump.two_body_terms,

    tril, tril_diag, tril_shift = form_indices(orbitals)

    Hcore = np.zeros((orbitals,orbitals))
    for [i, j], val in FCIDump.one_body_terms:
        Hcore[i,j] = val
        Hcore[j,i] = val
    #Hcore = np.array(Hcore)

    
    combined_index = lambda i, j : int(max(i, j) * (max(i, j) + 1) / 2 + min(i, j))
    
    H2_lower = np.zeros((orbitals*(orbitals+1)//2,orbitals*(orbitals+1)//2)).astype(dtype)
    for [i, j, k, l], val in FCIDump.two_body_terms:
        ii, jj = combined_index(i,l), combined_index(j,k)
        H2_lower[ii,jj] = val
        H2_lower[jj,ii] = val
    #H2_lower = jnp.array(H2_lower + H2_lower.T - np.diag(np.diag(H2_lower))).astype(dtype)

    H2_full = H2_full_from_lower(H2_lower)
    
    H1_DF = np.zeros((orbitals,orbitals))
    H1_DF = np.array(Hcore + np.einsum('pprs->rs',H2_full) - 0.5 * np.einsum('pqqs->ps',H2_full))
    
    H1_THC = np.array(Hcore - 0.5 * np.einsum('pqqs->ps',H2_full))
    ecore_THC = FCIDump.ecore + np.trace(H1_THC)
    
    ecore_DF = FCIDump.ecore + np.trace(H1_DF) - 0.5 * np.einsum('ppqq', H2_full)

    return {"orbitals": orbitals, 
            "electrons": electrons,
            "H1": Hcore, 
            #"H2_full": H2_full, 
            "ref": {"orbitals": orbitals, 
                    "electrons": electrons,
                    "tril": tril, 
                    "tril_diag": tril_diag, 
                    "tril_shift": tril_shift, 
                    "ecore": FCIDump.ecore, 
                    'ecore_df': ecore_DF,
                    'ecore_thc': ecore_THC,
                    "H1_core": Hcore,
                    "H1_DF": H1_DF, 
                    "H1_THC": H1_THC,
                    "H2_lower": H2_lower,
                    "H2_full": H2_full}} 

def export_fcidump(path, preamble, ecore, Hcore, H2_full):
    orbitals = Hcore.shape[0]
    tril, tril_diag, tril_shift = form_indices(orbitals)
    contents = [preamble]

    x = np.tile(np.transpose(tril,(1,0)),(tril[0].shape[0], 1))
    y = np.tile(np.transpose(tril,(1,0)),(1,tril[0].shape[0])).reshape(-1,2)
    xy = np.hstack((x,y))
    H2_full_vals = H2_full[xy[:,0],xy[:,1],xy[:,2],xy[:,3]]
    H2_strings = [f"{val}    {i+1}    {j+1}    {k+1}    {l+1}" for (i,j,k,l), val in zip(xy, H2_full_vals)]
    #for k, l in jnp.transpose(jnp.array(tril)):
    #    for i, j in jnp.transpose(jnp.array(tril)):
    #        contents.append(f"{H2_full[i,j,k,l]}   {i+1}    {k+1}    {l+1}    {j+1}")
    for i, j in jnp.transpose(jnp.array(tril)):
        contents.append(f"{Hcore[i,j]}   {i+1}    {j+1}    0    0")
    
    with open(path, 'w') as f:
        f.write(preamble)
        f.write("\n".join(H2_strings))
        f.write("\n")
        for i, j in jnp.transpose(jnp.array(tril)):
            f.write(f"{Hcore[i,j]}   {i+1}    {j+1}    0    0\n")
        f.write(f"{ecore}    0    0    0    0")