"""
Block encoding costs
"""
from typing import Tuple
from dataclasses import dataclass
import math
import numpy as np
from scipy import optimize

def QR(L: int, M1: int) -> Tuple[int, int]:
    """This gives the optimal k and minimum cost for a QROM over L values of
        size M.

    Args:
        L (int) -
        M1 (int) -

    Returns:
       k_opt (int) - k that yields minimal (optimal) cost of QROM
       val_opt (int) - minimal (optimal) cost of QROM
    """
    k = 0.5 * np.log2(L / M1)
    if k < 0:
        k = 0
    value = lambda k: L / np.power(2.0, k) + M1 * (np.power(2.0, k) - 1)
    k_int = [np.floor(k), np.ceil(k)]  # restrict optimal k to integers
    k_opt = k_int[np.argmin(value(k_int))]  # obtain optimal k
    val_opt = np.ceil(value(k_opt))  # obtain ceiling of optimal value given k
    assert k_opt.is_integer()
    assert val_opt.is_integer()
    return int(k_opt), int(val_opt)

def QI(L: int) -> Tuple[int, int]:
    """This gives the opt k and minimum cost for an inverse QROM over L vals

    Args:
        L (int) -

    Returns:
       k_opt (int) - k that yiles minimal (optimal) cost of inverse QROM
       val_opt (int) - minimal (optimal) cost of inverse QROM
    """
    k = 0.5 * np.log2(L)
    if k < 0:
        k = 0
    value = lambda k: L / np.power(2.0, k) + np.power(2.0, k) - 1
    k_int = [np.floor(k), np.ceil(k)]  # restrict optimal k to integers
    k_opt = k_int[np.argmin(value(k_int))]  # obtain optimal k
    val_opt = np.ceil(value(k_opt))  # obtain ceiling of optimal value given k
    assert k_opt.is_integer()
    assert val_opt.is_integer()
    return int(k_opt), int(val_opt)

@dataclass
class SpinFreeData_V1:
    """
    Costs for spin-free block encoding of SOS

    -this includes the minor improvements and corrections to address reviewers concerns

    Args:
        R (int) - R-rank usually related to single factorization rank
        B (int) - B-rank usually related to basis size
        C (int) - C-rank ensemble rank 
        qpe_eps (float) - precision of phase estimation
        chi (int) - bits of precision for keep values
        beta (int) - bits of precision for rotations
        sos_lambda (float) - sos-lambda value which should be sqrt{E_min} * sqrt(Lambda_SA) this should be units of energy
        sys_name (string) - Optional system name to easily identify object
        optimize_qubits (bool) - Optional=True minor sacrifice in Toffoli complexity to minimize qubit costs. This may return 
                                 the same value as optimize_qubits=False if beta is large enough.
        old_lambda (float) - Optional=None prior lambda to compare to
        old_walk_cost (int) - Optional=None prior block encoding costs to compare
    """
    R: int
    B: int
    C: int
    N: int
    qpe_eps: float
    chi: int
    beta: int
    sos_lambda: int
    sys_name: str=None
    optimize_qubits: bool=True
    old_lambda: float=None
    old_walk_cost: int=None
    
    def compare_to_prior(self):
        if self.old_lambda is not None and self.old_walk_cost is not None:
            old_product = self.old_lambda * self.old_walk_cost
            sossa_product = self.walk_costs() * self.sos_lambda
            return np.round(old_product / sossa_product, 1)

    def find_qroam_size_to_minimize_global_temp_ancilla(self, b2):
        """Find the qroam size for inner prep such that the temp qubit costs are less than 
        rotation persistent ancilla.  That way the persistent ancilla for rotations can be used
        for qroam temp ancilla eliminating this ancilla cost completely."""
        k2_opt = QR(self.R * self.C * self.B, b2)[0]
        if self.optimize_qubits:
            for j in range(k2_opt + 1): # state from zero incase k2_opt is already optimal
                k2_opt_new = k2_opt - j
                if math.ceil(math.log2(self.R * self.C * math.ceil(self.B / 2**k2_opt_new))) + ((2**k2_opt_new) - 1) * b2 < self.beta * (self.N - 1):
                    k2_opt = k2_opt_new
                    break
            else:
                raise ValueError("Could not find k2_opt that lowers temp inner-qroam costs below rotation persistent ancilla")
        return k2_opt

    # TOFFOLI COSTS
    def outer_state_prep(self,):
        """Toffoli complexity for state preparation over linear combination of T_2 
        """
        X_o = self.N + self.R * self.C
        uniform = 4 * math.ceil(math.log2(X_o))
        uniform_dag = uniform
        b_k1 = self.chi
        b1 = math.ceil(math.log2(X_o)) + b_k1
        alias_sampling = QR(X_o, b1)[1] + b1 # the additional bk1 is for inquality test and ceil(log2(X_o)) for swaps
        alias_sampling_dag = QI(X_o)[1] + b1 + 1
        return uniform + uniform_dag + alias_sampling + alias_sampling_dag

    def inner_state_prep(self,):
        b_k2 = self.chi
        b2 = math.ceil(math.log2(self.B)) + b_k2 + 1
        k2_opt = QR(self.R * self.C * self.B, b2)[0]

        if self.optimize_qubits:
            k2_opt = self.find_qroam_size_to_minimize_global_temp_ancilla(b2)

        # print(f"{self.B=}", f"{2**k2_opt=}", f"{k2_opt=}")
        if self.B < 2**k2_opt:
            for ii in range(k2_opt):
                k2_opt -= 1
                if self.B >= 2**k2_opt:
                    break
        # print(f"{self.B=}", f"{2**k2_opt=}", f"{k2_opt=}")
        assert self.B >= 2**k2_opt
        RC = self.R * self.C
        for_qroam = RC * math.ceil((self.B + 1) / (2**k2_opt)) + (2**k2_opt) * b2

        k4_opt, QI_cost = QR(self.R * self.C, self.B + 1) # gives ceil(RC / 2^k) + (2^k - 1) (B + 1)  so we need to add B to this quanity in the next line
        for_qroam_dag = QI_cost + (self.B + 1) + self.N + b2  # the self
        uniform_b = 4 * math.ceil(math.log2(self.B + 1))
        uniform_b_dag = uniform_b
        return for_qroam + for_qroam_dag + uniform_b + uniform_b_dag

    def givens_rotations(self) -> int:
        """Toffolis of the N - 1 Givens rotations across the forward and reverse passes."""
        return 2 * (self.N - 1) * (2 * self.beta - 1)

    def rotation_angles_state_prep(self,):
        output_prep = self.N + self.R * self.B
        output_prep_dag = self.R + self.B
        return output_prep + output_prep_dag

    # QUBIT COSTS
    def outer_state_prep_qubit_cost(self, ):
        X_o = self.N + self.R * self.C
        b_k1 = self.chi
        b1 = math.ceil(math.log2(X_o)) + b_k1
        uniform_persistent =  math.ceil(math.log2(X_o)) + 2
        uniform_temp = max(0, math.ceil(math.log2(X_o)) - 2)
        persistent_qrom_out_superposition_reg = b1 + b_k1
        k = QR(X_o, b1)[0]
        qrom_temp = math.ceil(math.log2(X_o / 2**k)) + 2**k * b1
        return {'persistent': persistent_qrom_out_superposition_reg + uniform_persistent, 
                'temporary': qrom_temp + uniform_temp,
                'qrom_temp': qrom_temp,
                'qrom_persistent': persistent_qrom_out_superposition_reg,
                'uniform_temp': uniform_temp,
                'uniform_persistent': uniform_persistent}

    def inner_state_prep_qubit_cost(self, ):
        b_k2 = self.chi
        b2 = math.ceil(math.log2(self.B)) + b_k2 + 1
        k2_opt = QR(self.R * self.C * self.B, b2)[0]

        if self.optimize_qubits:
            k2_opt = self.find_qroam_size_to_minimize_global_temp_ancilla(b2)

        RC = self.R * self.C
        N = self.N
        persisitent_uniform = math.ceil(math.log2(self.B + 1)) + 2
        temporary_uniform = max(0, math.ceil(math.log2(self.B + 1)) - 2)
        persistent_for_qroam = b2 + b_k2 + math.ceil(math.log2(self.R)) + 2
        temporary_for_qroam = math.ceil(math.log2(RC * math.ceil((self.B + 1) / 2**k2_opt))) + ((2**k2_opt) - 1) * b2

        return {'persistent': persistent_for_qroam + persisitent_uniform,
                'temporary': temporary_uniform + temporary_for_qroam,
                'uniform_persistant': persisitent_uniform,
                'uniform_temporary': temporary_uniform,
                'persistent_for_qroam': persistent_for_qroam,
                'temporary_for_qroam': temporary_for_qroam
                }

    def rotation_angles_state_prep_qubit_cost(self):
        persistant = (self.N - 1) * self.beta
        X_o = self.N + self.R * self.C
        temporary = max(math.ceil(math.log2(X_o)), math.ceil(math.log2(self.R * self.B)))
        return {'persistent': persistant,
                'temporary': temporary,
                'rotations': persistant,
                'temp_unary_rb': math.ceil(math.log2(self.R * self.B)),
                'temp_unary_x0':  math.ceil(math.log2(X_o)),
                }

    # reflection costs
    def t2_block_encoding_reflection_cost(self,):
        bk2 = self.chi
        return math.ceil(math.log2(self.B + 1)) + bk2 + 1

    def walk_reflection_cost(self,):
        bk2 = self.chi
        bk1 = self.chi
        X0 = self.N + self.R * self.C
        return math.ceil(math.log2(X0)) + math.ceil(math.log2(self.B + 1)) + bk2 + bk1 + 2
    
    # TOTAL COSTS
    def walk_costs(self,):
        total_walk_costs = self.outer_state_prep() + 2 * self.inner_state_prep() + 2 * self.rotation_angles_state_prep() + 2 * self.select_cost() + self.t2_block_encoding_reflection_cost() + self.walk_reflection_cost()
        return total_walk_costs

    def select_cost(self,):
        basis_rotations = 4 * (self.N - 1) * self.beta
        majorana_selction = 7
        cswap = 2 * self.N
        return basis_rotations + majorana_selction + cswap

    def select_qubits(self,):
        # phase gradient over beta bits
        return {'persistent': self.beta, 'temporary': 0} 

    def qpe_iterations(self,):
        return math.ceil(math.pi * self.sos_lambda / (self.qpe_eps * 2))
    
    def total_costs(self,):
        return self.qpe_iterations() * self.walk_costs()

    def system_qubits(self,):
        return {'persistent': 2 * self.N, 'temporary': 0}

    def total_qubit_costs(self,):
        subroutines = [self.outer_state_prep_qubit_cost,
                       self.inner_state_prep_qubit_cost,
                       self.rotation_angles_state_prep_qubit_cost,
                       self.select_qubits,
                       self.system_qubits]
        total_persistent = 0
        for subroutine in subroutines:
            subroutine_qubit_costs = subroutine()
            total_persistent += subroutine_qubit_costs['persistent']
            # print(subroutine.__name__, subroutine_qubit_costs['persistent'])
        # print(f"{total_persistent=}")
        subroutines.pop(1) # remove inner state prep_qubit_costs()

        temporary = []
        for subroutine in subroutines:
            subroutine_qubit_costs = subroutine()
            temporary.append(subroutine_qubit_costs['temporary'])
        if self.optimize_qubits == False:
            # add back the difference of the for_x_qroam temp costs minus rotation_angles_state_prep_qubit_costs
            if self.inner_state_prep_qubit_cost()['temporary_for_qroam'] > self.rotation_angles_state_prep_qubit_cost()['rotations']:
                temporary.append(self.inner_state_prep_qubit_cost()['temporary_for_qroam'] - self.rotation_angles_state_prep_qubit_cost()['rotations'])
                # print("DIFFERENCE OF FOR_X_QROAM AND PERSISTENT ROTS ", temporary[-1])
        # print(f"{max(temporary)=}")
        return total_persistent + max(temporary)

    
    def print(self):
        print("SYSTEM NAME : ", self.sys_name)
        print("Keep-num-bits for inner and outer : ", self.chi)
        print("rotation-num-bits : ", self.beta)
        print("Outer PREP : ", self.outer_state_prep())
        print("Inner PREP : ", 2 * self.inner_state_prep())
        print("Rotation angle PREP : ", 2 * self.rotation_angles_state_prep())
        print("SEL : ", 2 * self.select_cost())
        print("WALK COST :", int(self.walk_costs()))
        print("Total COST :",  "{: 4.4e}".format(self.total_costs()))
        print("Total Qubit COST :",  int(self.total_qubit_costs()))
        print('-----------------------', end='\n\n\n\n')



@dataclass
class DFTHCSOSSASYSTEMS:
    R: int
    B: int
    C: int
    N: int
    lam: float
    prior_lam: float=None
    prior_walk_cost: int=None
    name: str=None

    def make_spin_from_data(self, qpe_eps=1.6E-3, chi=15, beta=15) -> SpinFreeData_V1:
        return SpinFreeData_V1(R=self.R,
                               B=self.B,
                               C=self.C,
                               N=self.N,
                               qpe_eps=qpe_eps,
                               chi=chi,
                               beta=beta,
                               sos_lambda=self.lam,
                               sys_name=self.name,
                               old_lambda=self.prior_lam,
                               old_walk_cost=self.prior_walk_cost,
                               optimize_qubits=False)
