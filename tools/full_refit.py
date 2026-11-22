#!/usr/bin/env python3
"""Full DF-THC-SOS refit: jointly optimize factorization weights AND angles.

Unlike chem_refit (which keeps angles fixed), this optimizes:
  - Continuous angle parameters theta (which will be quantized to brot bits)
  - W weights (through H2_I parametrization)

The goal: minimize lambda_eff while keeping error_1, error_2 within gates,
at a TARGET brot (even lower than the original).

Based on the verifier's math (dfthc_math.py) translated to pytorch.
"""
import sys, os, math, time, argparse
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

torch.set_default_dtype(torch.float64)

sys.path.insert(0, '/home/claude/competition_data/kickmix-1.0.0-cp312-manylinux_x86_64/kickmix-1.0.0.data/purelib')
sys.path.insert(0, '/home/claude/competition_data/verifiers')

from dfthcsos.verify import load_inputs, QUBIT_CAP
from dfthcsos.scoreinfra.dfthc_math import evaluate_dfthc_angle_words, reconstruct_u_vectors

ERR_GATE = 0.020
E_GAP_FLOOR = 3.914898
LAMBDA_FLOOR = 49.0


def reconstruct_u_torch(theta_over_pi, edges):
    """Givens rotation tree: theta_over_pi -> unit vectors. Differentiable."""
    num_vectors, num_angles = theta_over_pi.shape
    vectors = torch.zeros(num_vectors, num_angles + 1, dtype=torch.float64)
    vectors[:, 0] = 1.0
    for angle_index in range(num_angles):
        parent, child = int(edges[angle_index, 0]), int(edges[angle_index, 1])
        parent_value = vectors[:, parent].clone()
        child_value = vectors[:, child].clone()
        theta = math.pi * theta_over_pi[:, angle_index]
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        vectors = vectors.clone()
        vectors[:, parent] = cosine * parent_value - sine * child_value
        vectors[:, child] = sine * parent_value + cosine * child_value
    return vectors


class FullModel:
    """Computes all dfthcsos metrics with differentiable pytorch ops."""

    def __init__(self, inputs, edges, brot_target):
        ref = inputs.ref
        N = inputs.num_orbitals
        self.N = N
        self.eta = int(ref["electrons"])
        self.ecore = float(ref["ecore"])
        self.energy_upper = float(inputs.energy_upper)
        self.beta_1 = float(inputs.beta_1)
        R, B, C = inputs.w.shape
        self.R, self.B, self.C = R, B, C
        self.A = N + R * B
        self.brot = brot_target
        self.edges = np.asarray(edges, dtype=np.int64)

        oei = np.asarray(ref["oei"], dtype=np.float64)
        tei = np.asarray(ref["tei"], dtype=np.float64)
        h1_thc = oei - 0.5 * np.einsum("prrq->pq", tei)
        self.h1_thc = torch.tensor(h1_thc)

        S = np.asarray(inputs.S, dtype=np.float64)
        self.S = torch.tensor(S)

        # error_2 target tensor
        target = tei.copy()
        d = np.arange(N)
        target[:, :, d, d] += 0.5 * S[:, :, None]
        target[d, d, :, :] += 0.5 * S[None, :, :]
        self.h2_target = torch.tensor(target).reshape(N * N, N * N)

        self.eye = torch.eye(N, dtype=torch.float64)

    def quantize(self, theta_continuous):
        """Quantize continuous theta/pi values to brot-bit integers, then back."""
        scale = 1 << (self.brot - 1)  # 2^(brot-1)
        words = torch.clamp(torch.round(theta_continuous * scale), 0, (1 << self.brot) - 1)
        return words / scale  # quantized theta/pi

    def metrics(self, theta_continuous, w_param, h_param, *, quantize=True, assignment=None):
        """Compute all metrics. theta_continuous = theta/pi (A, N-1)."""
        N = self.N
        R, B, C = self.R, self.B, self.C

        # Reconstruct u vectors from angles (with or without quantization)
        if quantize:
            # STE: forward uses quantized, backward passes through
            theta_q = theta_continuous + (self.quantize(theta_continuous) - theta_continuous).detach()
        else:
            theta_q = theta_continuous

        u_all = reconstruct_u_torch(theta_q, self.edges)

        # Normalize rows
        norms = torch.norm(u_all, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-30)
        u_all = u_all / norms

        # Split one-body and two-body
        one_body = u_all[:N]  # (N, N)
        u_b = u_all[N:].reshape(R, B, N)  # (R, B, N)
        u_b_norm = torch.norm(u_b, dim=2, keepdim=True)
        u_b = u_b / torch.clamp(u_b_norm, min=1e-30)

        # W from stored parametrization
        W = h_param - w_param.sum(dim=1)  # (R, C) = H2_I - sum_b w

        # X tensor
        X = torch.einsum("rbp,rbq,rbc->rcpq", u_b, u_b, w_param)  # (R, C, N, N)

        # H1'
        h1p = (self.h1_thc - 0.5 * self.eta * self.S + self.beta_1 * self.eye
               - torch.einsum("rc,rcpq->pq", W, X))

        eig, vec = torch.linalg.eigh(h1p)
        lam_h1 = eig.abs().sum()

        # error_1: match one-body rows to eigenvectors
        if assignment is None:
            overlaps = (one_body @ vec).abs().detach().numpy()
            qi, ii = linear_sum_assignment(-overlaps)
            assignment = (torch.as_tensor(qi), torch.as_tensor(ii))
        qi, ii = assignment
        matched = torch.empty_like(eig)
        matched[qi] = eig[ii]
        h1p_qrom = torch.einsum("j,jp,jq->pq", matched, one_body, one_body)
        recon = h1p_qrom + 0.5 * self.eta * self.S - self.beta_1 * self.eye + torch.einsum("rc,rcpq->pq", W, X)
        error_1 = torch.linalg.norm(recon - self.h1_thc)

        # error_2
        Xf = X.reshape(-1, N * N)
        error_2 = torch.linalg.norm(Xf.T @ Xf - self.h2_target)

        # Lambda
        w_B = W + w_param.sum(dim=1)
        lam_h2 = 0.25 * ((w_param.abs().sum(dim=1) + w_B.abs()) ** 2).sum()
        Lambda = lam_h1 + lam_h2

        E_SOS = self.ecore - self.eta * self.beta_1 - (lam_h1 - eig.sum()) - 0.5 * (W ** 2).sum()
        E_gap = self.energy_upper - E_SOS
        lambda_eff = torch.sqrt(torch.clamp(E_gap * (2 * Lambda - E_gap), min=1e-12))

        return {
            'error_1': error_1, 'error_2': error_2,
            'Lambda': Lambda, 'E_gap': E_gap,
            'lambda_eff': lambda_eff, 'E_SOS': E_SOS,
            'lam_h1': lam_h1, 'lam_h2': lam_h2,
        }, assignment


def optimize(model, theta0, w0, h0, *, steps=500, lr=1e-4, penalty=1000.0, log_every=50):
    """Optimize lambda_eff with penalty for constraint violations."""
    theta = torch.tensor(theta0, requires_grad=True)
    w = torch.tensor(w0, requires_grad=True)
    h = torch.tensor(h0, requires_grad=True)

    optimizer = torch.optim.Adam([theta, w, h], lr=lr)

    best_leff = float('inf')
    best_state = None
    assign = None

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        m, assign = model.metrics(theta, w, h,
                                   assignment=None if step % 20 == 1 else assign)

        err1 = float(m['error_1'].detach())
        err2 = float(m['error_2'].detach())
        egap = float(m['E_gap'].detach())
        leff = float(m['lambda_eff'].detach())
        lam = float(m['Lambda'].detach())

        # Penalty for constraint violations
        loss = m['lambda_eff']
        if err1 > ERR_GATE:
            loss = loss + penalty * (m['error_1'] - ERR_GATE) ** 2
        if err2 > ERR_GATE:
            loss = loss + penalty * (m['error_2'] - ERR_GATE) ** 2
        if egap < E_GAP_FLOOR:
            loss = loss + penalty * (E_GAP_FLOOR - m['E_gap']) ** 2
        if lam < LAMBDA_FLOOR:
            loss = loss + penalty * (LAMBDA_FLOOR - m['Lambda']) ** 2

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([theta, w, h], 1.0)

        optimizer.step()

        # Clamp theta to valid range
        with torch.no_grad():
            theta.clamp_(0, (1 << model.brot) / (1 << (model.brot - 1)) - 1e-10)

        feasible = (err1 <= ERR_GATE and err2 <= ERR_GATE and
                   egap >= E_GAP_FLOOR and lam >= LAMBDA_FLOOR)

        if feasible and leff < best_leff:
            best_leff = leff
            best_state = {
                'theta': theta.detach().clone(),
                'w': w.detach().clone(),
                'h': h.detach().clone(),
                'leff': leff,
                'err1': err1, 'err2': err2,
                'egap': egap, 'Lambda': lam,
            }

        if step % log_every == 0 or step == 1:
            status = "PASS" if feasible else "FAIL"
            best_str = f" best={best_leff:.4f}" if best_state else ""
            print(f"  step {step:4d}: leff={leff:.4f} err1={err1:.6f} err2={err2:.6f} "
                  f"E_gap={egap:.6f} Lam={lam:.2f} [{status}]{best_str}")

    return best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/claude/ahoirg_assets/submission.npz")
    ap.add_argument("--brot", type=int, default=7)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--penalty", type=float, default=5000.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print(f"Loading {a.input}...")
    sol = load_inputs(a.input)
    z = np.load(a.input, allow_pickle=False)
    words = z['angle_words']
    edges = z['tree_edges']
    R, B, C = sol.w.shape

    # Determine original brot
    orig_brot = int(np.ceil(np.log2(words.max() + 1))) if words.max() > 0 else 1
    print(f"R={R} B={B} C={C} orig_brot={orig_brot} target_brot={a.brot}")
    print(f"A={words.shape[0]} angles={words.shape[1]}")

    # Initialize continuous angles from original words
    theta0 = words.astype(np.float64) / (1 << (orig_brot - 1))  # theta/pi

    # Initialize weights
    w0 = np.asarray(sol.w, dtype=np.float64)  # (R, B, C)
    h0 = np.asarray(sol.params["H2_I"], dtype=np.float64)  # (R, C)

    model = FullModel(sol, edges, a.brot)

    # First evaluate at original brot
    m_orig = evaluate_dfthc_angle_words(words, sol.w, sol.W, sol.S, sol.beta_1, sol.ref,
                                         brot=orig_brot, tree_edges=edges,
                                         energy_upper=sol.energy_upper)
    print(f"\nOriginal (brot={orig_brot}):")
    print(f"  err1={m_orig['error_1']:.6f} err2={m_orig['error_2']:.6f} "
          f"lambda_eff={m_orig['lambda_eff']:.4f}")

    # Evaluate quantized at target brot
    scale = 1 << (a.brot - 1)
    words_target = np.clip(np.round(theta0 * scale), 0, (1 << a.brot) - 1).astype(np.int64)
    m_quant = evaluate_dfthc_angle_words(words_target, sol.w, sol.W, sol.S, sol.beta_1, sol.ref,
                                          brot=a.brot, tree_edges=edges,
                                          energy_upper=sol.energy_upper)
    print(f"\nNaive quantize to brot={a.brot}:")
    print(f"  err1={m_quant['error_1']:.6f} err2={m_quant['error_2']:.6f} "
          f"lambda_eff={m_quant['lambda_eff']:.4f}")

    print(f"\nOptimizing (steps={a.steps}, lr={a.lr}, penalty={a.penalty})...")
    t0 = time.time()
    result = optimize(model, theta0, w0, h0,
                     steps=a.steps, lr=a.lr, penalty=a.penalty, log_every=50)
    dt = time.time() - t0

    if result:
        print(f"\nBest feasible: lambda_eff={result['leff']:.4f} "
              f"err1={result['err1']:.6f} err2={result['err2']:.6f}")

        # Verify with official evaluator
        theta_opt = result['theta'].numpy()
        words_opt = np.clip(np.round(theta_opt * scale), 0, (1 << a.brot) - 1).astype(np.int64)

        w_opt = result['w'].numpy().astype(np.float32)
        h_opt = result['h'].numpy().astype(np.float64)

        print(f"\nOfficial verification at brot={a.brot}...")
        m_v = evaluate_dfthc_angle_words(words_opt, w_opt,
                                          h_opt - w_opt.sum(axis=1),  # W = H2_I - sum_b w
                                          sol.S, sol.beta_1, sol.ref,
                                          brot=a.brot, tree_edges=edges,
                                          energy_upper=sol.energy_upper)
        print(f"  err1={m_v['error_1']:.6f} err2={m_v['error_2']:.6f} "
              f"lambda_eff={m_v['lambda_eff']:.4f}")
        passed = (m_v['error_2'] <= 0.020 and m_v['error_1'] <= 0.020 and
                 m_v['Lambda'] > 49.0 and m_v['E_gap'] > 3.914898)
        print(f"  PASS={passed}")

        if a.out and passed:
            np.savez(a.out,
                     U=sol.params.get('U', np.eye(76)),
                     W=w_opt,
                     H2_I=h_opt.astype(np.float64),
                     H1_I=sol.params.get('H1_I', np.zeros(76*(76+1)//2)),
                     Hsym_lower=sol.params.get('Hsym_lower', np.zeros(76*(76+1)//2)),
                     tree_edges=edges,
                     angle_words=words_opt)
            print(f"  Saved to {a.out}")
    else:
        print(f"\nNo feasible solution found in {a.steps} steps ({dt:.1f}s)")

    print(f"\nTotal time: {dt:.1f}s")


if __name__ == "__main__":
    main()
