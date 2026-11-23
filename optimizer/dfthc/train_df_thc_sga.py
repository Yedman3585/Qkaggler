from dfthc import df_thc_sga
from dfthc import load_integrals
import optax
import time
from jax import jit
import jax

import argparse
import jax.numpy as jnp
from starter.task1_factorization.solution_io import read_factorization, save_solution_npz

# Other generating sets (THC, Spin-free2, One-body, Spin-symmetric,
# General-shared, H2-symmetric-shared, H1H2-symmetric[-shared]) were moved to
# dfthc/old/ along with their modules. The historical "Spin-free" objective
# (Compute_df_thc_sga_cost_function) was removed; "Qaggle" supersedes it.
generating_sets = {"Qaggle": (df_thc_sga.Compute_qaggle_df_thc_cost_function,
                                df_thc_sga.Generate_params,
                                lambda *args, **kwargs: '')}




def Train(generating_set: str,
         method: str,
         epochs: int, 
         learning_rate: float, 
         rho: float, 
         params,
         ref,
         print_interval = 10000,
         epoch_start = 0,
         generators = None,
         solver = "adam",
         learning_rate_final: float = 0.0001, 
         frozen_params = None,
         **kwargs
         ):
    """Optimize ``params``; ``frozen_params`` are held fixed (no gradients or optimizer state).

    ``frozen_params`` maps parameter names to the constants that replace them
    during training, e.g. the benchmark's ``Hsym_lower`` and ``H1_I``. The
    returned ``params_best`` contains the trained and the frozen leaves.
    """
    print(f"Training U/V {params['U'].shape} and W {params['W'].shape} with generators {generators} for {epochs} epochs with {method}.")
    frozen_params = dict(frozen_params or {})
    params = {key: value for key, value in params.items() if key not in frozen_params}
    complete_params = lambda active: {**active, **frozen_params}
    epoch_counter = epoch_start
    params_best = params

    compute_all = lambda params: generating_sets[generating_set][0](complete_params(params), ref, rho, method, generators = generators, **kwargs)
    compute_all_jit = jit(compute_all)
    compute_cost = lambda params: compute_all(params)['cost']
    if solver == "adam":
        scheduler = optax.exponential_decay(learning_rate, epochs, decay_rate = learning_rate_final / learning_rate)
        optimizer = optax.adam(learning_rate=scheduler)
        opt_state = optimizer.init(params)
        value_and_grad = jax.value_and_grad(compute_cost)
        def update(params, opt_state):
            value, grad = value_and_grad(params)
            updates, opt_state = optimizer.update(grad, opt_state, params, value=value, grad=grad, value_fn=compute_cost)  
            params_next = optax.apply_updates(params, updates)
            return (params_next, params, value, opt_state)
    elif solver == "lbfgs":
        optimizer = optax.lbfgs(learning_rate = learning_rate)
        opt_state = optimizer.init(params)
        value_and_grad = optax.value_and_grad_from_state(compute_cost)
            
        def update(params, opt_state):
            value, grad = value_and_grad(params, state = opt_state)
            updates, opt_state = optimizer.update(grad, opt_state, params, value=value, grad=grad, value_fn=compute_cost)  
            params_next = optax.apply_updates(params, updates)
            return (params_next, params, value, opt_state)
          
    update_jit = jit(update)
    
    time_start = time.time()
    time_counter = 0
    best_cost = float('inf')
    found_best = False
    best_counter = 0


    update_jit_2 = jit(lambda i, inputs: update_jit(inputs[0], inputs[3]))
    i_print_max = int(epochs / 10)
    for i_print in range(i_print_max):

        
        params, params_prev, value, opt_state = jax.lax.fori_loop(0, 10, update_jit_2, (params,params,0.,opt_state))
        
        '''
        for i in range(print_interval):
            params, params_prev, value, opt_state = update_jit(params, opt_state)
            
            if value < best_cost:
                best_cost = value
                found_best = True
                params_best = params_prev
        '''
        if value < best_cost:
            best_cost = value
            found_best = True
            params_best = params_prev
       
        if (i_print*10) % print_interval == 0 or i_print == i_print_max - 1:
            params_print = params_best if found_best else params
            message = ["B@"] if found_best else ["  "]
            o = compute_all_jit(params_print)
            #print(o)
            message.append(f"epoch {(i_print) * 10}: ")
            message.append(f"ε_fro={1000.0 * o['epsilon_fro']:.1f}mHa, ")
            message.append(
                f"λ_eff={o['lambda_eff']:.2f}Ha, "
                f"Λ={o['Lambda']:.2f}Ha "
                f"(Λ_1={o['lambda_H1']:.3f}, Λ_2={o['lambda_H2']:.1f}), "
            )
            #message.append(f"E∈[{o['H0']-o['lambda_H1']-o['error_h2_ord=1']:.1f},{o['H0']+o['lambda_H1']+o['Spectral norm above zero']+o['error_h2_ord=1']:.1f}]Ha, ")
            message.append(f"cost = {1000.0 * o['cost']:.1f}mHa, ")
            message.append(f"E_SOS: {o['E_SOS']:.3f}Ha, ")
            message.append(f"gap: {o['sos_gap']:.4f}Ha, ")
            message.append(f"ms/it: {1000.0 * (time.time() - time_start) / print_interval:.2f}")
            
            print("".join(message))        
            time_start = time.time()
            found_best = False
        if ((i_print*10) % (10*print_interval) == 0 and i_print > 0) or i_print == i_print_max - 1:
            generating_sets[generating_set][2](complete_params(params_best),ref,generators)
    '''
    for i in range(epochs):
        time_counter += 1
            
        params, params_prev, value, opt_state = update_jit(params, opt_state)
        if value < best_cost:
            best_cost = value
            found_best = True
            best_counter = epoch_counter
            params_best = params_prev
        if i % print_interval == 0 or i == epochs - 1:
            if i != 0 or print_first:
                params_print = params_best if found_best else params
                message = "B@" if found_best else "  "
                best_counter = best_counter if found_best else epoch_counter
                o = compute_all_jit(params_print)
                if jnp.isnan(o['error_h2']):
                    print("Nan error")
                    print(o)
                    return {"params_best": params_print, "values": compute_all_jit(params_print)}
                    
                message = message + (
                                    f"epoch {best_counter}: "
                                    f"error = {1000.0 * o['error_h2']:.1f}mHa, " 
                                    f"λ_(eff,df,1,2),=({o['lambda_eff']:.2f}, {o['lambda_DFTHC']:.2f}, {o['lambda_H1']:.2f}, {o['lambda_H2']:.1f})Ha, "
                                    f"E∈[{o['H0']-o['lambda_H1']-o['error_h2_ord=1']:.1f},{o['H0']+o['lambda_H1']+o['Spectral norm above zero']+o['error_h2_ord=1']:.1f}]Ha, "
                                    f"cost = {1000.0 * o['cost']:.1f}mHa, "
                                    f"H0: {o['H0']:.3f}Ha, "
                                    f"gap: {o['sos_gap']:.3f}Ha, "
                                    f"ms/it: {1000.0 * (time.time() - time_start) / time_counter:.2f}")
                print(message)        
                time_start = time.time()
                time_counter = 0
                found_best = False
        epoch_counter += 1
    '''
    # Print some outputs
    generating_sets[generating_set][2](complete_params(params_best),ref,generators)
    
    return {"params_best": complete_params(params_best), "values": compute_all_jit(params_best)}

''' 
Example usage:
python train_df_thc_sga.py '../integrals/XVIII-highCD-cas4-fb-64e56o.txt' 'Spin-symmetric' 'DF-THC' 6 100 56 1000 0.001 0.001 --generators 'g2_aa/ab'
# Add --resume previous.npz --fileout resumed.npz to continue an NPZ solution.
'''

def main():
    # Reuse the starter's baseline and no-overwrite protections for CLI runs.
    from starter.task1_factorization.train_df_thc_qaggle import check_output_file

    parser = argparse.ArgumentParser(
                    prog='DF-THC-SOS',
                    description='Find DF-THC-SOS solution',
                    epilog='[Epilog]')
    parser.add_argument('filename', type = str)
    parser.add_argument('generating_set', type = str)      
    parser.add_argument('method', type = str)                        
    parser.add_argument('rank', type = int)                  
    parser.add_argument('bases', type = int)                      
    parser.add_argument('copies', type = int)               
    parser.add_argument('epochs', type = int)            
    parser.add_argument('learning_rate', type = float)            
    parser.add_argument('rho', type = float)                 
    parser.add_argument('--energy_upper', type = float, required = False, default = 0.0)      
    parser.add_argument('--key', type = int, required = False, default = 200) 
    parser.add_argument('--fileout', type = str, required = False,
                        help='New .npz output file; existing files are never overwritten.')
    #parser.add_argument('--H2_I', type = bool, required = False, default = False)
    #parser.add_argument('--H_symm', type = bool, required = False, default = True)
    parser.add_argument('--resume', type = str, required = False,
                        help='Seed parameters from a previous .npz solution.')
    parser.add_argument('--generators', nargs='+', required=False, default = None) 
    parser.add_argument('--spin', type = float, required = False, default = 0.) 
    parser.add_argument('--num_alpha', required = False, default = None) 
    
    args = parser.parse_args()
    fileout = check_output_file(
        args.fileout if args.fileout is not None else f"{args.filename}_{args.method}.npz"
    )
    resume_params = read_factorization(args.resume)[0] if args.resume else None
    print(args)
    print(f"Has GPU: {jax.default_backend()}")

    if args.generating_set in ['Spin-symmetric','General']:
        if args.generators is None:
            print(f"--generators argument required for {args.generating_set}")
    
    # Key for jax random number generation
    '''
    if args.energy_upper is None:
        if args.method in ['DF-THC-SOS', 'DF-THC-SOS Max Gap', 'Max SOS Gap']:
            print(f"--energy_upper argument required for {method}")
            return 0
        energy_upper = 0.0
    '''
    if args.generators is not None:
        print(args.generators)
    ref = Initialize(args.filename, args.energy_upper)
    params = Params(args.generating_set, args.rank, args.bases, args.copies, ref['orbitals'], args.key, args.generators)
    ref['spin'] = args.spin
    ref['num_alpha'] = args.num_alpha if args.num_alpha is not None else int(ref['electrons']/2)
    ref['num_beta'] = ref['electrons'] - ref['num_alpha']
    ref['energy_upper'] = args.energy_upper
    

    
    # Energy upper is used to compute SOS gaps.
    if resume_params is not None:
        params = {key: jnp.asarray(value, dtype=df_thc_sga.dtype)
                  for key, value in resume_params.items()}
        
    solution = Train(generating_set = args.generating_set,
                    method = args.method,
                    epochs = args.epochs, 
                    learning_rate = args.learning_rate, 
                    rho = args.rho, 
                    params = params,
                    ref = ref,
                    print_interval = 1000,
                    epoch_start = 0,
                    generators = args.generators,
                    energy_upper = ref['energy_upper']
    )
    
    check_output_file(fileout)
    save_solution_npz(fileout, solution['params_best'], values=solution['values'])

def Params(generating_set: str, rank: int, bases: int, copies: int, orbitals: int, seed: int = 0, generators = None):
    if generators is None:
        return generating_sets[generating_set][1](rank, bases, copies, orbitals, seed)
    else:
        return generating_sets[generating_set][1](rank, bases, copies, orbitals, seed, generators)
    
def Initialize(filename, energy_upper, format: str = "FCIDUMP"):
    if format == "FCIDUMP":
        data = load_integrals.fci_from_fcidump(filename)
    else:
        raise ValueError(f"Unsupported integral format: {format!r}")
    ref = data["ref"]
    ref['energy_upper'] = energy_upper
    return ref

if __name__ == "__main__":
    main()
