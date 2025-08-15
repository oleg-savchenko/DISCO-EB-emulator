import jax
import jax.numpy as jnp
from jax import config
from discoeb.background import evolve_background
from discoeb.perturbations import evolve_perturbations, evolve_perturbations_batched, get_power
import numpy as np
config.update("jax_enable_x64", True)
dtype = jnp.float64

training_data = np.load('./training_data/training_data_disco_eb_custom_k.npz')
training_data = {key: training_data[key] for key in training_data.files}
cosmo_params_samples = training_data['cosmo_params_samples']
kmodes_in_Mpc_h = training_data['k_modes']
print("k ranges:", kmodes_in_Mpc_h.min(), kmodes_in_Mpc_h.max())

@jax.jit
def compute_matter_power(cosmo_params, kmodes_in_Mpc_h):
    ob_h2, omc_h2, h, ns, lnAs, z = cosmo_params
    param = {}
    param['Omegam'] = (omc_h2 + ob_h2) / h**2
    param['Omegab'] = ob_h2 / h**2
    param['w_DE_0'] = -0.99
    param['w_DE_a'] = 0.0
    param['cs2_DE'] = 0.99  # sound speed of DE
    param['Omegak'] = 0.0  # NOTE: Omegak is ignored at the moment!
    param['A_s'] = jnp.exp(lnAs) * 1e-10
    param['n_s'] = ns
    param['H0'] = h * 100  # Hubble constant in km/s/Mpc
    param['Tcmb'] = 2.72548  # CMB temperature in K
    param['YHe'] = 0.248
    Neff = 3.046  # -1 if massive neutrino present
    N_nu_mass = 1.0
    Tnu = (4.0 / 11.0) ** (1.0 / 3.0)
    N_nu_rel = Neff - N_nu_mass * (Tnu / ((4.0 / 11.0) ** (1.0 / 3.0))) ** 4
    param['Neff'] = N_nu_rel
    param['Nmnu'] = N_nu_mass
    param['mnu'] = 0.06  # eV
    param['k_p'] = 0.05  # pivot scale for the primordial power spectrum in Mpc^-1
    param_bg = evolve_background(param=param, thermo_module='RECFAST')
    aexp_out = 1/(1 + z)

    # Compute perturbations
    y, kmodes = evolve_perturbations(param=param_bg, kmodes=h*kmodes_in_Mpc_h, #kmin=1e-5, kmax=1e1, num_k=256,
                        aexp_out=jnp.array([aexp_out]), lmaxg=11, lmaxgp=11, lmaxr=11, lmaxnu=11,
                        nqmax=3, max_steps=2048, rtol=1e-4, atol=1e-4)

    # Get the power spectrum at z=0
    Pk = get_power(k=kmodes, y=y[:, 0, :], idx=4, param=param_bg)

    # Convert to /h units
    # h = param['H0'] / 100.0
    Pk_in_Mpc_h = Pk * h** 3
    return Pk_in_Mpc_h, kmodes_in_Mpc_h

print("Running first sample.")
Pk_in_Mpc_h, kmodes_in_Mpc_h = compute_matter_power(cosmo_params=cosmo_params_samples[0], kmodes_in_Mpc_h=kmodes_in_Mpc_h)


_compute_matter_power_batched_impl = jax.vmap(compute_matter_power, in_axes=(0, None))

def compute_matter_power_batched(cosmo_params_batch, kmodes_in_Mpc_h):
    batch = _compute_matter_power_batched_impl(cosmo_params_batch, kmodes_in_Mpc_h)
    Pk_batch, kmodes_in_Mpc_h = batch[0], batch[1][0]
    return Pk_batch, kmodes_in_Mpc_h

Pks = []
for i in range(20):
    print(f"Running batch {i}, simulations #{500*i} to #{500*(i + 1) - 1}.")
    cosmo_params_batch = cosmo_params_samples[500*i:500*(i + 1)]
    Pk_in_Mpc_h_batch, kmodes_in_Mpc_h = compute_matter_power_batched(cosmo_params_batch, kmodes_in_Mpc_h)
    Pks.append(Pk_in_Mpc_h_batch)
Pks = np.concatenate(Pks, axis=0)

# Pks = []
# for i in range(2):
#     print(f"Running batch {i}.")
#     cosmo_params_batch = cosmo_params_samples[10*i:10*(i + 1)]
#     Pk_in_Mpc_h_batch, kmodes_in_Mpc_h = compute_matter_power_batched(cosmo_params_batch, kmodes_in_Mpc_h)
#     Pks.append(Pk_in_Mpc_h_batch)
# Pks = np.concatenate(Pks, axis=0)

# training_data['k_modes'] = np.array(kmodes_in_Mpc_h)
training_data['power_spectra'] = np.array(Pks)
print(Pks.shape)
np.savez('./training_data/training_data_disco_eb_custom_k.npz', **training_data)


