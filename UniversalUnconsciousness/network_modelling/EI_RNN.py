from math import sqrt
import torch
from tqdm.auto import tqdm

class EI_RNN:
    def __init__(self, g, alpha, N, K, m_0, tau, propofol_scale=1.0, ketamine_scale=1.0, ketamine_dose='low', build_network=True):
        # g: global coupling strength
        # alpha: inhibition-to-excitation ratio
        # N_E: number of excitatory neurons
        # N_I: number of inhibitory neurons
        # K: number of connections per neuron
        # m_0: external activity
        # tau: time constant
        # propofol_scale: scale factor for propofol
        self.g = g
        self.alpha = alpha
        self.N = N
        self.K = K
        self.m_0 = m_0
        self.tau = tau
        self.propofol_scale = propofol_scale
        self.ketamine_scale = ketamine_scale
        self.ketamine_dose = ketamine_dose
        if build_network:
            self.build_network()

    def build_network(self):
        J_EE = self.alpha * self.g * self.ketamine_scale if self.ketamine_dose != 'low' else self.alpha * self.g
        J_IE = self.alpha * self.g * self.ketamine_scale
        J_II = -self.g * self.propofol_scale
        J_EI = -1.11 * self.g * self.propofol_scale
        W_E = self.alpha * self.g
        W_I = 0.44*self.g

        g_EE = sqrt((1 - self.K/self.N) * J_EE**2)
        g_IE = sqrt((1 - self.K/self.N) * J_IE**2)
        g_EI = sqrt((1 - self.K/self.N) * J_EI**2)
        g_II = sqrt((1 - self.K/self.N) * J_II**2)

        g_bar_EE = sqrt(self.K) * J_EE
        g_bar_IE = sqrt(self.K) * J_IE
        g_bar_EI = sqrt(self.K) * J_EI
        g_bar_II = sqrt(self.K) * J_II

        J_ij_EE = torch.randn(self.N, self.N) * g_EE / sqrt(self.N) + (g_bar_EE / self.N)
        J_ij_IE = torch.randn(self.N, self.N) * g_IE / sqrt(self.N) + (g_bar_IE / self.N)
        J_ij_EI = torch.randn(self.N, self.N) * g_EI / sqrt(self.N) + (g_bar_EI / self.N)
        J_ij_II = torch.randn(self.N, self.N) * g_II / sqrt(self.N) + (g_bar_II / self.N)

        self.J = torch.cat([torch.cat([J_ij_EE, J_ij_EI], dim=1), torch.cat([J_ij_IE, J_ij_II], dim=1)], dim=0)
        self.W = torch.cat([W_E*torch.ones(self.N), W_I*torch.ones(self.N)])

    def forward(self, t, h):
        return (1/self.tau) * (-h + self.J @ torch.tanh(h) + self.W *self.m_0)

    def jac(self, h):
        return (1/self.tau) * (-torch.eye(2*self.N).to(h.device) + self.J @ torch.diag_embed((1 - torch.tanh(h)**2)))

    def to(self, device):
        self.J = self.J.to(device)
        self.W = self.W.to(device)
        return self

    def parameter_dict(self):
        param_dict = dict(
            g=self.g,
            alpha=self.alpha,
            N=self.N,
            K=self.K,
            m_0=self.m_0,
            tau=self.tau,
            propofol_scale=self.propofol_scale,
            ketamine_scale=self.ketamine_scale,
            ketamine_dose=self.ketamine_dose,
            J=self.J,
            W=self.W
        )
        return param_dict

    @classmethod
    def from_dict(cls, param_dict):
        """Initialize an EI_RNN instance from a parameter dictionary."""
        # Create instance with required parameters
        instance = cls(
            g=param_dict['g'],
            alpha=param_dict['alpha'],
            N=param_dict['N'],
            K=param_dict['K'],
            m_0=param_dict['m_0'],
            tau=param_dict['tau'],
            propofol_scale=param_dict['propofol_scale'],
            ketamine_scale=param_dict['ketamine_scale'],
            ketamine_dose=param_dict['ketamine_dose'],
            build_network=False if ('J' in param_dict and 'W' in param_dict) else True
        )
        
        # Optionally override the computed J and W if provided
        if 'J' in param_dict:
            instance.J = param_dict['J']
        if 'W' in param_dict:
            instance.W = param_dict['W']
            
        return instance

class EI_RNN_w_NMDA:
    def __init__(self, g, alpha, N, K, m_0, tau, propofol_scale=1.0, ketamine_scale=1.0, ketamine_dose='low', build_network=True):
        # g: global coupling strength
        # alpha: inhibition-to-excitation ratio
        # N_E: number of excitatory neurons
        # N_I: number of inhibitory neurons
        # K: number of connections per neuron
        # m_0: external activity
        # tau: time constant
        # propofol_scale: scale factor for propofol
        self.g = g
        self.alpha = alpha
        self.N = N
        self.K = K
        self.m_0 = m_0
        self.tau = tau
        self.propofol_scale = propofol_scale
        self.ketamine_scale = ketamine_scale
        self.ketamine_dose = ketamine_dose
        if build_network:
            self.build_network()

    def build_network(self):
        J_EE = self.alpha * self.g
        J_EE = J_EE * self.ketamine_scale if self.ketamine_dose != 'low' else J_EE
        J_IE = self.alpha * self.g * self.ketamine_scale
        J_II = -self.g * self.propofol_scale
        J_EI = -1.11 * self.g * self.propofol_scale
        W_E = self.alpha * self.g
        W_I = 0.44*self.g

        g_EE = sqrt((1 - self.K/self.N) * J_EE**2)
        g_IE = sqrt((1 - self.K/self.N) * J_IE**2)
        g_EI = sqrt((1 - self.K/self.N) * J_EI**2)
        g_II = sqrt((1 - self.K/self.N) * J_II**2)

        g_bar_EE = sqrt(self.K) * J_EE
        g_bar_IE = sqrt(self.K) * J_IE
        g_bar_EI = sqrt(self.K) * J_EI
        g_bar_II = sqrt(self.K) * J_II

        J_ij_EE = torch.randn(self.N, self.N) * g_EE / sqrt(self.N) + (g_bar_EE / self.N)
        J_ij_IE = torch.randn(self.N, self.N) * g_IE / sqrt(self.N) + (g_bar_IE / self.N)
        J_ij_EI = torch.randn(self.N, self.N) * g_EI / sqrt(self.N) + (g_bar_EI / self.N)
        J_ij_II = torch.randn(self.N, self.N) * g_II / sqrt(self.N) + (g_bar_II / self.N)

        self.J = torch.cat([torch.cat([J_ij_EE, J_ij_EI], dim=1), torch.cat([J_ij_IE, J_ij_II], dim=1)], dim=0)
        self.W = torch.cat([W_E*torch.ones(self.N), W_I*torch.ones(self.N)])

    def forward(self, t, h):
        return (1/self.tau) * (-h + self.J @ torch.tanh(h) + self.W *self.m_0)

    def jac(self, h):
        return (1/self.tau) * (-torch.eye(2*self.N).to(h.device) + self.J @ torch.diag_embed((1 - torch.tanh(h)**2)))

    def to(self, device):
        self.J = self.J.to(device)
        self.W = self.W.to(device)
        return self

    def parameter_dict(self):
        param_dict = dict(
            g=self.g,
            alpha=self.alpha,
            N=self.N,
            K=self.K,
            m_0=self.m_0,
            tau=self.tau,
            propofol_scale=self.propofol_scale,
            ketamine_scale=self.ketamine_scale,
            ketamine_dose=self.ketamine_dose,
            J=self.J,
            W=self.W
        )
        return param_dict

    @classmethod
    def from_dict(cls, param_dict):
        """Initialize an EI_RNN instance from a parameter dictionary."""
        # Create instance with required parameters
        instance = cls(
            g=param_dict['g'],
            alpha=param_dict['alpha'],
            N=param_dict['N'],
            K=param_dict['K'],
            m_0=param_dict['m_0'],
            tau=param_dict['tau'],
            propofol_scale=param_dict['propofol_scale'],
            ketamine_scale=param_dict['ketamine_scale'],
            ketamine_dose=param_dict['ketamine_dose'],
            build_network=False if ('J' in param_dict and 'W' in param_dict) else True
        )
        
        # Optionally override the computed J and W if provided
        if 'J' in param_dict:
            instance.J = param_dict['J']
        if 'W' in param_dict:
            instance.W = param_dict['W']
            
        return instance

def compute_lyaps_from_sol(sol_in, jac_func, dt=1, k=None, verbose=False):
    squeeze = False
    if len(sol_in.shape) == 2:
        sol = sol_in.unsqueeze(0)
        squeeze = True
    else:
        sol = sol_in

    T, n = sol.shape[-2], sol.shape[-1]
    old_Q = torch.eye(n, device=sol.device, dtype=sol.dtype)

    if k is None:
        k = n

    old_Q = old_Q[:, :k]
    lexp = torch.zeros(*sol.shape[:-2], k, device=sol.device, dtype=sol.dtype)
    lexp_counts = torch.zeros(*sol.shape[:-2], k, device=sol.device, dtype=sol.dtype)

    for t in tqdm(range(T), disable=not verbose):
            
        # QR-decomposition of Js[t] * old_Q
        J = jac_func(sol[..., t, :])*dt + torch.eye(n, device=sol.device, dtype=sol.dtype) # Linear approximation of Jacobian
        mat_Q, mat_R = torch.linalg.qr(torch.matmul(J, old_Q))
        
        # force diagonal of R to be positive
        # sign_diag = torch.sign(torch.diag(mat_R))
        diag_R = mat_R.diagonal(dim1=-2, dim2=-1)
        sign_diag = torch.sign(diag_R)
        sign_diag[sign_diag == 0] = 1
        sign_diag = torch.diag_embed(sign_diag)
        
        mat_Q = mat_Q @ sign_diag
        mat_R = sign_diag @ mat_R
        old_Q = mat_Q
        
        # Successively build sum for Lyapunov exponents
        diag_R = mat_R.diagonal(dim1=-2, dim2=-1)

        # Filter zeros in mat_R (would lead to -infs)
        idx = diag_R > 0
        lexp_i = torch.zeros_like(diag_R, dtype=sol.dtype, device=sol.device)
        lexp_i[idx] = torch.log(diag_R[idx])
        lexp[idx] += lexp_i[idx]
        lexp_counts[idx] += 1
    
    if squeeze:
        lexp = lexp.squeeze(0)
        lexp_counts = lexp_counts.squeeze(0)
    
    return torch.flip(torch.sort((lexp / lexp_counts) * (1 / dt), axis=-1)[0], dims=[-1])