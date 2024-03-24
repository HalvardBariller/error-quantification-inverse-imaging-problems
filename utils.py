import numpy as np
from torch.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import torch
import tqdm
# viewimage
import tempfile
import IPython
from skimage.transform import rescale

pi = torch.pi


def str2(chars):
    return "{:.2f}".format(chars)


######################################################################
##############  Visualization functions 
######################################################################

def rgb2gray(u):
    return 0.2989 * u[:,:,0] + 0.5870 * u[:,:,1] + 0.1140 * u[:,:,2]

def viewimage(im, normalize=True,z=2,order=0,titre='',displayfilename=False):
    imin= np.array(im).copy().astype(np.float32)
    imin = rescale(imin, z, order=order)
    if normalize:
        imin-=imin.min()
        if imin.max()>0:
            imin/=imin.max()
    else:
        imin=imin.clip(0,255)/255
    imin=(imin*255).astype(np.uint8)
    filename=tempfile.mktemp(titre+'.png')
    if displayfilename:
        print (filename)
    plt.imsave(filename, imin, cmap='gray')
    IPython.display.display(IPython.display.Image(filename))

# alternative viewimage if the other one does not work:
def Viewimage(im,dpi=100,cmap='gray'):
    plt.figure(dpi=dpi)
    if cmap is None:
        plt.imshow(np.array(im))
    else:
        plt.imshow(np.array(im),cmap=cmap)
    plt.axis('off')
    plt.show()



######################################################################
##############  Metric and optimization functions  ###################
######################################################################
    
def psnr(uref,ut,M=1):
    mse = np.sqrt(np.mean((np.array(uref)-np.array(ut))**2))
    return 20*np.log10(M/mse)

def optim(f, image_shape, niter=1000,lr=0.1):
    M,N = image_shape
    u = torch.randn(M,N, requires_grad=True)
    # optimu = torch.optim.SGD([u], lr=lr)
    optimu = torch.optim.Adam([u], lr=lr)
    losslist = []
    for it in range(niter):
        loss = f(u)
        losslist.append(loss.detach())
        optimu.zero_grad()
        loss.backward()
        optimu.step()
    return u.detach(),losslist


######################################################################
##############  Forward and adjoint operators  #######################
######################################################################

def forward_operator(x, k):
    """Compute the forward operator of x with kernel k.
    -------------------
    Parameters:
    x: torch.Tensor
        Image to compute the forward operator of.
    k: torch.Tensor
        Kernel.
    """
    return ifft2(fft2(k) * fft2(x)).real



######################################################################
##############  Bayesian functions  ##################################
######################################################################

def data_fidelity(z, x, sigma, k):
    """Compute the data fidelity term (- log p(z|x)) of z.
    -------------------
    Parameters:
    z: torch.Tensor
        Image to compute the data fidelity term of.
    x: torch.Tensor
        Original image.
    sigma: float
        Noise level.
    k: torch.Tensor
        Kernel for the forward operator.
    """
    norm = torch.linalg.vector_norm(forward_operator(x, k) - z)
    return 0.5 * torch.square(norm) / sigma**2


def smoothed_TV(x, eps=1e-3):
    """Compute the smoothed total variation norm of x.
    -------------------
    Parameters:
    x: torch.Tensor
        Image to compute the TV norm of.
    eps: float
        Smoothing parameter.
    """
    d1x = torch.roll(x, -1, 0) - x
    d2x = torch.roll(x, -1, 1) - x
    Dx = torch.sqrt(torch.square(d1x) + torch.square(d2x) + eps**2)
    return torch.sum(Dx)

def regularization_smoothed_TV(x, lambda_, eps=1e-3):
    """Compute the regularization term (- log p(x)) of x with smoothed TV norm.
    -------------------
    Parameters:
    x: torch.Tensor
        Image to compute the regularization term of.
    """
    return smoothed_TV(x, eps) / lambda_


def energy_functional_smoothedTV(x, z, lambda_, sigma, k, eps=1e-3):
    """Compute the energy functional of x.
    -------------------
    Parameters:
    x: torch.Tensor
        Original image.
    z: torch.Tensor
        Noisy image.
    lambda_: float
        Regularization parameter.
    sigma: float
        Noise level.
    k: torch.Tensor
        Kernel for the forward operator.
    regularization: function
        Regularization function.
    eps: float
        Smoothing parameter.
    """
    return regularization_smoothed_TV(x, lambda_, eps) + data_fidelity(z, x, sigma, k)


######################################################################
##############  Bayesian Estimation  #################################
######################################################################
    

def MAP(x, z, lambda_, sigma, k, eps=1e-3, lr=1e-2, niter=1000):
    """Compute the Maximum A Posteriori estimate of x.
    -------------------
    Parameters:
    x: torch.Tensor
        Original image.
    z: torch.Tensor
        Noisy image.
    lambda_: float
        Regularization parameter.
    sigma: float
        Noise level.
    k: torch.Tensor
        Kernel for the forward operator.
    eps: float
        Smoothing parameter.
    """
    M,N = x.shape
    E = lambda u: energy_functional_smoothedTV(u, z, lambda_, sigma, k, eps)
    return optim(E, (M,N), niter=niter, lr=lr)


def MMSE(chain_iterates):
    """Compute the MMSE estimate given the chain iterates obtained by sampling.
    -------------------
    Parameters:
    chain_iterates: List[torch.Tensor]
        List of chain iterates.
    """
    return torch.mean(torch.stack(chain_iterates), dim=0)


def posterior_variance(chain_iterates):
    """Compute the posterior variance given the chain iterates obtained by sampling.
    -------------------
    Parameters:
    chain_iterates: List[torch.Tensor]
        List of chain iterates.
    """
    return torch.var(torch.stack(chain_iterates), dim=0)



######################################################################
##############  Sampling functions  ##################################
######################################################################

def unadjusted_langevin_dynamics_TVSmoothed(z, sigma, k, lambda_=0.15, eps=1e-3, tau=1e-4, niter=2000):
    """Compute a sequence of iterates of x using ULA.
    -------------------
    Parameters:
    z: torch.Tensor
        Noisy image.
    sigma: float
        Noise level.
    k: torch.Tensor
        Kernel for the forward operator.
    lambda_: float
        Regularization parameter.
    eps: float
        Smoothing parameter.
    tau: float
        Step size.
    niter: int
        Number of iterations.
    """
    burn_in = max(int(0.9 * niter), niter - 1000)
    M,N = z.shape
    early_stopping = 1e-2
    E = lambda u: energy_functional_smoothedTV(u, z, lambda_, sigma, k, eps)
    x_t = torch.randn(M, N, requires_grad=True)
    energies = []
    chain_iterates = []
    for it in tqdm.tqdm(range(niter)):
        energy = E(x_t)
        energies.append(energy.detach())
        grad = torch.autograd.grad(energy, x_t)[0]
        x_t = x_t - tau * grad + torch.randn(M, N) * np.sqrt(2 * tau)
        x_t = x_t.detach().requires_grad_(True)
        if it > burn_in and it % 2 == 0:
            chain_iterates.append(x_t.detach())
        if len(chain_iterates) > 2 and torch.linalg.vector_norm(chain_iterates[-1] - chain_iterates[-2]) < early_stopping:
            print(f'Early stopping at iteration {it}')
            break
    return energies, chain_iterates


###########################################################################################
##### Unadjusted Langevin Primal-Dual Algorithm (ULPDA) for TV-l2 regularization ##########
###########################################################################################

def Dstar(pbar_k):
    ret = torch.zeros((pbar_k.size(0), pbar_k.size(1)), dtype=torch.double)
    ret[1:] += pbar_k[:-1, :, 0]
    ret[:-1] -= pbar_k[:-1, :, 0]
    ret[:, 1:] += pbar_k[:, 1:, 1]
    ret[:, :-1] -= pbar_k[:, :-1, 1]
    return ret

def D(x):
    ret = torch.zeros((x.size(0), x.size(1), 2), dtype=torch.double)
    ret[:-1, :, 0] = x[1:] - x[:-1]
    ret[:, :-1, 1] = x[:, 1:] - x[:, :-1]
    return ret

def prox_tauk_g(v, z, sigma, tau_k):
    return 1/(1+tau_k/sigma**2)*(tau_k/sigma**2*z + v)

def prox_sigmak_fstar(v, s_k, lamb):
    return torch.minimum(torch.tensor(1/lamb, dtype=v.dtype), torch.maximum(torch.tensor(-1/lamb, dtype=v.dtype), v))

def ulpda(z, x0, p0, sigma, lamb, s_k, tau_k, theta_k, K=1000, burnin=500):
    """
    Unadjusted Langevin primal-dual algorithm for TV-l2 regularisation.

    Parameters
    ----------
    x0 : (M, N) numpy.ndarray
        Initial image sample.
    p0 : (M, N, 2) numpy.ndarray
        Initial auxiliary vector.
    sigma : float
        Standard deviation of the noise added to the image.
    s_k : float
        $\\sigma_k$.
    tau_k : float
        $\\tau_k$.
    theta_k : float
        $\\theta_k$.
    """
    x = x0.detach().clone()
    p = p0.detach().clone()
    p_prev = None
    xhat = torch.zeros_like(x0)
    count = 0
    M2 = torch.zeros_like(x0)
    for k in range(K):
        pbar_k = p if k==0 else p + theta_k*(p-p_prev)
        v = x - tau_k*Dstar(pbar_k)
        xi_k = torch.randn(x.size())
        x = prox_tauk_g(v, z, sigma, tau_k) + (2*tau_k)**.5*xi_k
        p_prev = p
        p = prox_sigmak_fstar(p + s_k*D(x), s_k, lamb)
        if k >= burnin:
            count += 1
            delta = x-xhat
            xhat += delta/count
            delta2 = x-xhat
            M2 += delta*delta2
    return xhat, M2/count
