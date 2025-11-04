import torch
from torch.autograd.functional import jvp, vjp
from torch.func import functional_call, jacrev, vmap


# Helper functions
def gini(w: torch.Tensor) -> torch.Tensor:
    r"""The Gini coefficient from the `"Improving Molecular Graph Neural
    Network Explainability with Orthonormalization and Induced Sparsity"
    <https://arxiv.org/abs/2105.04854>`_ paper.

    Computes a regularization penalty :math:`\in [0, 1]` for each row of a
    matrix according to

    .. math::
        \mathcal{L}_\textrm{Gini}^i = \sum_j^n \sum_{j'}^n \frac{|w_{ij}
         - w_{ij'}|}{2 (n^2 - n)\bar{w_i}}

    and returns an average over all rows.

    Args:
        w (torch.Tensor): A two-dimensional tensor.
    """
    s = 0
    for row in w:
        t = row.repeat(row.size(0), 1)
        u = (t - t.T).abs().sum() / (2 * (row.size(-1)**2 - row.size(-1)) *
                                     row.abs().mean() + torch.finfo().eps)
        s += u
    s /= w.shape[0]
    return s

def get_batch_jacobian(func, inputs, chunk_size=64):
    params = dict(func.named_parameters())

    def fmodel(params, inputs):
        return functional_call(func, params, inputs.flatten().unsqueeze(0)).flatten()

    jacobians = []
    for i in range(0, inputs.size(0), chunk_size):
        chunk = inputs[i:i+chunk_size]
        chunk_jacobians = vmap(jacrev(fmodel, argnums=(1)), in_dims=(None,0))(params, chunk)
        jacobians.append(chunk_jacobians)
        
    jacobians =  torch.cat(jacobians, dim=0)
    torch.cuda.empty_cache()
    return jacobians

def get_JTJ_trace_estimate(func, inputs, chunk_size=64, num_samples=20, create_graph=False):
    bs = len(inputs)
    
    # Initialize result tensor
    trace_estimates = torch.zeros(bs, device=inputs.device)
    
    # Process in chunks
    for i in range(0, bs, chunk_size):
        end_idx = min(i + chunk_size, bs)
        chunk = inputs[i:end_idx]
        chunk_bs = end_idx - i
        
        chunk_traces = []
        for _ in range(num_samples):
            v = torch.randn_like(chunk)
            Jv = jvp(func, chunk, v=v, create_graph=create_graph)[1]
            trace_sample = torch.sum(Jv.view(chunk_bs, -1)**2, dim=1)
            chunk_traces.append(trace_sample)
        
        # Average over samples for this chunk
        chunk_trace = torch.stack(chunk_traces, dim=0).mean(dim=0)
        trace_estimates[i:end_idx] = chunk_trace
        
        # Clear memory after each chunk
        torch.cuda.empty_cache()
    
    return trace_estimates


# Evaluation function for conformality
def evaluate_conformality(model, data, double_precision=False):
    model.eval()
    if double_precision:
        model.double()
        data = data.double()
    with torch.no_grad():
        data_dim = data.shape[1]
        # latent samples
        latent = model.encode(data)
        if model.name == "VariationalAutoencoder":
            # for VAE, we sample from the posterior
            # latent = model.sample(latent)
            latent = model.reparameterize(latent[0], latent[1])
        latent_dim = latent.shape[1]

        # Compute the Jacobian of the decoder
        jacobians = get_batch_jacobian(model.decoder, latent)
        jTjs = torch.einsum('bji,bjk->bik', jacobians, jacobians)
        traces = vmap(torch.trace)(jTjs)
        lambda_factors = traces / latent.shape[1]

        # Estimate the trace of J^T J
        trace_estimate = get_JTJ_trace_estimate(model.decoder, latent)
        lambda_factors_estimate = trace_estimate / latent.shape[1]
        lambda_factors_meanerror = torch.mean(torch.abs(lambda_factors - lambda_factors_estimate))
        lambda_factors_meanerror_normalized = lambda_factors_meanerror / (lambda_factors.mean() + torch.finfo(torch.float32).eps)

        # gini for diagonal elements
        diagonals = jTjs.diagonal(dim1=1, dim2=2)
        gini_value = gini(diagonals)

        # off diag mean, norm
        off_diagonal = jTjs - torch.diag_embed(diagonals)
        off_diag_mean = off_diagonal.abs().mean(dim=(1, 2))
        off_diag_norm = off_diagonal.norm(dim=(1, 2))
        off_diag_mean_normed = off_diag_mean / lambda_factors
        off_diag_norm_normed = off_diag_norm / lambda_factors

        # jTj - lambda*I mean, norm
        eye = torch.eye(jTjs.shape[1], device=jTjs.device).repeat(jTjs.shape[0], 1, 1)
        jTj_minus_lambdaI = jTjs - lambda_factors.unsqueeze(1).unsqueeze(2).repeat(1, jTjs.shape[1], jTjs.shape[1]) * eye
        jTj_minus_lambdaI_mean = jTj_minus_lambdaI.abs().mean(dim=(1, 2))
        jTj_minus_lambdaI_norm = jTj_minus_lambdaI.norm(dim=(1, 2))
        jTj_minus_lambdaI_mean_normed = jTj_minus_lambdaI_mean / lambda_factors
        jTj_minus_lambdaI_norm_normed = jTj_minus_lambdaI_norm / lambda_factors

        # determinant of jacobian vs sqrt lambda**m
        jacobian_determinants = torch.sqrt(torch.linalg.det(jTjs))
        estimate_determinants = torch.sqrt(lambda_factors**latent_dim)
        estimate_estimate_determinants = torch.sqrt(lambda_factors_estimate**latent_dim)
        # old version where optimum == 1
        # determinant_vs_estimate = jacobian_determinants / (estimate_determinants + torch.finfo(torch.float32).eps)
        determinant_vs_estimate = torch.abs(estimate_determinants - jacobian_determinants) / (torch.abs(jacobian_determinants) + torch.finfo(torch.float32).eps)
        determinant_vs_estimate_meanerror = torch.mean(torch.abs((jacobian_determinants - estimate_determinants) / jacobian_determinants))
        # old version where optimum == 1
        # determinant_vs_estimate_estimate = jacobian_determinants / (estimate_estimate_determinants + torch.finfo(torch.float32).eps)
        determinant_vs_estimate_estimate = torch.abs((estimate_estimate_determinants - jacobian_determinants) / (torch.abs(jacobian_determinants) + torch.finfo(torch.float32).eps))

        # log determinant of jacobian vs 0.5 * latent_dim * log(lambda)
        jacobian_log_determinants = 0.5 * torch.logdet(jTjs)
        log_det_estimate = 0.5 * latent_dim * torch.log(lambda_factors)
        log_det_estimate_estimate = 0.5 * latent_dim * torch.log(lambda_factors_estimate)
        log_determinant_vs_estimate = torch.abs(jacobian_log_determinants - log_det_estimate)
        log_determinant_vs_estimate_estimate = torch.abs(jacobian_log_determinants - log_det_estimate_estimate)

        # std of latent space
        latent_std = latent.std(dim=0)
        
        return {
            'reconstruction_error': torch.nn.MSELoss()(data, model.decode(latent)).item(),
            'diagonal_gini': gini_value.item(),
            'lambda_mean': lambda_factors.mean().item(),
            'lambda_std': lambda_factors.std().item(),
            'lambda_std_normed': (lambda_factors.std() /lambda_factors.mean()).item(),
            'lambda_factors_meanerror': lambda_factors_meanerror.item(),
            'lambda_factors_meanerror_normalized': lambda_factors_meanerror_normalized.item(),
            'off_diag_mean': off_diag_mean.mean().item(),
            'off_diag_norm': off_diag_norm.mean().item(),
            'off_diag_mean_normed': off_diag_mean_normed.mean().item(),
            'off_diag_norm_normed': off_diag_norm_normed.mean().item(),
            'jTj_minus_lambdaI_mean': jTj_minus_lambdaI_mean.mean().item(),
            'jTj_minus_lambdaI_norm': jTj_minus_lambdaI_norm.mean().item(),
            'jTj_minus_lambdaI_mean_normed': jTj_minus_lambdaI_mean_normed.mean().item(),
            'jTj_minus_lambdaI_norm_normed': jTj_minus_lambdaI_norm_normed.mean().item(),
            'determinant_vs_estimate_mean': determinant_vs_estimate.mean().item(),
            'determinant_vs_estimate_std': determinant_vs_estimate.std().item(),
            'determinant_vs_estimate_meanerror': determinant_vs_estimate_meanerror.item(),
            'determinant_vs_estimate_estimate_mean': determinant_vs_estimate_estimate.mean().item(),
            'log_determinant_vs_estimate_mean': log_determinant_vs_estimate.mean().item(),
            'log_determinant_vs_estimate_std': log_determinant_vs_estimate.std().item(),
            'log_determinant_vs_estimate_estimate_mean': log_determinant_vs_estimate_estimate.mean().item(),
            'latent_std': latent_std.mean().item(),
            'latent_std_max': latent_std.max().item(),
            'latent_std_min': latent_std.min().item(),
            'latent_norm': latent.norm().item(),
        }


# Isometry loss functions
def isometry_loss(func, z, create_graph=True):
    """
    z: (batch_size, latent_dim) latent vectors sampled from Piso
    """
    # Sample u ~ Uniform(S^{d-1}), i.e., unit vector on sphere
    u = torch.randn_like(z)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-8)

    # Compute Jv = df(z) @ u
    Jv = jvp(func, z, u, create_graph=create_graph)[1]

    # Compute norm of Jv and apply the isometric loss
    Jv_norm = Jv.norm(dim=1)
    loss = ((Jv_norm - 1.0) ** 2).mean()

    return loss

def isometry_jacobian_loss(func, z, create_graph=True):

    bs = len(z)

    jacobians = get_batch_jacobian(func, z, chunk_size=bs)
    jTjs = torch.einsum('bji,bjk->bik', jacobians, jacobians)
    eye = torch.eye(jTjs.shape[1], device=jTjs.device).repeat(jTjs.shape[0], 1, 1)
    jTj_minus_I = jTjs - eye
    loss = (jTj_minus_I**2).norm()
    return loss

def scaled_isometry_jacobian_loss(func, z, create_graph=True):
    
    bs = len(z)

    jacobians = get_batch_jacobian(func, z, chunk_size=bs)
    jTjs = torch.einsum('bji,bjk->bik', jacobians, jacobians)
    traces = vmap(torch.trace)(jTjs)
    lambda_factors = traces / z.shape[1]
    lambda_factors = torch.full_like(lambda_factors, lambda_factors.mean().item())
    eye = torch.eye(jTjs.shape[1], device=jTjs.device).repeat(jTjs.shape[0], 1, 1)
    jTj_minus_lambdaI = jTjs - lambda_factors.unsqueeze(1).unsqueeze(2).repeat(1, jTjs.shape[1], jTjs.shape[1]) * eye
    loss = (jTj_minus_lambdaI**2).norm()
    return loss

def conformality_jacobian_loss(func, z, create_graph=True):

    bs = len(z)
    m = z.shape[1]

    jacobians = get_batch_jacobian(func, z, chunk_size=bs)
    jTjs = torch.einsum('bji,bjk->bik', jacobians, jacobians)
    traces = vmap(torch.trace)(jTjs)
    lambda_factors = traces / m
    eye = torch.eye(jTjs.shape[1], device=jTjs.device).repeat(jTjs.shape[0], 1, 1)
    jTj_minus_lambdaI = jTjs - lambda_factors.unsqueeze(1).unsqueeze(2).repeat(1, jTjs.shape[1], jTjs.shape[1]) * eye
    loss = (jTj_minus_lambdaI**2).norm()
    return loss

def scaled_isometry_loss(func, z, create_graph=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)

    v = torch.randn(z.size()).to(z)
    Jv = jvp(
        func, z, v=v, create_graph=create_graph)[1]
    TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean()
    JTJv = (vjp(
        func, z, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
    TrG2 = torch.sum(JTJv**2, dim=1).mean()
    return TrG2/TrG**2


# Conformality loss functions
def conformality_trace_loss(func, z, num_samples=1, create_graph=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)
    m = z.shape[1]
    
    TrN_samples = []
    
    for _ in range(num_samples):
        v = torch.randn(z.size()).to(z)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-7) # this should be tested with unnormalized vectors
        
        Jv = jvp(func, z, v=v, create_graph=create_graph)[1]
        JTJv = vjp(func, z, v=Jv, create_graph=create_graph)[1]
        
        TrG = torch.sum(Jv.view(bs, -1)**2, dim=1)
        TrG2 = torch.sum(JTJv.view(bs, -1)**2, dim=1)
        TrN = (TrG2 - (1/m * (TrG**2)))
        
        TrN_samples.append(TrN)
    
    # Average over samples, then over batch
    TrN_mean = torch.stack(TrN_samples, dim=0).mean(dim=0)
    return (TrN_mean**2).mean() #does this also work for abs instead of square?

def conformality_trace2_loss(func, z, num_samples=1, create_graph=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)
    m = z.shape[1]
    
    TrN_samples = []
    
    for _ in range(num_samples):
        v = torch.randn(z.size()).to(z)
        # v = v / (v.norm(dim=1, keepdim=True) + 1e-7) # this should be tested with unnormalized vectors
        
        Jv = jvp(func, z, v=v, create_graph=create_graph)[1]
        JTJv = vjp(func, z, v=Jv, create_graph=create_graph)[1]
        
        TrG = torch.sum(Jv.view(bs, -1)**2, dim=1)
        TrG2 = torch.sum(JTJv.view(bs, -1)**2, dim=1)
        TrN = (TrG2 - (1/m * (TrG**2)))
        
        TrN_samples.append(TrN)
    
    # Average over samples, then over batch
    TrN_mean = torch.stack(TrN_samples, dim=0).mean(dim=0)
    return (TrN_mean**2).mean() #does this also work for abs instead of square?

def conformality_cosine_loss(f, z, create_graph=True):
    #sample batchsize pairs of orthogonal unit vectors
    bs = len(z)

    #
    # Could sample multiple pairs of vectors
    #
    
    u = torch.randn_like(z)
    v = torch.randn_like(z)
    cos_uv = torch.cosine_similarity(u,v, dim=1)

    Jv = jvp(f, z, v, create_graph=create_graph)[1].view(bs, -1)
    Ju = jvp(f, z, u, create_graph=create_graph)[1].view(bs, -1)
    cos_JuJv = torch.cosine_similarity(Ju, Jv, dim=1)

    angle_loss = torch.mean((cos_uv - cos_JuJv) ** 2)
    return angle_loss

##
def conformality_cosine_orthounit_loss(f, z, create_graph=True):
    #sample batchsize pairs of orthogonal unit vectors
    bs = len(z)
    
    u = torch.randn_like(z)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-8)

    def make_orthogonal(a):
        """
        a: Tensor of shape (batch_size, dim)
        Returns: Tensor of shape (batch_size, dim), each vector orthogonal to corresponding input
        """
        # Normalize input vectors (shape: [B, D])
        a_norm = a / a.norm(dim=1, keepdim=True)

        # Random vectors (same shape as input)
        b = torch.randn_like(a)

        # Dot product per batch (shape: [B])
        proj_coeff = torch.sum(a_norm * b, dim=1, keepdim=True)

        # Remove projection of b onto a => b_orth = b - proj_a(b)
        b_orth = b - proj_coeff * a_norm

        return b_orth
    
    v = make_orthogonal(u)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)


    Jv = jvp(f, z, v, create_graph=create_graph)[1]
    Ju = jvp(f, z, u, create_graph=create_graph)[1]

    # Compute the angle between the two vectors
    cos_angle = torch.cosine_similarity(Ju, Jv, dim=1)

    angle_loss = torch.mean((cos_angle) ** 2)
    return angle_loss


# Regularization functions for the decoder
def regularization(func, z, num_samples=1, create_graph=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)
    TrJTJ = get_JTJ_trace_estimate(func, z, chunk_size=bs, num_samples=num_samples, create_graph=create_graph)
    m = z.shape[1]

    return ((TrJTJ.mean() / m) -1)**2

##
def regularization5(func, z, goal_norm=40.0, create_graph=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)

    return (z.norm() - goal_norm)**2

def regularization4(f, z, create_graph=True):
    # sample pairs pairs of orthogonal unit vectors, compare cosine similarity and length
    bs = len(z)
    v = torch.randn_like(z)

    Jv = jvp(f, z, v, create_graph=create_graph)[1]

    v_len = v.norm(dim=1)
    Jv_len = Jv.norm(dim=1)

    return ((v_len/Jv_len).mean() -1)**2
