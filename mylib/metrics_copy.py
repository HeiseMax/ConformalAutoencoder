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

def gmean(input_x, dim):
    log_x = torch.log(torch.clamp(input_x, min=torch.finfo(torch.float32).eps))
    return torch.exp(torch.mean(log_x, dim=dim))

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

def get_JTJs(jacobians):
    return torch.einsum('bji,bjk->bik', jacobians, jacobians)

def get_batch_traces(jTjs):
    return vmap(torch.trace)(jTjs)

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

def get_determinants(jTjs):
    jTjs = jTjs.double()
    dets = torch.sqrt(torch.linalg.det(jTjs))
    
    del jTjs
    torch.cuda.empty_cache()

    return dets

def get_log_determinants(jTjs):
    jTjs = jTjs.double()
    log_dets = 0.5 * torch.logdet(jTjs)
    
    del jTjs
    torch.cuda.empty_cache()

    return log_dets

def get_determinant_estimates(conformal_factors, latent_dim):
    conformal_factors = conformal_factors.double()
    det_estimates = torch.sqrt(conformal_factors**latent_dim)

    del conformal_factors
    torch.cuda.empty_cache()

    return det_estimates

def get_log_determinant_estimates(conformal_factors, latent_dim):
    conformal_factors = conformal_factors.double()
    log_det_estimates = 0.5 * latent_dim * torch.log(conformal_factors)

    del conformal_factors
    torch.cuda.empty_cache()

    return log_det_estimates

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


# Isometry loss functions

def isometry_loss(func, z, create_graph=True):

    bs = len(z)

    jacobians = get_batch_jacobian(func, z, chunk_size=bs)
    jTjs = torch.einsum('bji,bjk->bik', jacobians, jacobians)
    eye = torch.eye(jTjs.shape[1], device=jTjs.device).repeat(jTjs.shape[0], 1, 1)
    jTj_minus_I = jTjs - eye
    return (jTj_minus_I**2).mean()

def isometry_trace_loss(func, z, create_graph=True):

    bs = len(z)
    m = z.shape[1]

    v = torch.randn(z.size()).to(z)
    Jv = jvp(
        func, z, v=v, create_graph=create_graph)[1]
    JTJv = (vjp(
        func, z, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
    TrG2 = torch.sum(JTJv**2, dim=1).mean()
    return (((TrG2 / m) -1)**2)

def isometry_angle_loss(f, z, create_graph=True):
    #sample batchsize pairs of orthogonal unit vectors
    bs = len(z)

    #
    # Could sample multiple pairs of vectors
    #
    
    u = torch.randn_like(z)
    # v = make_orthogonal(u)
    v = torch.randn_like(z)
    dot_uv = torch.sum(u * v, dim=1)

    Jv = jvp(f, z, v, create_graph=create_graph)[1].view(bs, -1)
    Ju = jvp(f, z, u, create_graph=create_graph)[1].view(bs, -1)
    dot_JuJv = torch.sum(Ju * Jv, dim=1)

    angle_loss = torch.mean((dot_uv - dot_JuJv) ** 2)
    return angle_loss


# Scaled isometry loss functions

def scaled_isometry_loss(func, z, create_graph=True):
    
    bs = len(z)

    jacobians = get_batch_jacobian(func, z, chunk_size=bs)
    jTjs = torch.einsum('bji,bjk->bik', jacobians, jacobians)
    traces = vmap(torch.trace)(jTjs)
    lambda_factors = traces / z.shape[1]
    lambda_factors = torch.full_like(lambda_factors, lambda_factors.mean().item())
    eye = torch.eye(jTjs.shape[1], device=jTjs.device).repeat(jTjs.shape[0], 1, 1)
    jTj_minus_lambdaI = jTjs - lambda_factors.unsqueeze(1).unsqueeze(2).repeat(1, jTjs.shape[1], jTjs.shape[1]) * eye
    return (jTj_minus_lambdaI**2).mean()

def scaled_isometry_trace_loss(func, z, create_graph=True):
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
    return (((TrG2/(TrG**2)) -1)**2)

def scaled_isometry_angle_loss(f, z, create_graph=True):
    #sample batchsize pairs of orthogonal unit vectors
    bs = len(z)
    
    # either 4 vectors
    u1 = torch.randn_like(z)
    u2 = torch.randn_like(z)
    v1 = torch.randn_like(z)
    v2 = torch.randn_like(z)
    cos_u1v1 = torch.cosine_similarity(u1,v1, dim=1)
    cos_u2v2 = torch.cosine_similarity(u2,v2, dim=1)

    Ju1 = jvp(f, z, u1, create_graph=create_graph)[1].view(bs, -1)
    Ju2 = jvp(f, z, u2, create_graph=create_graph)[1].view(bs, -1)
    Jv1 = jvp(f, z, v1, create_graph=create_graph)[1].view(bs, -1)
    Jv2 = jvp(f, z, v2, create_graph=create_graph)[1].view(bs, -1)
    
    cos_Ju1Jv1 = torch.cosine_similarity(Ju1, Jv1, dim=1)
    cos_Ju2Jv2 = torch.cosine_similarity(Ju2, Jv2, dim=1)

    angle_loss = torch.mean(((cos_u1v1 / cos_u2v2) - (cos_Ju1Jv1 / cos_Ju2Jv2)) ** 2)
    return angle_loss


# Conformality loss functions

def conformality_loss(func, z, create_graph=True):

    bs = len(z)
    m = z.shape[1]

    jacobians = get_batch_jacobian(func, z, chunk_size=bs)
    jTjs = torch.einsum('bji,bjk->bik', jacobians, jacobians)
    traces = vmap(torch.trace)(jTjs)
    lambda_factors = traces / m
    eye = torch.eye(jTjs.shape[1], device=jTjs.device).repeat(jTjs.shape[0], 1, 1)
    jTj_minus_lambdaI = jTjs - lambda_factors.unsqueeze(1).unsqueeze(2).repeat(1, jTjs.shape[1], jTjs.shape[1]) * eye
    loss = (jTj_minus_lambdaI**2).mean()
    return loss

def conformality_trace_loss(func, z, num_samples=1, regularize=False, create_graph=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)
    m = z.shape[1]
    
    TrN_samples = []
    reg_loss = 0.0
    
    for _ in range(num_samples):
        v = torch.randn(z.size()).to(z)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-7)
        
        Jv = jvp(func, z, v=v, create_graph=create_graph)[1]
        JTJv = vjp(func, z, v=Jv, create_graph=create_graph)[1]
        
        TrG = torch.sum(Jv.view(bs, -1)**2, dim=1)
        TrG2 = torch.sum(JTJv.view(bs, -1)**2, dim=1)
        TrN = (TrG2 - (1/m * (TrG**2)))
        
        TrN_samples.append(TrN)

        if regularize:
            # Regularization term to keep Jv norm close to 1
            m = z.shape[1]
            reg_loss += (gmean(TrG / m, dim=0) -1)**2
    
    # Average over samples, then over batch
    TrN_mean = torch.stack(TrN_samples, dim=0).mean(dim=0)
    conf_loss = (TrN_mean**2).mean()
    if regularize:
        return conf_loss, reg_loss / num_samples
    else:
        return conf_loss

def conformality_angle_loss(f, z, regularize=False, create_graph=True):
    #sample batchsize pairs of orthogonal unit vectors
    bs = len(z)

    #
    # Could sample multiple pairs of vectors
    #
    
    u = torch.randn_like(z)
    # v = make_orthogonal(u)
    v = torch.randn_like(z)
    cos_uv = torch.cosine_similarity(u,v, dim=1)

    Jv = jvp(f, z, v, create_graph=create_graph)[1].view(bs, -1)
    Ju = jvp(f, z, u, create_graph=create_graph)[1].view(bs, -1)
    cos_JuJv = torch.cosine_similarity(Ju, Jv, dim=1)

    angle_loss = torch.mean((cos_uv - cos_JuJv) ** 2)
    if regularize:
        TrG = torch.sum(Ju**2, dim=1)
        reg_loss = (gmean(TrG / z.shape[1], dim=0) -1)**2
        return angle_loss, reg_loss
    return angle_loss


# Regularization functions for the decoder

def regularize_conformal_factor(func, z, factor=1, num_samples=1, create_graph=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)
    TrJTJ = get_JTJ_trace_estimate(func, z, chunk_size=bs, num_samples=num_samples, create_graph=create_graph)
    m = z.shape[1]

    return (gmean(TrJTJ / m, dim=0) -factor)**2

def regularize_latent_norm(func, z, goal_norm=40.0, create_graph=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)

    return (z.norm() - goal_norm)**2

def regularize_vector_scaling(f, z, create_graph=True):
    # sample pairs pairs of orthogonal unit vectors, compare cosine similarity and length
    bs = len(z)
    v = torch.randn_like(z)

    Jv = jvp(f, z, v, create_graph=create_graph)[1]

    v_len = v.norm(dim=1)
    Jv_len = Jv.norm(dim=1)

    return ((v_len/Jv_len).mean() -1)**2


# Evaluation function for conformality

def evaluate_conformality(model, data, double_precision=False, chunk_size=64):
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
        jacobians = get_batch_jacobian(model.decoder, latent, chunk_size=chunk_size)
        jTjs = get_JTJs(jacobians)
        traces = get_batch_traces(jTjs)
        conformal_factors = traces / latent.shape[1]

        # Estimate the trace of J^T J
        trace_estimate = get_JTJ_trace_estimate(model.decoder, latent, chunk_size=chunk_size)
        conformal_factors_estimate = trace_estimate / latent.shape[1]

        # Errors in conformal factors estimate
        conformal_factors_meanerror = torch.mean(torch.abs(conformal_factors - conformal_factors_estimate))
        conformal_factors_meanerror_normalized = conformal_factors_meanerror / (conformal_factors.mean() + torch.finfo(torch.float32).eps)

        # gini for diagonal elements
        diagonals = jTjs.diagonal(dim1=1, dim2=2)
        gini_value = gini(diagonals)

        # off diag mean, norm
        off_diagonal = jTjs - torch.diag_embed(diagonals)
        off_diag_mean = off_diagonal.abs().mean(dim=(1, 2))
        off_diag_norm = off_diagonal.norm(dim=(1, 2))
        off_diag_mean_normalized = off_diag_mean / conformal_factors
        off_diag_norm_normalized = off_diag_norm / conformal_factors

        # jTj - lambda*I mean, norm
        eye = torch.eye(jTjs.shape[1], device=jTjs.device).repeat(jTjs.shape[0], 1, 1)
        jTj_minus_cI = jTjs - conformal_factors.unsqueeze(1).unsqueeze(2).repeat(1, jTjs.shape[1], jTjs.shape[1]) * eye

        conformal_mean = jTj_minus_cI.abs().mean(dim=(1, 2))
        conformal_norm = jTj_minus_cI.norm(dim=(1, 2))
        conformal_mean_normalized = conformal_mean / conformal_factors
        conformal_norm_normalized = conformal_norm / conformal_factors

        # determinant of jacobian vs sqrt lambda**m
        jacobian_determinants = get_determinants(jTjs)
        estimate_determinants = get_determinant_estimates(conformal_factors, latent_dim)
        estimate_estimate_determinants = get_determinant_estimates(conformal_factors_estimate, latent_dim)
        
        determinant_estimate_errors = torch.abs(jacobian_determinants - estimate_determinants)
        deterimnant_estimate_errors_normalized = determinant_estimate_errors / (torch.abs(jacobian_determinants) + torch.finfo(torch.float32).eps)
        determinant_estimate_estimate_errors = torch.abs(jacobian_determinants - estimate_estimate_determinants)
        determinant_estimate_estimate_errors_normalized = determinant_estimate_estimate_errors / (torch.abs(jacobian_determinants) + torch.finfo(torch.float32).eps)

        # log determinant of jacobian vs 0.5 * latent_dim * log(lambda)
        jacobian_log_determinants = get_log_determinants(jTjs)
        estimate_log_determinants = get_log_determinant_estimates(conformal_factors, latent_dim)
        estimate_estimate_log_determinants = get_log_determinant_estimates(conformal_factors_estimate, latent_dim)

        log_determinant_estimate_errors = torch.abs(jacobian_log_determinants - estimate_log_determinants)
        log_determinant_estimate_estimate_errors = torch.abs(jacobian_log_determinants - estimate_estimate_log_determinants)

        # std of latent space
        latent_std = latent.std(dim=0)
        
        return {
            'Reconstruction Error': torch.nn.MSELoss()(data, model.decode(latent)).item(),
            'Diagonal Uniformity (gini)': gini_value.item(),

            'Conformal Factor Mean': conformal_factors.mean().item(),
            'Conformal Factor Std': conformal_factors.std().item(),
            'Conformal Factor Std (normalized)': (conformal_factors.std() /conformal_factors.mean()).item(),
            'Conformal Factor Estimate Error': conformal_factors_meanerror.item(),
            'Conformal Factor Estimate Error (normalized)': conformal_factors_meanerror_normalized.item(),

            'Off-diagonal Mean': off_diag_mean.mean().item(),
            'Off-diagonal Mean (normalized)': off_diag_mean_normalized.mean().item(),
            'Off-diagonal Norm': off_diag_norm.mean().item(),
            'Off-diagonal Norm (normalized)': off_diag_norm_normalized.mean().item(),

            'Conformality Mean': conformal_mean.mean().item(),
            'Conformality Mean (normalized)': conformal_mean_normalized.mean().item(),
            'Conformality Norm': conformal_norm.mean().item(),
            'Conformality Norm (normalized)': conformal_norm_normalized.mean().item(),

            'Determinant Estimation Error': determinant_estimate_errors.mean().item(),
            'Determinant Estimation Error (normalized)': deterimnant_estimate_errors_normalized.mean().item(),
            'Determinant Estimation from Estimate Error': determinant_estimate_estimate_errors.mean().item(),
            'Determinant Estimation from Estimate Error (normalized)': determinant_estimate_estimate_errors_normalized.mean().item(),

            'Log Determinant Estimation Error': log_determinant_estimate_errors.mean().item(),
            'Log Determinant Estimation from Estimate Error': log_determinant_estimate_estimate_errors.mean().item(),
            # 'determinant_vs_estimate_std': determinant_vs_estimate.std().item(),
            # 'log_determinant_vs_estimate_std': log_determinant_vs_estimate.std().item(),
            
            'Latent Std': latent_std.mean().item(),
            'Latent Std Max': latent_std.max().item(),
            'Latent Std Min': latent_std.min().item(),
            'Latent Norm': latent.norm().item(),
        }
