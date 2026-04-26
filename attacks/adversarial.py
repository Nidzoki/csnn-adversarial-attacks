import torch
import torch.nn.functional as F

def get_model_prediction(model, data, is_snn=False):
    """Unified prediction getter for CNN and SNN."""
    output = model(data)
    if is_snn:
        # SNN returns [T, batch, output_dim], sum over time
        return output.sum(dim=0)
    return output

def fgsm_attack(image, epsilon, data_grad):
    """Fast Gradient Sign Method attack."""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def pgd_attack(model, device, data, target, epsilon, alpha=2/255, num_iter=10, is_snn=False):
    """Projected Gradient Descent attack."""
    data, target = data.to(device), target.to(device)
    perturbed_data = data.clone().detach()
    
    for _ in range(num_iter):
        perturbed_data.requires_grad = True
        output = get_model_prediction(model, perturbed_data, is_snn)
        loss = F.cross_entropy(output, target)
        
        model.zero_grad()
        loss.backward()
        
        adv_grad = perturbed_data.grad.data
        perturbed_data = perturbed_data + alpha * adv_grad.sign()
        
        # Project back to epsilon-ball and [0, 1] range
        eta = torch.clamp(perturbed_data - data, min=-epsilon, max=epsilon)
        perturbed_data = torch.clamp(data + eta, min=0, max=1).detach()
        
    return perturbed_data

def add_salt_and_pepper_noise(image, amount):
    """Add salt and pepper noise to images."""
    noisy_image = image.clone()
    # Salt
    salt = torch.rand_like(image) < (amount / 2)
    noisy_image[salt] = 1.0
    # Pepper
    pepper = torch.rand_like(image) < (amount / 2)
    noisy_image[pepper] = 0.0
    return noisy_image

def snn_fgsm_attack(model, device, data, target, epsilon):
    """FGSM adapted for SNN: accumulate gradients over T steps."""
    data, target = data.to(device), target.to(device)
    data.requires_grad = True
    
    output = get_model_prediction(model, data, is_snn=True)
    loss = F.cross_entropy(output, target)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = data.grad.data
    return fgsm_attack(data, epsilon, data_grad)
