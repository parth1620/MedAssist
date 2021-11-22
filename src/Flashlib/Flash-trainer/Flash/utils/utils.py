import torch 
from tqdm import tqdm


'''
utils 
- update_bn : update batch normalization for SWA model 
- replace_activation : replace default activation to new activation 
- convert_norm_layers : replace BatchNorm to GroupNorm
'''



@torch.no_grad()
def update_bn(loader, model, device=None):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in tqdm(loader):
        for k,v in input.items():
            input[k] = v.to(device)
        
        model(**input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)



def replace_activations(model, existing_layer, new_layer):
    print("Replace Activation initiated...")
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_activations(module, existing_layer, new_layer)

        if type(module) == existing_layer:
            layer_old = module
            layer_new = new_layer
            model._modules[name] = layer_new
    return model


def convert_norm_layers(model, old_layer_type, new_layer_type, convert_weights=False, num_groups=None):

    print("Replacing BatchNorm to GroupNorm....")

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_norm_layers(module, old_layer_type, new_layer_type, convert_weights, num_groups=num_groups)

        if type(module) == old_layer_type:
            old_layer = module
            new_layer = new_layer_type(module.num_features if num_groups is None else num_groups, module.num_features, module.eps, module.affine) 

            if convert_weights:
                new_layer.weight = old_layer.weight
                new_layer.bias = old_layer.bias

            model._modules[name] = new_layer

    return model