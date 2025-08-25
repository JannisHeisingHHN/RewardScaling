from typing import Any

def alter_settings(settings: dict[str, Any]) -> dict[str, Any]:
    policy = settings['policy']
    architecture = [eval(layer) for layer in policy['architecture']]

    policy_args = {
        'architecture': architecture,
        'out_size': policy['out_size'],
        'lower_bounds': policy['lower_bounds'],
        'upper_bounds': policy['upper_bounds'],
        'device': settings['setup']['device'],
    }

    settings['parameters']['model_parameters']['policy_class'] = policy['policy_class']
    settings['parameters']['model_parameters']['policy_args'] = policy_args

    return settings
