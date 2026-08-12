import inspect

class Configurable:
    @classmethod
    def from_config(cls, config: dict):
        """
        Factory method: inspects the child class's __init__,
        extracts relevant keys from 'config', and instantiates.
        """
        # Get the signature of the actual class (e.g., Hamiltonian)
        signature = inspect.signature(cls.__init__)

        # Filter the big config dict
        valid_params = {}
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue

            if param_name in config:
                valid_params[param_name] = config[param_name]

        # Instantiate. Python will raise a TypeError here if
        # required arguments are missing, preserving strictness.
        return cls(**valid_params)
