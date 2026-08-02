
from models import model_STRMSR

def create_model(opts):
    if opts.model_type == 'STRMSR':
        model = model_STRMSR.RecurrentModel(opts)
    else:
        raise NotImplementedError
    return model
