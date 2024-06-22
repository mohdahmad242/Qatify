from typing import List, Tuple
from torch.nn import Module
from mix_precision_main.hessian_per_layer import hessian_per_layer
# from mix_precision_main.hessian_per_layer import hessian_per_layer
from mix_precision_main.logger import logger
import torch
from mix_precision_main.mix_precision import mixprecision_profiling, mixprecision_bit_selection


class Qatify:
    def __init__(self, model, quantized_model, bit_list, model_size_constraints):
        """
        Initializes the Qatify class with the provided parameters.

        Parameters:
        model (torch model): FP32 Model.
        quantized_model (torch model): Quantize model (fake quant model).
        bit_list (list): List of bits, example [2, 4, 8, 16].
        model_size_constraints (float): Model size constraints in MB.
        """
        self.model = model
        self.quantized_model = quantized_model
        self.bitwidth_list = bit_list
        self.model_size_constraints = model_size_constraints
        self.traced_sensetivity_dict = None
        self.layer_parameters_dict = None
    
    def model_size_analysis(self, model):
        '''
        Calculate the given fp32 model size and layer parameters
        
        Parameters:
        model (torch model): FP32 Model.
        '''
        layer_parameters_dict = {}
        for name, mod in model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                layer_parameters_dict[name] = mod.weight.numel()

        self.layer_parameters_dict = layer_parameters_dict

        model_size = sum(list(layer_parameters_dict.values())) * 32 / 8 / 1024 / 1024
        logger.info("FP model size: {:.2f} MB".format(model_size))
    
    def get_targets(self, model, inputs):
        '''
        Generate targets required for mixprecision profiling 

        Prameters:
        model (torch model): FP32 Model.
        '''
        with torch.no_grad():
            targets = model(inputs)
        return targets

    def get_trace_sensetive_dict(self):
        inputs = torch.rand(2, 3, 224, 224)
        targets = self.get_targets(self.model, inputs)

        self.traced_sensetivity_dict = mixprecision_profiling(self.model.cuda(), self.quantized_model, self.bitwidth_list,
                                                data=(inputs, targets), criterion=torch.nn.CrossEntropyLoss(), algo='hawq_trace')
        
    def get_selected_mixprecision_bit(self, traced_sensetivity_dict):

        mixprecision_bit_selection(self.bitwidth_list, 
                            self.traced_sensetivity_dict,
                            self.layer_parameters_dict,
                            model_size_constraints=self.model_size_constraints, latency_constraints=None)
        
    def run(self):
        # Run model analysis
        self.model_size_analysis(self.model)

        # Get traced sensetivity analysis
        self.get_trace_sensetive_dict()

        # Run mixprecision bit selection process
        self.get_selected_mixprecision_bit(self.traced_sensetivity_dict)

            
