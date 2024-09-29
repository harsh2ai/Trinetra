import torch
from torch.cuda.amp import autocast, GradScaler

class PrecisionFormat:
    def __init__(self, precision):
        self.precision = precision
        self.use_amp = precision == "FP16"
        self.scaler = GradScaler() if self.use_amp else None

    def cast_model(self, model):
        if self.precision == "FP64":
            model = model.double()
        return model

    def cast_inputs(self, inputs):
        if self.precision == "FP64":
            inputs = inputs.double()
        return inputs

    def train_step(self, model, inputs, labels, criterion, optimizer):
        optimizer.zero_grad()
        
        if self.use_amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        return loss, outputs

def get_precision_format(precision):
    supported_formats = ["FP16", "FP32", "FP64"]
    if precision not in supported_formats:
        raise ValueError(f"Unsupported precision format: {precision}")
    return PrecisionFormat(precision)