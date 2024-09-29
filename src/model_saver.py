import torch
import onnx
import onnxruntime

class ModelSaver:
    @staticmethod
    def save_pth(model, path):
        torch.save(model.state_dict(), path)

    @staticmethod
    def save_pt(model, path):
        torch.save(model, path)

    @staticmethod
    def save_onnx(model, path, input_shape):
        dummy_input = torch.randn(input_shape)
        torch.onnx.export(model, dummy_input, path, export_params=True, opset_version=11)

    @staticmethod
    def save_model(model, path, format, input_shape=None):
        if format == 'pth':
            ModelSaver.save_pth(model, path)
        elif format == 'pt':
            ModelSaver.save_pt(model, path)
        elif format == 'onnx':
            if input_shape is None:
                raise ValueError("input_shape must be provided for ONNX format")
            ModelSaver.save_onnx(model, path, input_shape)
        else:
            raise ValueError(f"Unsupported format: {format}")