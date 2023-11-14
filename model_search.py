from pathlib import Path
import typing as T
import numpy as np
import torch
from torch.utils.data import DataLoader
from colorizers.generator import ModelConfig, generate_model
from train import build_optimizer, build_criterion, train, TrainingLogger, get_dataloader
from utils import get_device, get_root_dir

class MSPipeline:
    def __init__(
        self, output_dir: str, models_config: T.List[ModelConfig],
        trainloader: DataLoader, testloader: DataLoader
    ):
        self.output_dir = Path(output_dir)
        self.models_config = models_config
        self.trainloader = trainloader
        self.testloader = testloader

    def train(self, n_epochs: int = 1, device: str = 'cpu') -> None:
        """
        Train all models in the pipeline.
        """
        best_eval_loss = np.inf
        best_model = None
        best_model_config = None
        for model_config in self.models_config:
            model = generate_model(model_config)
            model.to(device)
            optimizer = build_optimizer('Adam', {'params': model.parameters(), 'lr': 0.001})
            criterion = build_criterion('CrossEntropyLoss', {})
            logger = TrainingLogger()
            eval_loss, model_params = train(
                model, optimizer, self.trainloader, self.testloader, device, criterion, n_epochs, logger
            )

            # Save training plot
            logger.save_plot(self.output_dir, model_config.name)
            
            # Save model
            torch.save(model_params, self.output_dir / f'{model_config.name}.pth')

            # Save model config
            model_config.dump(self.output_dir / f'{model_config.name}.json')

            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model = model_params
                best_model_config = model_config
        
        # Save best model
        torch.save(best_model, self.output_dir / 'best_model.pth')

        # Save best model config
        best_model_config.dump(self.output_dir / 'best_model.json')


if __name__ == '__main__':
    train_data_path = ""
    test_data_path = ""
    trainloader = get_dataloader(train_data_path)
    testloader = get_dataloader(test_data_path)

    # Define models
    models_config = [
        ModelConfig('model1', dropout=[]),
    ]

    # Train
    output_dir = get_root_dir() / 'output'
    pipeline = MSPipeline(output_dir, models_config, trainloader, testloader)
    device = get_device()
    pipeline.train(n_epochs=1, device=device)
