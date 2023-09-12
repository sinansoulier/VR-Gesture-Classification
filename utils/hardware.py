import torch

class Hardware:
    """
    This class contains methods to determine the hardware to use for the pipeline.
    """

    @staticmethod
    def device() -> str:
        """
        Returns the device to use for training.
        Returns:
            device: str - Device to use for training
        """
        # Define CPU as default device
        device = "cpu"

        # Use Cuda acceleration if available (Nvidia GPU)
        if torch.cuda.is_available():
            device = "cuda:0"
        # Use Metal acceleration if available (MacOS)
        elif torch.backends.mps.is_available():
            device = "mps:0"
    
        return device