"""
Dataset class implementation.
"""

from ..core import Tensor

class Dataset:
    """
    Abstract dataset class.
    
    Provides an interface for accessing data samples.
    """
    
    def __init__(self):
        """
        Initialize a dataset.
        """
        pass
        
    def __getitem__(self, idx):
        """
        Get a data sample.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            A data sample
        """
        raise NotImplementedError
        
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            Dataset size
        """
        raise NotImplementedError


class TensorDataset(Dataset):
    """
    Dataset wrapping tensors.
    
    Each sample will be retrieved by indexing tensors along the first dimension.
    """
    
    def __init__(self, *tensors):
        """
        Initialize a TensorDataset.
        
        Args:
            *tensors: Tensors that have the same size of the first dimension
        """
        super().__init__()
        # TODO: Check that all tensors have the same size in the first dimension
        if not all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors):
            raise ValueError("All tensors must have the same size in the first dimension")
        self.tensors = tensors
        
    def __getitem__(self, idx):
        """
        Get a data sample.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple of tensor samples
        """
        # TODO: Return a tuple of tensor samples at the given index
        return tuple(tensor[idx] for tensor in self.tensors)
        
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            Dataset size
        """
        # TODO: Return the size of the dataset 
        return self.tensors[0].shape[0]