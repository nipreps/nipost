import nibabel as nb
import numpy as np
import pytest


@pytest.fixture
def ramp_volume():
    """A 3D volume whose values are a linear ramp — easy to check interpolation."""
    data = np.arange(5 * 5 * 5, dtype='f4').reshape(5, 5, 5)
    return nb.Nifti1Image(data, np.eye(4))
