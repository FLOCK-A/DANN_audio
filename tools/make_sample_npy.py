import os
import numpy as np

root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(root, exist_ok=True)

filenames = [f'audio_0000{i}.npy' for i in range(1,6)]
for name in filenames:
    path = os.path.join(root, name)
    # create a small random log-mel like array: (time_steps, n_mels)
    arr = np.random.randn(100, 64).astype(np.float32)
    np.save(path, arr)
    print('wrote', path)
print('done')

