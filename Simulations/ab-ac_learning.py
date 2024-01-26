import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from Models.model import Categorizer, Trainer

# ---- Generate Data ----
# Create pairings (a,b) and (a,c)
# where b=0 for 0 <= a < B_BOUNDARY and b=1 for B_BOUNDARY <= a < RANGE
# where c=0 for 0 <= a < C_BOUNDARY and c=1 for C_BOUNDARY <= a < RANGE
RANGE = 20
# It doesn't make sense to have a test size if you are teaching arbitrary associations
a_data = np.arange(RANGE)
b_data = np.random.permutation(a_data)
c_data = np.random.permutation(b_data)
b_condition = np.zeros(RANGE)
c_condition = np.ones(RANGE)

# bX_data = np.stack([a_data, b_condition], axis=-1)
# cX_data = np.stack([a_data, c_condition], axis=-1)
# bx_train, bx_test, by_train, by_test = train_test_split(bX_data, b_data, test_size=0.2, random_state=42)
# cx_train, cx_test, cy_train, cy_test = train_test_split(cX_data, c_data, test_size=0.2, random_state=42)
b_experiment = np.stack([a_data, b_condition, b_data], axis=-1)
c_experiment = np.stack([a_data, c_condition, c_data], axis=-1)
# data = np.stack([a_data, b_data, c_data], axis=-1)
# df = pd.DataFrame(data=data, columns=['A', 'B', 'C'])

# ---- Initialize Model and Metrics ----
model = Categorizer()
trainer = Trainer(model)
EPOCHS = 500

# ---- Train Model ----
# -- ab training loop --
b_losses, _ = trainer.train(b_experiment, batch_size=10, n_features=2, test=b_experiment, num_epochs=EPOCHS)

# -- ac training loop --
c_losses, second_test = trainer.train(c_experiment, batch_size=10, n_features=2,
                                      test=b_experiment, num_epochs=EPOCHS)

# ---- Present Results ----
epochs = np.arange(len(c_losses))
plt.figure(figsize=(10, 5))
plt.plot(epochs, c_losses, label='C Test Loss', color='black')
plt.plot(epochs, second_test[0], label='B 2nd Test Loss', color='orange')
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# print(f'Input: {data.numpy()}, Predicted Output: {predicted_output.item()}')
