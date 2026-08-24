# Project 4: Comparative Study of Optimization and Regularization Techniques
# Dataset: Fashion-MNIST
#
# Expected Kaggle files in the working folder:
#   fashion-mnist_train.csv
#   fashion-mnist_test.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping


# ============================================================
# Step 1: Load and Prepare the Fashion-MNIST Dataset
# ============================================================

train_path = "fashion-mnist_train.csv"
test_path = "fashion-mnist_test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("\n===== Dataset Size =====")
print("Training samples:", len(train_df))
print("Testing samples :", len(test_df))

# Separate pixel values from labels
X_train = train_df.iloc[:, 1:].values / 255.0
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values / 255.0
y_test = test_df.iloc[:, 0].values


# ============================================================
# Step 2: Display One Image from Each Fashion-MNIST Class
# ============================================================

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
shown = set()

for i in range(len(y_train)):
    label = y_train[i]

    if label not in shown:
        image = X_train[i].reshape(28, 28)
        ax = axes[label // 5, label % 5]

        ax.imshow(image, cmap="gray")
        ax.set_title(class_names[label])
        ax.axis("off")

        shown.add(label)

    if len(shown) == 10:
        break

plt.tight_layout()
plt.show()


# ============================================================
# Step 3: Class Distribution
# ============================================================

classes, counts = np.unique(y_train, return_counts=True)

plt.figure(figsize=(10, 5))
plt.bar(class_names, counts)
plt.xlabel("Class Name")
plt.ylabel("Number of Samples")
plt.title("Class Distribution in Fashion-MNIST")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ============================================================
# Step 4: Enable Eager Execution
# ============================================================

tf.config.run_functions_eagerly(True)


# ============================================================
# Step 5: Common MLP Architecture
# ============================================================

def create_mlp():
    return Sequential([
        Dense(128, activation="relu", input_shape=(784,)),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])


# ============================================================
# Step 6: Training-Loss Callback
# ============================================================

class PrintLossPerEpoch(Callback):

    def __init__(self, label):
        super().__init__()
        self.label = label

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")

        if loss is not None:
            print(
                f"[{self.label}] "
                f"Epoch {epoch + 1}: Loss = {loss:.4f}"
            )


# ============================================================
# Step 7: Train a Model with a Selected Optimizer
# ============================================================

def train_optimizer(optimizer, label):
    model = create_mlp()

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=3,
        batch_size=128,
        validation_split=0.2,
        verbose=0,
        callbacks=[PrintLossPerEpoch(label)]
    )

    return model, history


# ============================================================
# Step 8: SGD Optimizer
# ============================================================

sgd_model, history_sgd = train_optimizer(
    SGD(learning_rate=0.01),
    "SGD"
)


# Plot SGD loss
loss_values = history_sgd.history["loss"]
epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_values, marker="o")
plt.xlabel("Epoch Number")
plt.ylabel("Training Loss")
plt.title("SGD Optimizer: Epoch vs Loss")
plt.xticks(list(epochs))
plt.grid(True)
plt.show()


# ============================================================
# Step 9: Regularization Models
# ============================================================

def create_l1():
    return Sequential([
        Dense(
            128,
            activation="relu",
            input_shape=(784,),
            kernel_regularizer=l1(0.001)
        ),
        Dense(
            64,
            activation="relu",
            kernel_regularizer=l1(0.001)
        ),
        Dense(10, activation="softmax")
    ])


def create_l2():
    return Sequential([
        Dense(
            128,
            activation="relu",
            input_shape=(784,),
            kernel_regularizer=l2(0.001)
        ),
        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(0.001)
        ),
        Dense(10, activation="softmax")
    ])


def create_dropout():
    return Sequential([
        Dense(128, activation="relu", input_shape=(784,)),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])


def create_batchnorm():
    return Sequential([
        Dense(128, input_shape=(784,)),
        BatchNormalization(),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(10, activation="softmax")
    ])


def train_regularization(model_fn, label, use_early_stop=False):
    model = model_fn()

    optimizer = SGD(learning_rate=0.01)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [PrintLossPerEpoch(label)]

    if use_early_stop:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )
        )

    history = model.fit(
        X_train,
        y_train,
        epochs=3,
        batch_size=128,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks
    )

    return model, history


# Train all regularization techniques
l1_model, history_l1 = train_regularization(
    create_l1,
    "L1 Regularisation"
)

l2_model, history_l2 = train_regularization(
    create_l2,
    "L2 Regularisation"
)

dropout_model, history_dropout = train_regularization(
    create_dropout,
    "Dropout"
)

early_model, history_early = train_regularization(
    create_l2,
    "Early Stopping",
    use_early_stop=True
)

batchnorm_model, history_batchnorm = train_regularization(
    create_batchnorm,
    "Batch Normalisation"
)


# ============================================================
# Step 10: Compare Regularization Techniques
# ============================================================

loss_l1 = history_l1.history["loss"]
loss_l2 = history_l2.history["loss"]
loss_dropout = history_dropout.history["loss"]
loss_early = history_early.history["loss"]
loss_batchnorm = history_batchnorm.history["loss"]

max_epochs = max(
    len(loss_l1),
    len(loss_l2),
    len(loss_dropout),
    len(loss_early),
    len(loss_batchnorm)
)


def pad_with_stopped(loss_list, max_len):
    return loss_list + ["STOPPED"] * (max_len - len(loss_list))


loss_table_regularization = pd.DataFrame({
    "Epoch": range(1, max_epochs + 1),
    "L1": pad_with_stopped(loss_l1, max_epochs),
    "L2": pad_with_stopped(loss_l2, max_epochs),
    "Dropout": pad_with_stopped(loss_dropout, max_epochs),
    "EarlyStopping": pad_with_stopped(loss_early, max_epochs),
    "BatchNorm": pad_with_stopped(loss_batchnorm, max_epochs)
})

print("\n===== Regularization Loss Comparison =====")
print(loss_table_regularization)


plt.figure(figsize=(10, 6))


def plot_regularization_loss(epoch, loss, label):
    values = [
        np.nan if value == "STOPPED" else value
        for value in loss
    ]
    plt.plot(epoch, values, marker="o", label=label)


epochs_reg = loss_table_regularization["Epoch"]

plot_regularization_loss(
    epochs_reg,
    loss_table_regularization["L1"],
    "L1 Regularisation"
)

plot_regularization_loss(
    epochs_reg,
    loss_table_regularization["L2"],
    "L2 Regularisation"
)

plot_regularization_loss(
    epochs_reg,
    loss_table_regularization["Dropout"],
    "Dropout"
)

plot_regularization_loss(
    epochs_reg,
    loss_table_regularization["EarlyStopping"],
    "Early Stopping"
)

plot_regularization_loss(
    epochs_reg,
    loss_table_regularization["BatchNorm"],
    "Batch Normalisation"
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Epoch vs Loss (Regularisation Techniques Comparison)")
plt.xticks(list(epochs_reg))
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# Step 11: AdaGrad Optimizer
# ============================================================

adagrad_model, history_adagrad = train_optimizer(
    Adagrad(learning_rate=0.01),
    "AdaGrad"
)

loss_values = history_adagrad.history["loss"]
epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_values, marker="o")
plt.xlabel("Epoch Number")
plt.ylabel("Training Loss")
plt.title("AdaGrad Optimizer: Epoch vs Loss")
plt.xticks(list(epochs))
plt.grid(True)
plt.show()


# ============================================================
# Step 12: AdaDelta Optimizer
# ============================================================

adadelta_model, history_adadelta = train_optimizer(
    Adadelta(),
    "AdaDelta"
)

loss_values = history_adadelta.history["loss"]
epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_values, marker="o")
plt.xlabel("Epoch Number")
plt.ylabel("Training Loss")
plt.title("AdaDelta Optimizer: Epoch vs Loss")
plt.xticks(list(epochs))
plt.grid(True)
plt.show()


# ============================================================
# Step 13: RMSprop Optimizer
# ============================================================

rmsprop_model, history_rmsprop = train_optimizer(
    RMSprop(learning_rate=0.01),
    "RMSprop"
)

loss_values = history_rmsprop.history["loss"]
epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_values, marker="o")
plt.xlabel("Epoch Number")
plt.ylabel("Training Loss")
plt.title("RMSprop Optimizer: Epoch vs Loss")
plt.xticks(list(epochs))
plt.grid(True)
plt.show()


# ============================================================
# Step 14: Nesterov Accelerated Gradient (NAG)
# ============================================================

nag_model, history_nag = train_optimizer(
    SGD(
        learning_rate=0.01,
        momentum=0.9,
        nesterov=True
    ),
    "NAG"
)

loss_values = history_nag.history["loss"]
epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_values, marker="o")
plt.xlabel("Epoch Number")
plt.ylabel("Training Loss")
plt.title("NAG Optimizer: Epoch vs Loss")
plt.xticks(list(epochs))
plt.grid(True)
plt.show()


# ============================================================
# Step 15: Adam Optimizer
# ============================================================

adam_model, history_adam = train_optimizer(
    Adam(learning_rate=0.01),
    "Adam"
)

loss_values = history_adam.history["loss"]
epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_values, marker="o")
plt.xlabel("Epoch Number")
plt.ylabel("Training Loss")
plt.title("Adam Optimizer: Epoch vs Loss")
plt.xticks(list(epochs))
plt.grid(True)
plt.show()


# ============================================================
# Step 16: Compare All Optimization Techniques
# ============================================================

loss_dict = {
    "SGD": history_sgd.history["loss"],
    "AdaGrad": history_adagrad.history["loss"],
    "AdaDelta": history_adadelta.history["loss"],
    "RMSprop": history_rmsprop.history["loss"],
    "NAG": history_nag.history["loss"],
    "Adam": history_adam.history["loss"]
}

max_epochs = max(
    len(values)
    for values in loss_dict.values()
)


def pad_loss(loss_list, max_len):
    return loss_list + ["STOPPED"] * (max_len - len(loss_list))


loss_table_optimizers = pd.DataFrame({
    "Epoch": range(1, max_epochs + 1),
    **{
        name: pad_loss(values, max_epochs)
        for name, values in loss_dict.items()
    }
})

print("\n===== Optimizer Loss Comparison =====")
print(loss_table_optimizers)


plt.figure(figsize=(10, 6))

epochs_opt = loss_table_optimizers["Epoch"]


def plot_optimizer(col_name):
    values = [
        np.nan if value == "STOPPED" else value
        for value in loss_table_optimizers[col_name]
    ]
    plt.plot(
        epochs_opt,
        values,
        marker="o",
        label=col_name
    )


plot_optimizer("SGD")
plot_optimizer("AdaGrad")
plot_optimizer("AdaDelta")
plot_optimizer("RMSprop")
plot_optimizer("NAG")
plot_optimizer("Adam")

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Epoch vs Loss for Different Optimizers")
plt.xticks(list(epochs_opt))
plt.legend()
plt.grid(True)
plt.show()


print("\n===== Project Completed =====")
print("Optimization techniques compared:")
print("SGD, AdaGrad, AdaDelta, RMSprop, NAG and Adam")

print("\nRegularization techniques compared:")
print("L1, L2, Dropout, Early Stopping and Batch Normalisation")
