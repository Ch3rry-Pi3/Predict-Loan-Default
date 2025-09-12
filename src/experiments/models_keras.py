# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

from typing import Any
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------------------------------------------
# Model builder helper
# -------------------------------------------------------------------

def build_keras_model(
    hidden_layers: int = 2,
    hidden_units: int = 128,
    dropout: float = 0.2,
    lr: float = 1e-3,
    input_dim: int | None = None,
    **kwargs: Any,
) -> keras.Model:
    """
    Factory for a binary-classification MLP in Keras (for use with SciKeras).

    Parameters
    ----------
    hidden_layers : int, default=2
        Number of hidden Dense layers.
    hidden_units : int, default=128
        Units per hidden layer.
    dropout : float, default=0.2
        Dropout probability applied after each hidden layer (0 disables).
    lr : float, default=1e-3
        Learning rate for the Adam optimiser.
    input_dim : int or None, default=None
        Number of input features. If None, attempts to infer from SciKeras meta.
    **kwargs : Any
        Extra arguments tolerated for SciKeras (e.g., classes, n_features_in_, meta).

    Returns
    -------
    keras.Model
        A compiled Keras model with a sigmoid output for binary classification.
    """

    # Infer input_dim if not provided
    if input_dim is None:
        input_dim = kwargs.get("n_features_in_") or kwargs.get("meta", {}).get("n_features_in_")

    # Define a simple MLP
    model = keras.Sequential([layers.Input(shape=(input_dim,))])
    for _ in range(hidden_layers):
        model.add(layers.Dense(hidden_units, activation="relu"))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compile with accuracy metric
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model