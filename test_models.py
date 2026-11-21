import torch

from models import create_model


INPUT_DIM = 104

MODELS = [
    "Linear",
    "SmallMLP",
    "DeepMLP",
    "CNN1D"
]


print("=" * 60)
print("MODEL VERIFICATION")
print("=" * 60)


# Fake batch only for architecture testing
x = torch.randn(8, INPUT_DIM)


for model_name in MODELS:

    print("\nTesting:", model_name)

    model = create_model(
        model_name,
        INPUT_DIM
    )

    output = model(x)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Output shape:", output.shape)

    print(
        "Trainable parameters:",
        parameters
    )


print("\n" + "=" * 60)
print("ALL MODELS VERIFIED")
print("=" * 60)