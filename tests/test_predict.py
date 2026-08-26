import torch

from src.model import CatsDogsCNN


def test_model_output_shape():

    model = CatsDogsCNN()

    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    )

    output = model(dummy_input)

    assert output.shape == (
        2,
        2
    )


def test_probabilities_sum_to_one():

    model = CatsDogsCNN()

    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    )

    with torch.no_grad():

        logits = model(dummy_input)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    assert abs(
        probabilities.sum().item() - 1.0
    ) < 1e-5