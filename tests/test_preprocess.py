from PIL import Image

from src.preprocess import get_eval_transform


def test_preprocessing_output_shape():

    image = Image.new(
        "RGB",
        (500, 300)
    )

    transform = get_eval_transform()

    tensor = transform(image)

    assert tensor.shape == (
        3,
        224,
        224
    )