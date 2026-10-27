def resolve_input_shape(input_size, input_height=None, input_width=None):
    """Return the training crop shape as (height, width)."""
    if input_size is None and (input_height is None or input_width is None):
        raise ValueError("input_size is required unless both input_height and input_width are set.")

    height = input_height if input_height is not None else input_size
    width = input_width if input_width is not None else input_size
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"Input shape must be positive, got height={height}, width={width}.")
    return height, width
