import math
from numbers import Number
from typing import Iterable, Sequence, TypeVar


NT = TypeVar("NT", bound=Number)


def vector_add(
    vector_a: Iterable[NT],
    vector_b: Iterable[NT] | NT
) -> list[NT]:
    """Add two vectors together."""
    if isinstance(vector_b, Number):
        return [a + vector_b for a in vector_a]
    return [a + b for a, b in zip(vector_a, vector_b)]


def vector_subtract(
    vector_a: Iterable[NT],
    vector_b: Iterable[NT] | NT
) -> list[NT]:
    """Subtract two vectors from each other."""
    if isinstance(vector_b, Number):
        return [a - vector_b for a in vector_a]
    return [a - b for a, b in zip(vector_a, vector_b)]


def vector_multiply(
    vector_a: Iterable[NT],
    vector_b: Iterable[NT] | NT
) -> list[NT]:
    """Multiply two vectors together."""
    if isinstance(vector_b, Number):
        return [a * vector_b for a in vector_a]
    return [a * b for a, b in zip(vector_a, vector_b)]


def vector_divide(
    vector_a: Iterable[NT],
    vector_b: Iterable[NT] | NT
) -> list[NT]:
    """Divide two vectors from each other."""
    if isinstance(vector_b, Number):
        return [a / vector_b for a in vector_a]
    return [a / b for a, b in zip(vector_a, vector_b)]


def vector_dot(
    vector_a: Iterable[NT],
    vector_b: Iterable[NT]
) -> NT:
    """Dot two vectors together."""
    return sum(vector_multiply(vector_a, vector_b))


def vector_cross(
    vector_a: Sequence[NT],
    vector_b: Sequence[NT]
) -> list[NT]:
    """
    Cross two vectors together.
 
    The cross product of two vectors is a vector perpendicular to both.
    This is only valid for 3D vectors.
    """
    if len(vector_a) != 3 or len(vector_b) != 3:
        raise ValueError("Cross product only valid for 3D vectors.")
    return [
        (vector_a[1] * vector_b[2]) - (vector_a[2] * vector_b[1]),
        (vector_a[2] * vector_b[0]) - (vector_a[0] * vector_b[2]),
        (vector_a[0] * vector_b[1]) - (vector_a[1] * vector_b[0])
    ]


def vector_magnitude(vector: Iterable[NT]) -> NT:
    """Get the magnitude of a vector."""
    return sum([item ** 2 for item in vector]) ** 0.5


def vector_normalize(vector: Iterable[NT]) -> list[NT]:
    """Normalize a vector."""
    magnitude = vector_magnitude(vector)
    return [item / magnitude for item in vector]


def vector_angle_between(
    vector_a: Iterable[NT],
    vector_b: Iterable[NT]
) -> NT:
    """Get the angle between two vectors."""
    return (vector_dot(vector_a, vector_b)
            / (vector_magnitude(vector_a)
               * vector_magnitude(vector_b)))


def vector_angle_around_axis(
    vector_a: Iterable[NT],
    vector_b: Iterable[NT],
    axis: Iterable[NT]
) -> NT:
    """
    Get the angle between two vectors around an axis.

    This is only valid for 3D vectors.
    """
    if len(vector_a) != 3 or len(vector_b) != 3 or len(axis) != 3:
        raise ValueError("Angle around axis only valid for 3D vectors.")
    return math.acos(
        vector_dot(
            vector_normalize(vector_a),
            vector_normalize(vector_b)
        )
    )


def vector_rotate_around_origin(
    vector: Iterable[NT],
    axis: Iterable[NT],
    angle: NT
) -> list[NT]:
    """
    Rotate a vector around the origin by a given angle.

    This is only valid for 3D vectors.
    """
    if len(vector) != 3 or len(axis) != 3:
        raise ValueError("Rotation only valid for 3D vectors.")
    return vector_rotate_around_point(
        vector,
        (0, 0, 0),
        axis,
        angle
    )


def vector_rotate_around_point(
    vector: Iterable[NT],
    point: Iterable[NT],
    axis: Iterable[NT],
    angle: NT
) -> list[NT]:
    """
    Rotate a vector around a point by a given angle.

    This is only valid for 3D vectors.
    """
    if len(vector) != 3 or len(point) != 3 or len(axis) != 3:
        raise ValueError("Rotation only valid for 3D vectors.")
    return vector_add(
        vector_rotate_around_axis(
            vector_subtract(vector, point),
            axis,
            angle
        ),
        point
    )


def vector_rotate_around_axis(
    vector: Iterable[NT],
    axis: Iterable[NT],
    angle: NT
) -> list[NT]:
    """
    Rotate a vector around an axis by a given angle.

    This is only valid for 3D vectors.
    """
    if len(vector) != 3 or len(axis) != 3:
        raise ValueError("Rotation only valid for 3D vectors.")
    axis = vector_normalize(axis)
    x, y, z = vector
    u, v, w = axis
    cos = math.cos(angle)
    sin = math.sin(angle)
    return [
        (u * (u * x + v * y + w * z)
         * (1 - cos)
         + x * cos
         + (-w * y + v * z) * sin),
        (v * (u * x + v * y + w * z)
         * (1 - cos)
         + y * cos
         + (w * x - u * z) * sin),
        (w * (u * x + v * y + w * z)
         * (1 - cos)
         + z * cos
         + (-v * x + u * y) * sin)
    ]


def vector_rotate(
    vector: Iterable[NT],
    axis: Iterable[NT],
    angle: NT
) -> list[NT]:
    """
    Rotate a vector around an axis by a given angle.

    This is only valid for 3D vectors.
    """
    if len(vector) != 3 or len(axis) != 3:
        raise ValueError("Rotation only valid for 3D vectors.")
    axis = vector_normalize(axis)
    x, y, z = vector
    u, v, w = axis
    cos = math.cos(angle)
    sin = math.sin(angle)
    return [
        (u * (u * x + v * y + w * z)
         * (1 - cos)
         + x * cos
         + (-w * y + v * z) * sin),
        (v * (u * x + v * y + w * z)
         * (1 - cos)
         + y * cos
         + (w * x - u * z) * sin),
        (w * (u * x + v * y + w * z)
         * (1 - cos)
         + z * cos
         + (-v * x + u * y) * sin)
    ]


def vector_reflect(
    vector: Iterable[NT],
    normal: Iterable[NT]
) -> list[NT]:
    """Reflect a vector off a normal."""
    return vector_subtract(
        vector,
        vector_multiply(
            normal,
            2 * vector_dot(vector, normal)
        )
    )


def vector_project_onto(
    vector_a: Iterable[NT],
    vector_b: Iterable[NT]
) -> list[NT]:
    """Project a vector onto another vector."""
    return vector_multiply(vector_b,
                           vector_dot(vector_a, vector_b)
                           / vector_dot(vector_b, vector_b))


def enlarge_vector_through_point(   
    vector: Iterable[NT],
    point: Iterable[NT],
    factor: NT
) -> list[NT]:
    """Enlarge a vector through a point by a given factor."""
    return vector_add(
        vector_multiply(vector, factor),
        vector_multiply(point, 1 - factor)
    )

def vector_refraction(
    vector: Iterable[NT],
    normal: Iterable[NT],
    ratio: NT
) -> list[NT]:
    """Get the refraction of a vector off a normal."""
    dot = vector_dot(vector, normal)
    return vector_subtract(
        vector_multiply(vector, ratio),
        vector_multiply(normal, ratio * dot + math.sqrt(1 - ratio ** 2 * (1 - dot ** 2)))
    )


def vector_reject(
    vector_a: Iterable[NT],
    vector_b: Iterable[NT]
) -> list[NT]:
    """Reject a vector from another vector."""
    return vector_subtract(vector_a, vector_project_onto(vector_a, vector_b))


def vector_interpolate(
    vector_a: Iterable[NT],
    vector_b: Iterable[NT],
    factor: NT
) -> list[NT]:
    """Interpolate between two vectors."""
    return vector_add(
        vector_multiply(vector_a, 1 - factor),
        vector_multiply(vector_b, factor)
    )
