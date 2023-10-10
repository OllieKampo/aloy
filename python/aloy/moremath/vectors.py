import math
from numbers import Number
from typing import Iterable, Sequence, SupportsFloat, TypeVar


_NT = TypeVar("_NT", bound=SupportsFloat)
_NTC = TypeVar("_NTC", bound=Number)


def vector_cast(
    vector: Iterable[_NT],
    type_: type[_NTC]
) -> list[_NT] | list[_NTC]:
    """Cast a vector to a different type."""
    return [type_(item) for item in vector]


def vector_add(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT] | _NT
) -> list[float]:
    """Add two vectors together."""
    if isinstance(vector_b, Number):
        return [a + vector_b for a in vector_a]
    return [float(a + b) for a, b in zip(vector_a, vector_b)]


def vector_subtract(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT] | _NT
) -> list[float]:
    """Subtract two vectors from each other."""
    if isinstance(vector_b, Number):
        return [a - vector_b for a in vector_a]
    return [float(a - b) for a, b in zip(vector_a, vector_b)]


def vector_multiply(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT] | _NT
) -> list[float]:
    """Multiply two vectors together."""
    if isinstance(vector_b, Number):
        return [a * vector_b for a in vector_a]
    return [float(a * b) for a, b in zip(vector_a, vector_b)]


def vector_divide(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT] | _NT
) -> list[float]:
    """Divide two vectors from each other."""
    if isinstance(vector_b, Number):
        return [a / vector_b for a in vector_a]
    return [float(a / b) for a, b in zip(vector_a, vector_b)]


def vector_modulo(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT] | _NT
) -> list[float]:
    """Get the modulo of two vectors."""
    if isinstance(vector_b, Number):
        return [a % vector_b for a in vector_a]
    return [float(a % b) for a, b in zip(vector_a, vector_b)]


def vector_dot(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT]
) -> float:
    """Dot two vectors together."""
    return sum(vector_multiply(vector_a, vector_b))


def vector_cross(
    vector_a: Sequence[_NT],
    vector_b: Sequence[_NT]
) -> list[float]:
    """
    Cross two vectors together.

    The cross product of two vectors is a vector perpendicular to both.
    This is only valid for 3D vectors.
    """
    if len(vector_a) != 3 or len(vector_b) != 3:
        raise ValueError("Cross product only valid for 3D vectors.")
    return [
        float((vector_a[1] * vector_b[2]) - (vector_a[2] * vector_b[1])),
        float((vector_a[2] * vector_b[0]) - (vector_a[0] * vector_b[2])),
        float((vector_a[0] * vector_b[1]) - (vector_a[1] * vector_b[0]))
    ]


def vector_magnitude(vector: Iterable[_NT]) -> float:
    """Get the magnitude of a vector."""
    return math.sqrt(sum([item ** 2 for item in vector]))


def vector_distance(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT],
    manhattan: bool = False
) -> float:
    """Get the distance between two vectors."""
    if manhattan:
        return sum(abs(a - b) for a, b in zip(vector_a, vector_b))
    return vector_magnitude(vector_subtract(vector_a, vector_b))


def vector_distance_torus_wrapped(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT],
    size: Sequence[_NT],
    manhattan: bool = False
) -> float:
    """Get the distance between two vectors on a torus."""
    delta_abs = vector_abs(vector_subtract(vector_a, vector_b))
    for index, component in enumerate(delta_abs):
        if component > size[index] / 2.0:
            delta_abs[index] = size[index] - component
    if manhattan:
        return sum(delta_abs)
    return vector_magnitude(delta_abs)


def vector_between_torus_wrapped(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT],
    size: Sequence[_NT]
) -> list[float]:
    """Get the shortest vector between two vectors on a torus."""
    delta = vector_multiply(vector_subtract(vector_a, vector_b), -1.0)
    for index, component in enumerate(delta):
        if component > size[index] / 2.0:
            delta[index] = component - size[index]
        elif component < -size[index] / 2.0:
            delta[index] = component + size[index]
    return delta


def vector_midpoint(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT]
) -> list[float]:
    """Get the midpoint between two vectors."""
    return vector_divide(vector_add(vector_a, vector_b), 2.0)


def vector_normalize(vector: Iterable[_NT]) -> list[float]:
    """Normalize a vector."""
    magnitude = vector_magnitude(vector)
    return [item / magnitude for item in vector]


def vector_abs(vector: Iterable[_NT]) -> list[float]:
    """Get the absolute value of a vector."""
    return [float(abs(item)) for item in vector]


def vector_angle_around_point(
    vector_a: Iterable[_NT],
    point: Iterable[_NT]
) -> float:
    """
    Get the angle of a vector around a point in radians.

    This is only valid for 2D vectors.
    """
    return math.atan2(
        vector_a[1] - point[1],
        vector_a[0] - point[0]
    )


def vector_angle_between(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT]
) -> float:
    """Get the angle between two vectors in radians."""
    return math.acos(
        vector_dot(
            vector_normalize(vector_a),
            vector_normalize(vector_b)
        )
    )


def vector_angle_between_around_point(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT],
    point: Iterable[_NT]
) -> float:
    """Get the angle between two vectors around a point in radians."""
    return vector_angle_between(
        vector_subtract(vector_a, point),
        vector_subtract(vector_b, point)
    )


def vector_angle_between_around_axis(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT],
    axis: Iterable[_NT]
) -> float:
    """
    Get the angle between two vectors around an axis.

    This is only valid for 3D or higher vectors.
    """
    # TODO; Doesn't actually find angle around axis.
    if len(vector_a) < 3 or len(vector_b) < 3 or len(axis) < 3:
        raise ValueError("Rotation only valid for 3D or higher vectors.")
    return math.acos(
        vector_dot(
            vector_normalize(vector_a),
            vector_normalize(vector_b)
        )
    )


def vector_rotate_around_origin(
    vector: Iterable[_NT],
    axis: Iterable[_NT],
    angle: _NT
) -> list[float]:
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
    vector: Iterable[_NT],
    point: Iterable[_NT],
    axis: Iterable[_NT],
    angle: _NT
) -> list[float]:
    """
    Rotate a vector around an axis through a point by a given angle.

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
    vector: Iterable[_NT],
    axis: Iterable[_NT],
    angle: _NT
) -> list[float]:
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
    vector: Iterable[_NT],
    axis: Iterable[_NT],
    angle: _NT
) -> list[float]:
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
    vector: Iterable[_NT],
    normal: Iterable[_NT]
) -> list[float]:
    """Reflect a vector off a normal."""
    return vector_subtract(
        vector,
        vector_multiply(
            normal,
            2 * vector_dot(vector, normal)
        )
    )


def vector_project_onto(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT]
) -> list[float]:
    """Project a vector onto another vector."""
    return vector_multiply(
        vector_b,
        vector_dot(vector_a, vector_b)
        / vector_dot(vector_b, vector_b)
    )


def enlarge_vector_through_point(
    vector: Iterable[_NT],
    point: Iterable[_NT],
    factor: _NT
) -> list[float]:
    """Enlarge a vector through a point by a given factor."""
    return vector_add(
        vector_multiply(vector, factor),
        vector_multiply(point, 1 - factor)
    )


def vector_refraction(
    vector: Iterable[_NT],
    normal: Iterable[_NT],
    ratio: _NT
) -> list[float]:
    """Get the refraction of a vector off a normal."""
    dot = vector_dot(vector, normal)
    return vector_subtract(
        vector_multiply(vector, ratio),
        vector_multiply(
            normal,
            ratio * dot + math.sqrt(
                1.0 - ratio ** 2 * (1.0 - dot ** 2)
            )
        )
    )


def vector_reject(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT]
) -> list[float]:
    """Reject a vector from another vector."""
    return vector_subtract(vector_a, vector_project_onto(vector_a, vector_b))


def vector_interpolate(
    vector_a: Iterable[_NT],
    vector_b: Iterable[_NT],
    factor: _NT
) -> list[float]:
    """Interpolate between two vectors."""
    return vector_add(
        vector_multiply(vector_a, 1 - factor),
        vector_multiply(vector_b, factor)
    )
