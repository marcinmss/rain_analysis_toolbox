from numpy import empty, log2, ndarray, ones
from numpy.random import choice

"""
Creates generates an array with the Alpha model
"""


def alpha_sym(n: int, c: float, gamma_bost: float) -> ndarray:
    # Assert that all variables are in order
    assert n > 0, "The value of n needs to be bigger than 0."
    assert c > 0, "The value of c needs to be bigger than 0."
    assert gamma_bost > 0, r"The value of gamma_+ needs to be bigger than 0."

    lamb1 = 2
    # Calculate the probablities
    boost_prob = lamb1**-c
    deacrease_prob = 1 - boost_prob
    boost_factor = lamb1**gamma_bost
    deacrease_factor = (1 - boost_factor * boost_prob) / deacrease_prob
    print(f"Decrease_factor = {deacrease_factor}")
    print(f"Gamma_decrease = {log2(deacrease_factor)}")

    # Generate the initial array
    old_array = ones(1)
    new_array = empty(1, dtype=float)

    # Start the iteration
    n_items = 1
    for i in range(n):
        # Update the number of items and create a new array
        n_items *= lamb1
        new_array = empty(n_items, dtype=float)

        # Fill the new array with the
        for i in range(n_items):
            idx = i // lamb1
            mult_factor = choice(
                (boost_factor, deacrease_factor, 0), p=(boost_prob, deacrease_prob, 0)
            )
            new_array[i] = old_array[idx] * mult_factor
        old_array = new_array

    return new_array


"""
Creates generates an array with the Beta model
"""


def beta_sym(n: int, c: float):
    assert n >= 1, "The value n needs to be bigger than 0."

    lamb1 = 2
    alive_prob = lamb1 ** (-c)
    dead_prob = 1 - alive_prob
    old_array = ones(1)
    new_array = empty(1, dtype=float)
    n_items = 1
    for i in range(n):
        # Update the number of items and create a new array
        n_items *= lamb1
        new_array = empty(n_items, dtype=float)

        # Fill the new array with the
        for i in range(n_items):
            idx = i // lamb1
            mult_factor = choice((1, 0), p=(alive_prob, dead_prob))
            new_array[i] = old_array[idx] * mult_factor
        old_array = new_array

    return new_array
