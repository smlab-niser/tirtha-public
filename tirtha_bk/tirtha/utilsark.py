"""
Utility code adapted for Project Tirtha from `arklet` - 
https://github.com/internetarchive/arklet (MIT license)

"""
import secrets
from typing import Tuple


BETANUMERIC = "0123456789bcdfghjkmnpqrstvwxz"

def noid_check_digit(noid: str) -> str:
    """
    Calculate the check digit for an ARK.

    Parameters
    ----------
    noid : str
        The noid portion of an ARK
    
    Returns
    -------
    str
        The check digit for the ARK
    
    References
    ----------
    .. [1] https://metacpan.org/dist/Noid/view/noid#NOID-CHECK-DIGIT-ALGORITHM

    """
    total = 0
    for pos, char in enumerate(noid, start=1):
        score = BETANUMERIC.find(char)
        if score > 0:
            total += pos * score
    remainder = total % 29  # 29 == len(BETANUMERIC)

    return BETANUMERIC[remainder]  # IndexError may be long ARK

def generate_noid(length: int) -> str:
    """
    Generate a random NOID of the given length.

    Parameters
    ----------
    length : int
        The length of the NOID to generate

    Returns
    -------
    str
        The generated NOID

    """
    return "".join(secrets.choice(BETANUMERIC) for _ in range(length))

def parse_ark(ark: str) -> Tuple[int, str]:
    """
    Parse an ARK into its constituent parts.

    Parameters
    ----------
    ark : str
        The ARK to parse

    Returns
    -------
    Tuple[int, str]
        The ARK's NAAN, and name (includes shoulder)

    Raises
    ------
    ValueError
        If the ARK is not valid
    ValueError
        If the ARK's NAAN is not an integer

    """
    parts = ark.split("ark:")
    if len(parts) != 2:
        raise ValueError("Not a valid ARK.")

    _, ark = parts
    ark = ark.lstrip("/")
    parts = ark.split("/")
    if len(parts) < 2:
        raise ValueError("Not a valid ARK.")

    naan, name = parts[:2]
    try:
        naan_int = int(naan)
    except ValueError:
        raise ValueError("ARK NAAN must be an integer")

    return naan_int, name
