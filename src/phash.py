import numpy as np

class PHash:
    """Perceptual Hash abstract class..."""
    def __init__(self, img):
        """Contructor.

        :param img: the image to get the hash to compute.
        
        :type img: numpy.ndarray
        """
        NotImplemented

    def __compute_hash(self):
        """Compute the hash if not computed, 
        else return the hash allready computed

        :return: perceptual hash of the image
        """
        NotImplemented

    def __int__(self):
        """To get the perceptual hash as an int.

        :return: perceptual hash
        :rtype: int
        """
        NotImplemented

    def __str__(self):
        """To get the perceptual hash as a str.

        :return: perceptual hash
        :rtype: str
        """
        NotImplemented

    def __sub__(self, other_hash):
        """To get the distance between 2 PHash.

        :param other_hash: the other hash to compute the distance
        
        :type other_hash: PHash


        :return: the distance between 2 hashes
        :rtype: float
        """
        NotImplemented

