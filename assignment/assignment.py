import logging
from numpy.typing import NDArray
from typing import Optional
import numpy as np
from cse587Autils.SequenceObjects.SequenceModel import SequenceModel

logger = logging.getLogger(__name__)

class SamplingSequenceModel(SequenceModel):
    def sample_motif(self):
        vectorized_choice = np.vectorize(
            # 4 base 
            pyfunc=lambda prob_base_i: np.random.choice(4, size=1, p=prob_base_i),
            signature="(m) -> ()"
        )
        return vectorized_choice(self.site_base_probs)

    def sample_background(self):
        # print("size",self.motif_length, "p",self.background_base_probs)
        return np.random.choice(
            4, # # mapping: 0 = A, 1 = C, 2 = G, 3 = T
            size= self.motif_length(),
            replace=True, 
            p=self.background_base_probs 
        )
    
    @classmethod
    def from_parent(cls, obj:SequenceModel):
        return cls(
            obj.site_prior, obj.site_base_probs, obj.background_base_probs,
            obj._precision, obj._tolerance
        )





def site_sample(sequence_model: SequenceModel,
                num_draws: int,
                seed: Optional[int] = None) -> NDArray:
    """Draw sequences based on parameters describing the probability of drawing
    a site sequence vs a background sequence. The parameters are stored in an
    object of the type SequenceModel. The probability of drawing a given
    base at a given position in a bound site is determined by the site_base_probs 
    while the probability of drawing a given base in a background sequence is 
    given by the background_base_probs. Note that the background_base_probs is a
    list of four floats, one for each base. The site_base_probs is a list of
    lists, where each internal list provides the probability of the four bases
    at that position in the site sequence.

    :param sequence_model: A SequenceModel object containing the site and
        background parameters.
    :type sequence_model: SequenceModel
    :param num_draws: The number of sequences to draw to create the sample.
    :type num_draws: int
    :param seed: The random seed to use. Defaults to 314.
    :type seed: int

    :return: A numpy array of shape (num_draws, sequence_model.motif_length())
        containing the drawn sequences.
    :rtype: numpy.ndarray

    :raises ValueError: If num_draws is not an integer greater than 1.

    :Example:

    >>> site_base_probs=[[0.2, 0.3, 0.1, 0.4], [0.1, 0.2, 0.4, 0.3]]
    >>> background_base_probs=[0.25, 0.25, 0.25, 0.25]
    >>> sm = SequenceModel(0.7, site_base_probs, background_base_probs)
    >>> result=site_sample(sm, 5)
    >>> print(result)
    [[2 1]
     [3 3]
     [1 3]
     [3 1]
     [1 3]]
    """
    if not isinstance(num_draws, int) or num_draws < 1:
        raise ValueError("Argument num_draws must be an integer >= 1")
    
    # set the random seed
    if seed != None:
        np.random.seed(seed)

    # Initialize a numpy array to store the randomly generated sequences
    # sample = np.zeros((num_draws, sequence_model.motif_length()), dtype=int)
    # Decide whether this draw will be a bound (motif) or an unbound 
        # (background) sequence. Hint: use numpy.random.choice
    draw_types = np.random.choice(
        2,
        size=num_draws,
        replace=True,
        p=[sequence_model.background_prior, sequence_model.site_prior]
    ).astype(bool)
    # Loop through num_draws to generate that many sequences
    sequence_model = SamplingSequenceModel.from_parent(sequence_model)


    def draw(draw_type:bool) -> NDArray[np.integer]:
        """
        draw_type: 0 if from background, 1 if from motif
        """
        return (
            sequence_model.sample_motif() 
            if draw_type else sequence_model.sample_background()   
            )
    sample = np.array(tuple(map(draw, draw_types)))

    # for draw_index in range(num_draws):

                # YOUR CODE HERE

        # Generate a sequence based on whether it's bound or not. Remember that
        # we are using integers to represent the bases with the following
                
        # YOUR CODE HERE "sequence_model.motif_length()" will be useful
 
    return sample
