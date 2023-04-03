from typing import Callable, List


def length_check_basic(
    tokenizer: Callable[[str], List[str]], condition: Callable[[str], bool]
) -> Callable[[str], int]:
    """generic count function generator

    Parameters
    ----------
    tokenizer: Callable[[str],List[str]]
        a function that splits a text into components. Usually into words
    condition: Callable[[str],bool]
        a function that returns true if the token be counted

    Returns
    -------
    Callable[[str],float]
        a function that receives text as string,
        and outputs the number of tokens that are valid according to `condition`.

    """

    def length_check_basic_function(text: str) -> int:
        """length calculator
        calculates the count of a conditioned tokens

        Parameters
        ----------
        text: str
            the text to count on

        Returns
        -------
        float
            the count of tokenized and filtered text

        """
        tokens = tokenizer(text)
        corpus_filter = filter(condition, tokens)
        corpus_filter_count = len(list(corpus_filter))
        return corpus_filter_count

    return length_check_basic_function


def avg_check_basic(
    tokenizer: Callable[[str], List[str]], condition: Callable[[str], bool]
) -> Callable[[str], float]:
    """generic avg counter generator

    Parameters
    ----------
    tokenizer: Callable[[str],List[str]]
        a function that splits a text into components. Usually into words
    condition: Callable[[str],bool]
        a function that returns true if the token be counted

    Returns
    -------
    Callable[[str],float]
        a function that receives text as string,
        and outputs the avg length of tokens that are valid according to `condition`.

    """

    def avg_check_basic_function(text: str) -> float:
        """avg calculator
        calculates the avg length of a conditioned token

        Parameters
        ----------
        text: str
            the text to calculate the avg on

        Returns
        -------
        float
            the average length of a pre-conditioned text

        """
        tokens = tokenizer(text)
        corpus_filter = list(map(len, filter(condition, tokens)))
        corpus_filter_count = (
            sum(corpus_filter) / len(corpus_filter) if len(corpus_filter) != 0 else 0
        )
        return corpus_filter_count

    return avg_check_basic_function
