def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) between two transcriptions using jiwer.
    Applies proper text normalization before comparison.

    Args:
        reference (str): Reference transcription
        hypothesis (str): Hypothesis transcription

    Returns:
        float: WER as a value between 0 and 1 (0 = perfect match, 1 = completely different)
    """
    import jiwer

    transformation = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )

    ref_normalized = transformation(reference)
    hyp_normalized = transformation(hypothesis)

    if not ref_normalized[0] and not hyp_normalized[0]:
        return 0.0
    elif not ref_normalized[0] or not hyp_normalized[0]:
        return 1.0
    else:
        return jiwer.wer(reference, hypothesis)
