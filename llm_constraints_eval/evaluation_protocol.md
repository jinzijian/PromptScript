- Evaluations
    - 2 dims: 3 types X difficulty rating (human labelled)
    - Bucketing difficulty rating to low (1-3) and high (4-5) to minimize human label ambiguity, e.g. 1 and 5 are rare
    - Single type: 3x3 evals
        - Subset per type with high / low
        - Target analysis:
            - high -> binary, test give up rate, should be close to 100%
            - low + high -> test if giving-up is aligned with the human expectation
            - any difference on the constraint type?
    - Multiple types: 
        - 2 / 3 types all low: 3 + 1 -> test if model can pass with multiple easy constraints
        - 1 / 2 low + 1 high: 3 x 3 + 3 -> test if model is sensitive to the hardest in all given constraints
    - Test run:
        - 40x 3 types all low + 20x 2 types low + 1 high (per type) x 3 = 100 items 