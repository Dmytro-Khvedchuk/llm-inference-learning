# Brief notes of the decode loop from scratch.

1. By using KV cache we have ≈ O(N) time complexity with a slight drift. By not using it we have the ≈ O(N^2)
2. The top_p usually good to apply after the temperature. When we use the high temperature and after that apply the top_p we are increasing the number of tokens that goes under our top_p threshold. The top_k is left untouched.
3. Currently it processes the (1, T, V) but sooner we will use the batches. (first dimension)