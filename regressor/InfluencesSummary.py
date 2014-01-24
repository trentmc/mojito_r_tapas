class InfluencesSummary:
    """Use this object to pass around semi-raw influence information.
    Can construct this automatically by calling rfor.influencesSummary().
    Rfor and Sgb have routines to calculate this object directly; other
    regressors can too.

    @attributes:
    
        infl_per_var -- 1d array of length nvars -- 1-variable influences
          where infl_per_var[i] = relative influence of var 'i'
        infl_per_2var -- dict of 2var_tuple : relative_influence --
          2-var influences, where a 2var_tuple is (int i, int j) and i < j,
          and relative_influence is a float
    """
    def __init__(self, infl_per_var, infl_per_2var):
        """
        """
        self.infl_per_var = infl_per_var
        self.infl_per_2var = infl_per_2var
