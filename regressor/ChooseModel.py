""" ChooseModel.py

Provides routines for choosing one model over another,
based on their attributes such as nmse and normality.
""" 

from util import mathutil

def chooseModel(model1, norm1, nmse1, model2, norm2, nmse2,
                normality_threshold = None):
    """
    @description
    
        Choose between two models. Choose first by infinite error, 
        then by normality (how close to normal distribution),
        finally by nmse.
    
    @arguments
    
        model1, model2: first and second model to choose between
        nmse1, nmse2: nmse for first and second model
        norm1, norm2: normality measure for first and second model 
        normality_threshold: if normality is below this, then choice
         will be based on normality.  Default is 0.60. (1.0=perfect, 0.0=worst)
    
    @return
    
        Tuple of the (best model, its normality, its nmse)

    @exceptions

    @notes

        Independent of the type of actual models, e.g. model1 could be
        a LinearModel and model2 could be an Rfor model.
    """
    if model1IsBest(model1, norm1, nmse1, model2, norm2, nmse2,
                    normality_threshold):
        return model1, norm1, nmse1
    else:
        return model2, norm2, nmse2
    
def model1IsBest(model1, norm1, nmse1, model2, norm2, nmse2,
                 normality_threshold = None):
    """
    @description

        Is model1 better than model2?
        Choose by: first by infinite error, then by normality (how close
        to normal distribution), finally by nmse.
    
    @arguments
    
        model1, model2: first and second model to choose between
        nmse1, nmse2: nmse for first and second model
        norm1, norm2: normality measure for first and second model 
        normality_threshold: if normality is below this, then choice
         will be based on normality.  Default is 0.60. (1.0=perfect, 0.0=worst)
    
    @return
    
        model1_is_best -- bool -- True if model1 is better than model2

    @exceptions

    @notes

        Independent of the type of actual models, e.g. model1 could be
        a LinearModel and model2 could be an Rfor model.
    """
    m1_is_best = True
    m2_is_best = False
    
    #if one or both models have 'inf' nmses, choose off that
    inf = float('Inf')
    if nmse1 == inf and nmse2 == inf: return m1_is_best
    if nmse1 == inf:                  return m2_is_best
    if nmse2 == inf:                  return m1_is_best

    #if one model has a terrible norm but other doesn't, choose from better norm
    if normality_threshold is None:
        normality_threshold = 0.60 # magic number
    if norm1 < normality_threshold and norm2 >= normality_threshold:
        return m2_is_best
    if norm2 < normality_threshold and norm1 >= normality_threshold:
        return m1_is_best

    #choose based on best nmse
    if nmse1 < nmse2: return m1_is_best
    else:             return m2_is_best

