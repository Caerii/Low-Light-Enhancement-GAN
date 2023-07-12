import numpy as np
    
def GetEigenvector(refl = None,retainE = None): 
    A = refl * np.transpose(refl)
    if (len(varargin) == 1):
        retainE = 6
    
    e,v = eig(A)
    v = diag(v)
    v = v(np.arange(end() - retainE + 1,end()+1))
    e = e(:,np.arange(end() - retainE + 1,end()+1))
    v = flipdim(v,1)
    e = flipdim(e,2)
    return e,v