from math import ceil, floor
import itertools
# TODO remove or repackage these utility functions

    
class Hyperrect:
    "Class for a hyperrectangels"
    
    def __init__(self,ll,tr):
        """
        ll:lowerleft vector
        tr:topright vector
        """
        # checking if dimensions match
        if len(ll) != len(tr):
            raise ValueError('LowerLeft and TopRight vector size mismatch')
        
        # checking if ll values are leq tr values
        for j in range(0,len(ll)):
            if ll[j] > tr[j]:
                raise ValueError('LowerLeft not \leq TopRight at ' + str(j))
        
        self.ll = ll
        self.tr = tr
        self.dim = len(ll)
        self.gridsize = None
        self.gridlist = None
    
    @classmethod
    def MakeBall(cls, center, radius):
        ll = []
        tr = []
        for i in range(0, len(center)):
            ll.append(center[i]-radius[i])
            tr.append(center[i]+radius[i])
        return [ll, tr]
        
    def print_function(self):
        print('Hyperrectangle \n')
        print(str(self.ll) +'->'+ str(self.tr)+'\n')
        print(self.gridlist)
        
    def contains(self,x):
        "returns 1 if x \in self else returns 0"
        if len(x) != self.dim:
            raise ValueError('x and hyperrectangle dim')
        for i in range(0,self.dim):
            if x[i] > self.tr[i] or x[i] < self.ll[i]:
                return 0
        return 1
            
    '''
    def grid_1d(self,k,delta):
        llk = self.ll[k]
        dimrange0 = float(self.tr[k] - llk)
        dimcenter = llk + dimrange0/2
        A = []
        print "ceil: ", dimrange0/delta, " ; ", dimrange0/delta
        #for j in range(-1 * int(ceil(dimrange0/(2 * delta))), 1+ int(ceil(dimrange0/(2 * delta)))):
            # could add 1 in the end point of the range
            #    A.append(dimcenter + j*delta)
        return A
    '''
    def grid_1d(self,k,delta):
        llk = self.ll[k]
        dimrange0 = float(self.tr[k] - llk)
        A = []
        for j in range(0, int(ceil(dimrange0/delta))):
            # Sept. There was a 1 + int() in calculating the range.
            # could add 1 in the end point of the range
            A.append(llk+(delta/2) + j*delta)
        return A
    
    def grid_all(self,delta):
        "initializes the hyperrectangle with a grid"
        L = [self.grid_1d(0, delta[0])]
        for i in range(1,self.dim):
            L.append( self.grid_1d(i, delta[i]))
        self.gridlist = L
        self.gridsize = delta
        return L
     
    def grid(self, delta):
        "May make sense to use grid_all instead of this function"
        L = self.grid_all(delta)
        # print(L)
        G=[]
        for element in itertools.product(*L):
            G.append(element)
        return G
    
    def quantize(self,x):
        """Call this after calling grid_all on hyperrectangle
            returns quantized version of x
        """
        if len(x) != self.dim:
            raise ValueError('x and hyperrectangle dim')
        if self.gridlist == None:
            raise ValueError('hyperrectangle not gridded')
        if not self.contains(x):
            raise ValueError('hyperrectangle does not contain x')
        L = self.gridlist
        q = []
        for j in range(0,self.dim):
            xj = int(floor((x[j]-self.ll[j])/self.gridsize[j])) # ceil should also work
            #print(j, x[j], xj, L[j][xj])
            q.append(L[j][xj])
        return q  


# some code for testing the above functions

#R = Hyperrect([3,1,2],[9,4,6])
#R.print_function()
#R.grid_all([.7, .7, .7])
#R.print_function()
#q = R.quantize([3, 4.000, 6])
#print (q)
#print (Hyperrect.MakeBall(q,.1))
#A0 =  R.grid_1d(0,0.5)
#A1 =  R.grid_1d(1,0.5)
#L = R.grid_all(1)
#print(L)
#L = R.grid(.1)
#print(L)

#for element in itertools.product(*L):
#    print (element)
