"""Python script for Exercise set 6 of the Unsupervised and Reinforcement Learning.
"""

import numpy as np
import matplotlib.pylab as plb
from math import exp, log
import sys
from scipy import stats
from scipy.interpolate import KroghInterpolator

def kohonen():
    """Example for using create_data, plot_data and som_step.
    """
    plb.close('all')

    dynamicEta=True
    dynamicSigma=True
    Equilibrate=False
    
    dim = 28*28
    data_range = 255.0
    
    # load in data and labels
    data = np.array(np.loadtxt('data.txt'))
    labels = np.loadtxt('labels.txt')
    
    # select 4 digits
    name = 'Lorkowski' # REPLACE BY YOUR OWN NAME
    targetdigits = name2digits(name) # assign the four digits that should be used
    print targetdigits # output the digits that were selected
    # this selects all data vectors that corresponds to one of the four digits
    data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]

    # filter the label
    labels = labels[np.logical_or.reduce([labels==x for x in targetdigits])]
    
    dy, dx = data.shape
    
    #set the size of the Kohonen map. In this case it will be 6 X 6
    size_k = 8
    
    #set the width of the neighborhood via the width of the gaussian that
    #describes it
    #initial_sigma = 5
    initial_sigma = float(size_k/2)
    sigma = [initial_sigma]

    #initialise the centers randomly
    centers = np.random.rand(size_k**2, dim) * data_range
    
    #build a neighborhood matrix
    neighbor = np.arange(size_k**2).reshape((size_k, size_k))

    #set the learning rate
    eta = [0.1] # HERE YOU HAVE TO SET YOUR OWN LEARNING RATE
    
    #set the maximal iteration count
    tmax = (500*size_k*size_k) + 1000 # this might or might not work; use your own convergence criterion
    #tmax = 20000
    
    #set the random order in which the datapoints should be presented
    i_random = np.arange(tmax) % dy
    np.random.shuffle(i_random)

    #convergence criteria
    tol = 0.1
    previousCenters = np.copy(centers);
    
    errors = []
    mErrors = []
    logError = []
    finalErrors = []
    
    tailErrors = [0.0]; # 500 last errors

    if ((dynamicEta == True)&(dynamicSigma == True)):
        filename = 'k'+str(size_k)+'dynamicEta'+str(eta[0])+'dynamicSigma'+str(sigma[0])+'_tmax'+str(tmax)
        print filename
    elif ((dynamicEta == True)&(dynamicSigma == False)):
        filename = 'k'+str(size_k)+'dynamicEta'+str(eta[0])+'sigma'+str(sigma[0])+'_tmax'+str(tmax)
        print filename
    elif ((dynamicEta == False)&(dynamicSigma == True)):
        filename = 'k'+str(size_k)+'eta'+str(eta[0])+'dynamicSigma'+str(sigma[0])+'_tmax'+str(tmax)
        print filename
    else:
        filename = 'k'+str(size_k)+'eta'+str(eta[0])+'sigma'+str(sigma[0])+'_tmax'+str(tmax)
        print filename

    #convergedList=[]
    #numConverged=0
    #holdConvergedLabelsCount=0
    #t=-1
        
    for t, i in enumerate(i_random):

        '''
        if ( labels[i] in convergedList ):
            holdConvergedLabelsCount += 1
            if (holdConvergedLabelsCount >= len(targetdigits)):
                del convergedList[:]
                holdConvergedLabelsCount = 0
                numConverged = 0
                print "releasing labels"
            continue

        t+=1 # If you use this with t in the iterator to tn
        '''

        if dynamicEta == True:
            new_eta = eta[0] * exp(-float(t)/float(tmax))
            '''
            C = tmax/100
            new_eta = C * eta[0] / (C+t)
            '''
            eta.append(new_eta)

        if dynamicSigma == True:
            if sigma[0] == 1:
                new_sigma = sigma[0]
            else:
                mlambda = tmax/log(sigma[0])
                new_sigma = sigma[0] * exp(-float(t/mlambda))
            sigma.append(new_sigma)

        # Change to sigma[0] for static and sigma[t] for dynamic neighborhood function
        if ((dynamicEta == True)&(dynamicSigma == True)):
            som_step(centers, data[i,:],neighbor,eta[t],sigma[t])
        elif ((dynamicEta == False)&(dynamicSigma == True)):
            som_step(centers, data[i,:],neighbor,eta[0],sigma[t])
        elif ((dynamicEta == True)&(dynamicSigma == False)):
            som_step(centers, data[i,:],neighbor,eta[t],sigma[0])
        else:
            som_step(centers, data[i,:],neighbor,eta[0],sigma[0])

        # convergence check
        e = sum(sum((centers - previousCenters)**2))
        tailErrors.append(e)

        # Since this is an online method, the centers will most likely change in
        # the future even though the current iteration has a residual of 0.
        # Basing the convergence on the residual of the mean of the last 500 errors
        # may be a better convergence criterion.
        if(t > 500):
            if(len(tailErrors) >= 500):
                tailErrors.pop(0)
            tailErrors.append(e)
            
            # Update the mean error term
            tmpError = sum(tailErrors) / len(tailErrors)
            mErrors.append(tmpError)

            if t > (500*size_k*size_k):
                tolerance_check = np.abs(mErrors[-1] - mErrors[-501])
                logError.append(tolerance_check)
                data_print_static("Tol Error Minimum Is: {0}, "
                                  "Iteration: {1}, Current Error: {2}".
                                  format(np.min(logError), t,
                                         tolerance_check))
                if logError[-1] < tol:
                    print ""
                    print "Converage after ", t, " iterations"
                    break 
                """
                future_tolerance_check = np.sum(mErrors[-500])/500
                past_tolerance_check = np.sum(mErrors[-1000:-500])/500
                tolerance_check = np.abs((future_tolerance_check -
                                          past_tolerance_check))
                logError.append(tolerance_check)
                if np.size(logError) > 2:
                    log_d_v = ((np.sqrt((logError[-2] - logError[-1])**2)))
                    # plb.scatter(t, log_d_v, color='red')
                    # plb.pause(0.1)
                    finalErrors.append(log_d_v)
                    data_print_static("Tol Error Minimum Is: {0}, "
                                      "Iteration: {1}, Current Error: {2}".
                                      format(np.min(finalErrors), t,
                                             log_d_v))
                    if log_d_v < tol:
                        print ""
                        print "Converage after ", t, " iterations"
                        break
                """
            """
            if (len(mErrors) >= 2):
                tolerance_check = np.abs(mErrors[-1] - mErrors[-2])
                finalErrors.append(tolerance_check)
                data_print_static("Tol Error Minimum Is: {0}, "
                                  "Iteration: {1}, Current Error: {2}".
                                  format(np.min(finalErrors), t, tolerance_check))
                if ((tolerance_check < tol) & (t >= 500*size_k*size_k)):
                    '''
                    numConverged +=1
                    convergedList.append(labels[i])
                    print "Holding "+str(labels[i])
                    if (numConverged == 4):
                        print ""
                        print "Converage after ", t, " iterations"
                        break
                    '''
                    print ""
                    print "Converage after ", t, " iterations"
                    break
            """
        errors.append(e)
        previousCenters = np.copy(centers);

    if Equilibrate == True:
        old_eta = eta[-1]
        new_tmax=0.10*tmax
        i_random = np.arange(new_tmax) % dy
        np.random.shuffle(i_random)
        for t, i in enumerate(i_random):
            new_eta=old_eta
            eta.append(new_eta)
                
            new_sigma = 1.0
            sigma.append(new_sigma)
                
            # Change to sigma[0] for static and sigma[t] for dynamic neighborhood function
            if ((dynamicEta == True)&(dynamicSigma == True)):
                som_step(centers, data[i,:],neighbor,eta[-1],sigma[-1])
            elif ((dynamicEta == False)&(dynamicSigma == True)):
                som_step(centers, data[i,:],neighbor,eta[0],sigma[-1])
            elif ((dynamicEta == True)&(dynamicSigma == False)):
                som_step(centers, data[i,:],neighbor,eta[-1],sigma[0])
            else:
                som_step(centers, data[i,:],neighbor,eta[0],sigma[0])

            # convergence check
            e = sum(sum((centers - previousCenters)**2))
            tailErrors.append(e)
            
            # Since this is an online method, the centers will most likely change in
            # the future even though the current iteration has a residual of 0.
            # Basing the convergence on the residual of the mean of the last 500 errors
            # may be a better convergence criterion.
            if(len(tailErrors) >= 500):
                tailErrors.pop(0)
                tailErrors.append(e)
            
            # Update the mean error term
            tmpError = sum(tailErrors) / len(tailErrors)
            mErrors.append(tmpError)

            if (len(mErrors) >= 2):
                tolerance_check = np.abs(mErrors[-1] - mErrors[-501])
                logError.append(tolerance_check)
                data_print_static("Tol Error Minimum Is: {0}, "
                                  "Iteration: {1}, Current Error: {2}".
                                  format(np.min(logError), t,
                                         tolerance_check))
                if logError[-1] < tol:
                    print ""
                    print "Converage after ", t, " iterations"
                    break 
            
            errors.append(e)
            previousCenters = np.copy(centers);
    

    # Find the digit assigned to each center
    index = 0;
    digits = []
    for i in range(0, size_k**2):
         index = np.argmin(np.sum((data[:] - centers[i, :])**2,1))
         digits.append(labels[index])
         
    print "Digit assignement to the clusters: \n"
    print np.resize(digits, (size_k, size_k))
    np.savetxt('data/'+filename+'_cluster.txt', np.resize(digits, (size_k, size_k)), fmt='%i')
        

    # for visualization, you can use this:
    for i in range(size_k**2):
        plb.subplot(size_k,size_k,i+1)
        
        plb.imshow(np.reshape(centers[i,:], [28, 28]),interpolation='bilinear')
        plb.axis('off')

        plb.draw()
        
    # leave the window open at the end of the loop
    plb.savefig('data/'+filename+'_kohonen.pdf')
    plb.show()
    # plb.draw()

    import seaborn as sb

    if dynamicSigma == True:
        plb.plot(sigma)
        plb.ylabel('Sigma values')
        plb.xlabel('Iterations')
        plb.savefig('data/sigma.pdf')
        plb.show()

    if dynamicEta == True:
        plb.plot(eta)
        plb.ylabel('Learning Rate')
        plb.xlabel('Iterations')
        plb.savefig('data/eta.pdf')
        plb.show()

    plb.plot(errors)
    plb.ylabel('Sum of the Squared Errors')
    plb.xlabel('Iterations')
    plb.savefig('data/'+filename+'_sqerrors.pdf')
    plb.show()
    
    plb.plot(mErrors)
    plb.ylabel('Mean of last 500 errors')
    plb.xlabel('Iterations')
    plb.savefig('data/'+filename+'_mean500.pdf')
    plb.show()

    plb.plot(logError)
    plb.ylabel('Convergence Criteria')
    plb.xlabel('Iterations')
    plb.savefig('data/'+filename + '_convergence.pdf')
    plb.show()

def som_step(centers,data,neighbor,eta,sigma):
    """Performs one step of the sequential learning for a 
    self-organized map (SOM).
    
      centers = som_step(centers,data,neighbor,eta,sigma)
    
      Input and output arguments: 
       centers  (matrix) cluster centres. Have to be in format:
                         center X dimension
       data     (vector) the actually presented datapoint to be presented in
                         this timestep
       neighbor (matrix) the coordinates of the centers in the desired
                         neighborhood.
       eta      (scalar) a learning rate
       sigma    (scalar) the width of the gaussian neighborhood function.
                         Effectively describing the width of the neighborhood
    """

    size_k = int(np.sqrt(len(centers)))
    
    #find the best matching unit via the minimal distance to the datapoint
    b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2,1))

    # find coordinates of the winner
    a,b = np.nonzero(neighbor == b)
        
    # update all units
    for j in range(size_k**2):
        # find coordinates of this unit
        a1,b1 = np.nonzero(neighbor==j)
        # calculate the distance and discounting factor
        disc=gauss(np.sqrt((a-a1)**2+(b-b1)**2),[0, sigma])
        # update weights
        centers[j,:] += disc * eta * (data - centers[j,:])
        

def gauss(x,p):
    """Return the gauss function N(x), with mean p[0] and std p[1].
    Normalized such that N(x=p[0]) = 1.
    """
    return np.exp((-(x - p[0])**2) / (2 * p[1]**2))

def name2digits(name):
    """ takes a string NAME and converts it into a pseudo-random selection of 4
     digits from 0-9.
     
     Example:
     name2digits('Felipe Gerhard')
     returns: [0 4 5 7]
     """
    
    name = name.lower()
    
    if len(name)>25:
        name = name[0:25]
        
    primenumbers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
    
    n = len(name)
    
    s = 0.0
    
    for i in range(n):
        s += primenumbers[i]*ord(name[i])*2.0**(i+1)

    import scipy.io.matlab
    Data = scipy.io.matlab.loadmat('hash.mat',struct_as_record=True)
    x = Data['x']
    t = np.mod(s,x.shape[0])

    return np.sort(x[t,:])


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def data_print_static(data):
    """                                                                                   
    :rtype: Prints one line to Terminal
    """
    sys.stdout.write("\r\x1b[K" + data)
    sys.stdout.flush()

if __name__ == "__main__":
    kohonen()

