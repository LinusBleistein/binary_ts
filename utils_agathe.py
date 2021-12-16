import numpy as np
import torch
from scipy.special import softmax,expit
import time



""" TODO
- ajouter batch
- bien definir les dimensions de mu, nu et omega
""" 

def simulation_binaryTS(d=10,T=10,N=10, true_A=None, true_B=None, true_sigma=None):
        
    """
    Samples N samples over T timesteps from the DGP. If fixdata is set to True, sampled x data is saved in 
    an attribute called self.data, along with (N,T) used for sampling in self.N and self.T. 
    """

    if true_A is None:
        true_A = 0.8*np.identity(d) 
        
    if true_B is None:     
        random_eig = np.random.uniform(low=-0.7,high=0.7,size=d) 
        true_B = np.diag(random_eig) 
    
    if true_sigma is None:
        random_matrix = np.random.randn(d,d)
        true_sigma = np.dot(random_matrix,random_matrix.T)
    
    true_sigma_inv = np.linalg.inv(true_sigma)
    
    x_data = np.zeros((N,d,T))
    z_data = np.zeros((N,d,T))
    p_data = np.zeros((N,d,T))
    
    z0 = np.random.randn(d,N)
    p = expit(z0)
    x0 = np.random.binomial(1,p,size=(d,N))
    
    x = x0
    z = z0
    
    x_data[:,:,0] = x0.T
    z_data[:,:,0] = z0.T
    p_data[:,:,0] = p.T
    
    for t in np.arange(1,T): 
        z = true_A@x + true_B@z + np.random.multivariate_normal(mean=np.zeros(d),
                                                                cov=true_sigma,size=N).T
        p = expit(z)## softmax(z,axis=0) 
        x = np.random.binomial(1,p,size=(d,N))
        x_data[:,:,t] = x.T
        z_data[:,:,t] = z.T
        p_data[:,:,t] = p.T
    x_data = torch.from_numpy(x_data).type(torch.FloatTensor)
    return x_data,z_data,p_data

def compute_precision(nu,omega): ## Probleme de dimension
    
    """
    Given a (N,d,T-1)-tensor as input for nu and a a (N,d,T-2)-tensor as input for omega, forms a (N,d,T-1,T-1)-tensor structured as follows: 
        - for given i and j, the elements ???? of the (T-1,T-1) variance of $z_i^(j)$
        
    """
    
    B=torch.diag_embed(nu)+torch.diag_embed(omega,offset=1)
    return torch.transpose(B,dim0=2,dim1=3)@B

def compute_variance(nu,omega): ## Probleme de dimension
    
    """
   a faire
    """
    
    B=torch.diag_embed(nu)+torch.diag_embed(omega,offset=1)
    return torch.inverse(torch.transpose(B,dim0=2,dim1=3)@B)

def variational_entropy(nu,omega): ## Probleme de dimension
    """
    term 1.1 in companion paper
    """
    return 2*torch.log(nu).sum()

def from_classif(x_data,A,B,sigma_inv,mu,nu,omega,R):
    """
    term 1.1 in companion paper
    """
    N,d,T = x_data.shape
    result = (x_data[:,:,1:]*mu).sum()

    covariance_matrix = compute_variance(nu,omega) ##Probleme de dimension
    diagonal_terms = torch.diagonal(covariance_matrix,offset=0,dim1=2,dim2=3)
    draw = torch.randn(R,N,d,T-1)
    transformed_draw = torch.pow(diagonal_terms,1/2)*draw + mu
    #print(transformed_draw.shape)
    logaddexp_draw = torch.logaddexp(torch.zeros(R,N,d,T-1),transformed_draw)

    if torch.any(torch.isnan(logaddexp_draw)):
        print('Nans in full log approx')

    expect_approx = logaddexp_draw.mean(axis=0).sum()
 
    result -= expect_approx
    return result

def log_det(x_data,sigma_inv):
    """
    term 1.2.1 in companion paper
    """
    N,d,T = x_data.shape
    log_det = N*((T-1)/2)*torch.logdet(sigma_inv)
    return log_det 


def compute_bilinear_term(x_data,A,B,sigma_inv,mu,nu,omega):## Probleme de dimension
    """
    term 1.2.2 in companion paper
    """
    N,d,T = x_data.shape
    batch = np.arange(N)


    result = np.array([ torch.trace(mu[i,:,:].T@sigma_inv@(mu[i,:,:])) for i in np.arange(N)]).sum()
    #### A changer ####
    # result += sum([torch.trace((A@(x_data[i,:,:-1])).T@sigma_inv@(A@(x_data[i,:,:-1]))) for i in batch])
  #  result += torch.trace(mu[:,:-1].T@(B.T)@sigma_inv@B@(mu[:,:-1]))
### Attention pas de *2
  #  result += -2*sum([torch.trace((mu[:,1:].T@sigma_inv@(A@(x_data[i,:,:-1])))) for i in batch])
  #  result += -2*torch.trace(mu[:,1:].T@sigma_inv@B@(mu[:,:T-1]))
  #  result += 2*sum([torch.trace((A@(x_data[i,:,:-1])).T@sigma_inv@B@(mu[:,:-1])) for i in batch])

    
    #### FIN = a changer ####
   
    


    #Compute trace term
    covariance_matrix = compute_variance(nu,omega)
    diagonal_terms = torch.diagonal(covariance_matrix,offset=0,dim1=1,dim2=2)
    sub_diagonal_terms = torch.diagonal(covariance_matrix,offset=1,dim1=1,dim2=2)
    # a verifier #
    trace_term =(torch.diagonal(sigma_inv)*torch.transpose(diagonal_terms[:,:],dim0=1,dim1=2)).sum()
    # a changer en fonction de la ligne précédente  #
  #  trace_term += (torch.diagonal(B.T@sigma_inv@B)*(diagonal_terms[:,:T-1])).sum(axis=1).sum()
### Attention pas de *2
  #  trace_term += -2*(torch.diagonal(sigma_inv@B)*(sub_diagonal_terms)).sum(axis=1).sum()


    result += trace_term 

    result *= 1/2
    return result

def compute_minus_elbo(x_data,A,B,sigma_inv,mu,nu,omega,R=500):

    """
    Returns the Evidence Lower Bound (ELBO) of the model, normalized by 1/(N*T*d), as a pytorch tensor.
    """


    N,d,T = x_data.shape

    batch = np.arange(N)



    from_clas = from_classif(x_data,A,B,sigma_inv,mu,nu,omega,R)
    log_deter = log_det(x_data,sigma_inv)
    bilin = compute_bilinear_term(x_data,A,B,sigma_inv,mu,nu,omega)
    var_entropy = variational_entropy(nu,omega)

    #Put all terms together

    elbo_value =from_clas + log_deter - bilin - var_entropy

    return - elbo_value

def block_cavi(x_data,R,latent_lrs=[1e-2,1e-2,1e-2,1e-2,1e-4,1e-4],mu_lr=1e-2,nu_lr=1e-2,
               omeg_lr=1e-2,max_iter=100,its=100):#,batchsize=2): 

    """
    Optimization of the ELBO using a block-coordinate-ascent algorithm. 

    Arguments: 
        - latent_lrs (list): learning rates for resp. A, B and sigma;

        - mu_lr (scalar > 0): learning rate for the variational density mean parameters;

        - var_lr (scalar > 0): learning rate for the variational density variance parameters. 

    """

    N,d,T = x_data.shape
    batch = np.arange(N)

    #Parameters initialization
    sigma_inv = 0.5*torch.eye(d,d)
    sigma_inv.requires_grad=True

    A = torch.ones(d,d) + torch.randn(d,d)
    A.requires_grad = True


    B = 0.5*torch.ones(d,d)+torch.randn(d,d)
    B.requires_grad = True

    print('data shape:', x_data.shape)

    mu = x_data.mean(axis=0)
    mu = mu.float()
    mu.requires_grad = True
    print(mu.shape)

    nu = torch.ones(d,T)
    nu = nu.float()
    nu.requires_grad = True
    print(nu.shape)
    
    omega = torch.ones(d,T-1)
    omega = omega.float()
    omega.requires_grad = True
    print(omega.shape)

    var_dic = {"A":{},"B":{},"sigma_inv":{},"mu":{},"nu":{},"omega":{}}

    var_dic["A"]['variable']= A
    var_dic["B"]['variable']= B
    var_dic["sigma_inv"]['variable']= sigma_inv
    var_dic["mu"]['variable']= mu
    var_dic["nu"]['variable']= nu
    var_dic["omega"]['variable']= omega

    for i,key in enumerate(var_dic.keys()):

        var_dic[key]["lr"] = latent_lrs[i]
        var_dic[key]["optimizer"] =torch.optim.Adam([var_dic[key]['variable']],
                                                            lr=var_dic[key]['lr'])


    n_params = 6 

    print('-----------------------------------------')
    print('---------------------------------')
    print('--------------------------')
    print('-------------------')  
    print('\n')
    print('Block-CAVI initialization...')
    print('\n')
    print('--------Dimensions of the model--------')
    print('T: ', T)
    print('d: ', d)
    print('N: ', N)
    print('\n')
    print('-------- Optimization parameters --------')
   # print('Batchsize: ', batchsize)
    print('\n')
    print('Optimizer for A:', var_dic['A']['optimizer'] )
    print('Optimizer for B:', var_dic['B']['optimizer'] )
    print('Optimizer for sigma_inv:', var_dic['sigma_inv']['optimizer'] )
    print('Optimizer for mu:', var_dic['mu']['optimizer'] )
    print('Optimizer for nu:', var_dic['nu']['optimizer'] )
    print('Optimizer for omega:', var_dic['omega']['optimizer'] )
    print('\n')
    print('-------------------') 
    print('--------------------------')
    print('---------------------------------')
    print('-----------------------------------------')
    print('\n')
    print('\n')
    print('-------- TRAINING STARTS --------')
    print('\n')
    print('\n')


    t1 = time.time()
    elbos = []
    for t in np.arange(max_iter):
        elbo0 = compute_minus_elbo(x_data,A,B,sigma_inv,mu,nu,omega,R).detach().numpy()
        elbos.append(elbo0)
        if t==0:
            print(elbo0)


        for j in np.arange(n_params): 

            parameter = list(var_dic.keys())[j]
            optimizer = var_dic[parameter]["optimizer"]

            for it in np.arange(its):
                elbo = compute_minus_elbo(x_data,A,B,sigma_inv,mu,nu,omega,R)
                elbo.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

 #       for j in np.arange(3,T*d+3): 

 #           j += - 3
 #           parameter = list(mu_dic.keys())[j]
 #           optimizer = mu_dic[parameter]["optimizer"]
            
 #           for it in np.arange(its):
 #               elbo = compute_minus_elbo(x_data,A,B,sigma_inv,mu,nu,omega,R)
 #               elbo.backward(retain_graph=True)
 #               optimizer.step()
 #               optimizer.zero_grad()

 #       for j in np.arange(T*d+3,2*(T*d)+3):
 #           j += -(T*d+3)
 #           parameter = list(nu_dic.keys())[j]
 #           optimizer = nu_dic[parameter]["optimizer"]

 #           for it in np.arange(its):
 #               elbo = compute_minus_elbo(x_data,A,B,sigma_inv,mu,nu,omega,R)
 #               elbo.backward(retain_graph=True)
 #               optimizer.step()
 #               optimizer.zero_grad()
                
 #       for j in np.arange(2*(T*d)+3,3 + T*d + d*(2*T-1)):
 #           j += -2*(T*d)-3
 #           parameter = list(omega_dic.keys())[j]
 #           optimizer = omega_dic[parameter]["optimizer"]

 #           for it in np.arange(its):
 #               elbo = compute_minus_elbo(x_data,A,B,sigma_inv,mu,nu,omega,R)
 #               elbo.backward(retain_graph=True)
 #               optimizer.step()
 #               optimizer.zero_grad()
        
        elbo_new = compute_minus_elbo(x_data,A,B,sigma_inv,mu,nu,omega,R).detach().numpy()     

        if (np.abs(elbo_new-elbo0)/np.abs(elbo0)< 1e-5):
            break

        if t%10 == 0 and t > 1:
            print('---------- Iteration ', t, ' ---------- ')

            print('Current ELBO value: ', elbo)
            print('param A', A)
            print('param B', B)
            print('param sigma_inv', sigma_inv)
            print('param mu', mu)
            print('param nu', nu)
            print('param omega', omega)
            

    t2 = time.time()

    print('-------- TRAINING FINISHED IN ', t2-t1, ' SECONDS --------')
    return elbos,A,B,sigma_inv