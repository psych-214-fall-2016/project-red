
# A more mathy explanation of MRF-EM

Let $P$ be the set of pixels, $x$ the set of labels for those pixels, and $y$ the set of pixel intensities. 

MRF-EM formulates segmentation as a maximum a posteriori (MAP) estimate. It tries to find the $x$ for which $P(x|y)$ is maximized. To find $x$, use Bayes' Rule: 

$P(x|y) \propto P(y|x) P(x)$

So all that needs to be done is find $P(y|x)$ and $P(x)$!

### Likelihood

The distribution of pixel intensities is modeled as a mixture of Gaussians, so each label is described by a Gaussian. The likelihood for each pixel $p \in P$ is 

$P(y_p|x_p) = \frac{1}{\sqrt{2 \pi \sigma_{x_p}^2}} exp\big(-\frac{(y_p - \mu_{x_p})^2}{2 \sigma_{x_p}^2}\big)
            = \frac{1}{\sqrt{2 \pi}} exp\big(-\frac{(y_p - \mu_{x_p})^2}{2 \sigma_{x_p}^2} - log(\sigma_{x_p})\big)
$

and with the assumption that pixel intensities are independent, 

$P(y|x) = \prod\limits_{p \in P} P(y_p|x_p)
        = \prod\limits_{p \in P} \Big[\frac{1}{\sqrt{2 \pi}} exp\big(-\frac{(y_p - \mu_{x_p})^2}{2 \sigma_{x_p}^2} - log(\sigma_{x_p})\big)\Big]$.
        
Pulling the constant out gives  

$P(y|x) = \frac{1}{\sqrt{2 \pi}} exp(-U(y|x))$ 

where

$U(y|x) = \sum\limits_{p \in P} \big(-\frac{(y_p - \mu_{x_p})^2}{2 \sigma_{x_p}^2} - log(\sigma_{x_p})\big)$

### Prior

This is where spatial information is incorporated! For reasons I don't understand, the set of labels $x$ can be thought of as an MRF. By the Hammersley-Clifford theorem (which I also do not understand), $x$ can be described with a Gibbs distribution. So, 

$P(x) = \frac{1}{Z} exp(-U(x))$.

The energy function $U(x)$ is described in terms of cliques, which are sets of pixels where every pixel in the set is the neighbor of every other pixel. MRF-EM only cares about "doubletons", or cliques of two pixels, when finding $U(x)$. $U(x)$ is defined as 

$U(x) = \sum\limits_{(p_i, p_{N_i}) \in C} V(x_{p_i}, x_{p_{N_i}})$

where $C$ is the set of doubletons, $(p_i, p_{N_i})$ is a clique (of pixel $i$ and its neighbor pixel $N_i$) and 

$V(x_{p_i}, x_{p_{N_i}}) = \beta \delta(x_{p_i}, x_{p_{N_i}})$, $\beta \in \mathbb{R}$. 

$\beta$ is set beforehand with larger values of $\beta$ corresponding to a more homogeneous labeling. 

### MAP

Now we have 

$P(x|y) \propto P(y|x) P(x) = \frac{1}{\sqrt{2 \pi}} exp(-U(y|x)) * \frac{1}{Z} exp(-U(x))$ 

This is a pain to calculate because of the exponential terms, so we find the log posterior instead: 

$log P(x|y) \propto log P(y|x) + log P(x) \propto -U(y|x) - U(x)$ 

Finding the maximum of $log P(x|y)$ is the same as finding the minimum of $-log P(x|y)$, so the MAP estimate is

$\hat x = \operatorname*{arg\,min}_x (U(y|x) + U(x))$. 

Although this equation looks simple, it is not trivial to find $\hat x$ as there is no analytical solution. I used the iterative conditional modes algorithm to try to estimate $\hat x$ (as suggested by the paper), which looks for the best labeling with at most one label different in every iteration. Iterative methods take forever and this one is no different. 

### EM

Oh no there's more? Unfortunately, yes. The liklihood estimation is based on modeling pixel intensities as a Mixture of Gaussians, but the parameters for each Gaussian need to be estimated. The $\mu, \sigma$ for each label is estimated using the M step of EM. The two steps of EM are

* Expectation: Given the current parameters, estimate the best labeling (the MAP estimate above)
* Maximization: Given the current labeling, estimate the best parameters

In the maximization step, $\mu_l$ and $\sigma_l$ for each label $l$ are updated to be the mean and variance, respectively, of the pixel intensities weighted by how likely it is for a pixel with some intensity to be assigned to that label:

$\mu_l^{new} = \frac{\sum\limits_{p \in P} P(l|y_p) y_p}{\sum\limits_{p \in P} P(l|y_p)}, 
 (\sigma_l^{new})^2 = \frac{\sum\limits_{p \in P} P(l|y_p) (y_p - \mu_l)^2}{\sum\limits_{p \in P} P(l|y_p)}$.
 
Here,

$P(l|y_p) = \frac{g(y_p|\mu_l, \sigma_l) P(l|x_{N_p})}{P(y_p)}$,

where

$g(y_p|\mu_l, \sigma_l) = \frac{1}{\sqrt{2 \pi \sigma_{l}^2}} exp\big(-\frac{(y_p - \mu_{l})^2}{2 \sigma_{l}^2}\big)$,

$P(l|x_{N_p}) = \frac{1}{Z} exp\big(-\sum\limits_{(p, p_{N_p}) \in C} V(l, x_{N_p})\big)$, 

$P(y_p) = \sum\limits_l P(y_p|l) = \sum\limits_l g(y_p|l)$. 

#### And that's it! Whoo!
