# Darcy Flow Circle Dataset

## Dataset

The dataset is the solution of the weak form of the Darcy Flow equation

given $$a\in H^{1}([0,1]^{2})$$
find
$$u\in H^{1}(\Omega) \text{s.t} $$ 
$$-\int_{\Omega}a(x) \nabla v(x) \cdot \nabla u(x)dx=\int_{\Omega}v(x)dx \quad \forall v\in H_{0}^{1}(\Omega)$$

where 

$$\Omega=\{(x^{2}-0.5)+(y^{2}-0.5)\le 0.45\}$$ 

For generating the training set we choose
$$a \sim \mu \text { where } \mu=f \sharp \mathcal{N}(0,C)$$ with
$$C(x,y)=e^{-||x-y||}$$ 
and 
$$f(x)=\begin{cases} -x & x<-1\\
\frac{1}{2}x^{2}+\frac{1}{2} -1 & \ge x\le 1 \\
x & x>1 .$$

This system has an unique solution. As everything is regular, the solution is also a strong solution.

