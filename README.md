# Darcy Flow Circle Dataset

## Dataset

The dataset is the solution of the weak form of the Darcy Flow equation
 $$\text{find } u\in H^{1}(\Omega) \text{s.t} $$ 
$$-\int_{\Omega}a(x) \nabla v(x) \cdot \nabla u(x)dx=\int_{\Omega}v(x)dx \quad \forall v\in H_{0}^{1}(\Omega)$$

where 

$$\Omega=\{(x^{2}-0.5)+(y^{2}-0.5)\le 0.45\}$$ 
and $$a \sim \mu \text { where } \mu=f\#\mathcal{F}\left(0,(-\Delta+9 I)^{-2}\right)$$ and $$f(x)=\begin{cases}
12 & x\ge 0\\
3 & x<0
\end{cases}$$

This system has an unique solution. Note that as $a(x)$ is not continous, the solution is not a strong solution.
In the dataset class file, some weights useful to compute quadrature formulas are computed.

