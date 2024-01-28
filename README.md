# Darcy Flow Circle Dataset

## Dataset

The dataset is the solution of

 $$-\nabla \cdot(a(x) \nabla u(x))=1 \quad x \in\Omega$$
 $$u=0 \quad x \in\partial\Omega$$
where 
$\Omega$ is the unit circe and $$a \sim \mu \text { where } \mu=Lognormal\left(0,(-\Delta+9 I)^{-2}\right)$$


As we are in 2D this system has an unique weak solution.
In the dataset class file, some weights useful to compute quadrature formulas are computed.

