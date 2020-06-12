Basics of Geometric Multigrid
=============================

Introduction
^^^^^^^^^^^^

In the following, we are describing the geometric multigrid method,
which for certain problems yields an iterative solver for linear equations
with optimal time complexity, i.e. the solver returns a solution to a PDE in
:math:`O(n_{\text{DoFs}})` arithmetic operations. We will show that this can also
be achieved for some convection-diffusion equations on uniformly refined triangular
meshes, when discretizing with linear finite elements.

Problem setup
^^^^^^^^^^^^^
TODO: function space :math:`V`
TODO: domain :math:`\Omega`
  
Find :math:`u \in V` such that

.. math::

  a(u,v) = f(v) \quad \forall v \in V

where :math:`a: V \times V \rightarrow \mathbb{R}` is the bilinear form defined as

.. math::

  a(u,v) := \text{TODO}

and the right hand side :math:`f: V \rightarrow \mathbb{R}` is a linear form defined as

.. math::

  f(v) := \text{TODO}

TODO: Code Examples !!!!!


.. math::

   \text{softmax}(x_i) := \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}.

::

   >>> from activations import Softmax
   >>> f = Softmax()

.. tip::

  Activation functions can also be added to dense layers. If no activation function is
  specified, the layer uses the Sigmoid function. E.g. you can add ReLU to a dense layer via

  ::

     >>> from layers import Dense
     >>> from activations import ReLU
     >>> denseLayer = Dense(inputDim = 3, outputDim = 4, activation = ReLU())
