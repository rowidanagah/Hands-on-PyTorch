![](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)


### *******I. Content*******

- **What is PyTorch**
- ***Tensor***
- **Automatic Differentiation and Autograd**
- **[Referances]**(#Referances)





[PyTorch](https://us.hidester.com/proxy.php?u=eJwrtjI0s1ISmnSq537GmV4ne9cnTHXz1JWsAXPICVc%3D&b=7) is an open source machine learning library based on the Torch library.
        
   >  It's an improvement overTorch framework, however, the most notable change is the adoption of a **Dynamic Computational Graph**.
        
   <img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/dynamic_graph.gif" width="800">

    
Developed by Facebook's AI Research lab *(FAIR)* .PyTorch emphasizes flexibility and allows DL models to be expressed in **idiomatic Python**.



PyTorch is a Python package that provides two high-level features:
   
   - Tensor computation (like NumPy) with strong GPU acceleration
   - Deep neural networks built on a tape-based autograd system

Similarly to NumPy, it also has a C backend, so they are both much faster than native Python, It's a replacement for NumPy to use the **power of GPUs** providing the max fexibility and speed.

- [More About PyTorch](https://github.com/pytorch/pytorch#more-about-pytorch)
- [Installation](https://github.com/pytorch/pytorch#installation)
 
 
 
 
### *******A GPU-Ready Tensor Library******* 
 
 
A `torch.Tensor` is a  specialized data structure that 1D or 2D matrix containing elements of a single data type .


PyTorch provides Tensors that can live either on the CPU or the GPU and accelerates the computation by a huge amount.

      
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
      
   - There are a few main ways to create a tensor, depending on your use case.
   - To create a tensor From a NumPy array , use `np.array(data)`
   - To create a tensor With random or constant values, use `torch.rand(shape)`.
   - To create a tensor From another tensor , `torch.rand_like(shape, datatype)`.
   - In-place operations Operations that have a _ suffix are in-place. For example: `x.copy_(y)`, `x.t_()`, will change x.




### *******Autograd*******


`torch.autograd` is PyTorch’s automatic differentiation engine that powers neural network training.

A tensor can be created with `requires_grad=True` so that torch.autograd records operations on them for automatic differentiation.

Autograd has multiple goals:

   - Full integration with neural network modules: mix and match auto-differentiation with user-provided gradients
   - The main intended application of Autograd is **gradient-based optimization**

*To get a conceptual understanding of how autograd helps a neural network train:*

![](https://blog.paperspace.com/content/images/2019/03/computation_graph.png)

   - Neural networks (NNs) are a collection of functions that are executed on input data and parameters (consisting of weights and biases),
   - Training a NN happens in two steps:
   
       1. A forward pass to compute the value of the loss function.
           > The forward pass is pretty straight forward. The output of one layer is the input to the next and so forth.
           
       2. A backward pass to compute the gradients of the learnable parameters
       ![](https://blog.paperspace.com/content/images/2019/03/full_graph.png)




III. **Referances**
------------

- [PyTorch docs autograd ](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [PyTorch docs Tensor](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch docs about Tensor](https://pytorch.org/docs/stable/tensors.html)

- [Developing deep learning models using  Pytorch](https://www.coursera.org/learn/deep-neural-networks-with-pytorch/home/welcome)

- [PyTorch Autograd](https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95)

- [Trying to understand what “save_for_backward” is in Pytorch question on stackoverflow](https://stackoverflow.com/questions/64460017/trying-to-understand-what-save-for-backward-is-in-pytorch)

- [PYTORCH: DEFINING NEW AUTOGRAD FUNCTIONS](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html)

- [Implementing a Transformer with PyTorch and PyTorch Lightning](https://www.linkedin.com/company/pytorch-lightning/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_recent_activity_details_shares%3BO3kQZoBQQd6AwlfGTfvmDg%3D%3D)

- [Full Stack Deep Learning, Colab:](https://lnkd.in/dRZTdBm)
