# A Machine Learning Framework Comparison

This repository is the companion code to my blog post [**Flux: The Flexible Machine Learning Framework for Julia**](https://alecokas.github.io/julia/flux/2020/07/09/flux-flexible-ml-for-julia.html), where I discuss using Flux as a framework for machine learning in Julia. This takes the form of presenting side-by-side comparisons between the Keras Functional API, PyTorch, and two Flux implementations. See the `src` directory for the full scripts.

## Training
The four scripts can be run as follows:
#### Functional Flux
Add all packages in the `julia-env/Project.toml` file.
```
julia --project=julia-env src/flux_functional_mnist.jl 
```
#### Modular Flux
Add all packages in the `julia-env/Project.toml` file.
```
julia --project=julia-env src/flux_modular_mnist.jl 
```
#### Tensorflow 2
```
$ python3 -m venv ../tf2env
$ source activate ../tf2env/bin/activate
$ pip install -r tf2-requirements.txt
$ python src/tf2_mnist.py
```
#### PyTorch
```
$ python3 -m venv ../ptenv
$ source activate ../ptenv/bin/activate
$ pip install -r pt-requirements.txt
$ python src/pt_mnist.py
```
