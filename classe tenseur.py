
import numpy as np

class Tenseur :
    def __init__(self, tenseur, require_grad=False):
        self.data = np.array(tenseur)
        self.require_grad = require_grad
        self.grad = np.zeros_like(self.data)
        self._backward = lambda : None
        self.graph = ("",()) #(opération : str, (parents : Tenseur))

    def __repr__(self):
        return f"Tenseur(data = {self.data}, grad = {self.grad})"

    def __add__(self, other):
        if isinstance(other, Tenseur):
            out = Tenseur(self.data + other.data, require_grad=(self.require_grad or other.require_grad))
            out.graph = ("add", (self, other))
            def _backward():
                if self.require_grad:
                    self.grad += out.grad
                if other.require_grad:
                    other.grad += out.grad
        else:
            out = Tenseur(self.data + other, require_grad=self.require_grad)
            out.graph = ("add", (self,))
            def _backward():
                if self.require_grad:
                    self.grad += out.grad
        
        out._backward = _backward
        
        return out
    
    def backward(self):
        #1 Construire l ordre topo (existe que si graphe orienté + acyclique)
        #2 Initialiser le gradient final
        #3 Parcourir le graphe en ordre inverse
        #4 Appeler _backward() sur chaque tenseur
        pass

print(Tenseur([[1, 2], [3, 4]]))