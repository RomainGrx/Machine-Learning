# Training Machine Learning

Ce repository a pour but de m'entrainer sur le machine learning par le biais de Python, TensorFlow et gym.

Les diffèrents modèles sont disponibles dans les ressources, tous les programmes sont disponibles dans le dossier source.

Convolutional Neural Networks (CNN)
===============
**Healthy.py**

Programme qui génère deux catégories de personnes: malade et en bonne santé et qui prédit ensuite si une personne est en bonne santé ou malade.

**fashion.py**

Programme qui s'entraine sur le dataset *fashion_mnist* de keras et qui classe les photos selon 10 catégories.

**10Categories.py & Cifar10.py**

Deux programmes qui gèrent le même dataset venant de 2 endroits différents.
*10Categories.py* gère un dataset téléchargé en local avec 50000 photos contenant 10 catégories.
*Cifar10.py* gère le dataset *cifar10* de keras avec ces mêmes photos et catégories.

Quality Learning (Q-Learning)
===============
You can train the model by running the program

Deep Quality Networks (DQN)
===============
You can play with the trained model
```Python
network.play(episodes = 1)
```
You can train the model
```Python
network.run(show = True)
```
