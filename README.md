# Solution approach
We devised an approach that focuses on processing and analysing the k LSBs of each pixel of the image under investigation. In essence, the k bits are used to yield a vectorial representation of the image to be offered to the DNN for the learning phase. The hidden layers combine this raw information and extract high-level discriminative features.

The DNN consists of a stack of different sub-nets. Each SubNet is composed of three main parts: a fully-connected dense layer equipped with a Rectified Linear Unit (ReLU), a batch-normalization layer and a dropout layer. The output layer of the DNN is instantiated on the basis of the specific task, detection or classification. 

## Authors

The code is developed and maintained by Massimo Guarascio and Nunziato Cassavia(massimo.guarascio@icar.cnr.it , nunziato.cassavia@icar.cnr.it)
