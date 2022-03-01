# Solution approach
A Deep Learning Based approach for detecting and classifying stegomalware embedded in favicons.
Our approach focuses on processing and analysing the k LSBs of each pixel of the image under investigation. In essence, the k bits are used to yield a vectorial representation of the image to be offered to the DNN for the learning phase. The hidden layers combine this raw information and extract high-level discriminative features.

The DNN consists of a stack of different sub-nets. Each SubNet is composed of three main parts: a fully-connected dense layer equipped with a Rectified Linear Unit (ReLU), a batch-normalization layer and a dropout layer. The output layer of the DNN is instantiated on the basis of the specific task, detection or classification. 

## Authors

The code is developed and maintained by Massimo Guarascio, Nunziato Cassavia and Marco Zuppelli (massimo.guarascio@icar.cnr.it , nunziato.cassavia@icar.cnr.it, marco.zuppelli@ge.imati.cnr.it)


### Input details
We assume that the main directory includes a number of subfolders, one for each class (legitimate favicons and the ones embedded with php scripts or urls). Each subfolder contains the corresponding favicons.
