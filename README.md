# CS6910 Assignment 3
Assignment 3 submission for the course CS6910 Fundamentals of Deep Learning.

Student Information: Shivam Kharat (CS22M082)

Find Wandb report here : [https://rb.gy/dvqrgv](https://wandb.ai/vilgax/CS6910_Assignment3/reports/ASSIGNMENT-3--Vmlldzo0MzI5OTcy?accessToken=b6218h2p1zh5xkb0scyh3reka1x5ka220tlwcgaurpand2p8ragrsmkn9b91ekfi)
---
Code for EncoderRNN , DecoderRNN and training seq2seq model can be found in file named **Assignment3.ipynb** 
## Question 1
The code for question 1 can be accessed [here](https://github.com/Shivam10816/cs6910_assignment1/blob/main/Q1.ipynb). The program, reads the data from `keras.datasets`, picks one example from each class and logs the same to `wandb`.

## Questions 2-4
The neural network is implemented by the class `NeuralNetwork`, present in the `neural_network.ipynb` file.  
### Building a `NeuralNetwork`
An instance of `neural_network` is as follows:
```Python
Net = neural_network(train_data,train_labels,test_data,test_labels)
Net.train(epoch, hidden_layers, size_of_layer, batch_size, activation, optimizer, weight_init, learning_rate, weight_decay,loss)
    
```

It can be implemented by passing the following values:

- **hidden_layers**  
    Number of Hidden layers in network
- **size_of_layer**

    Number of neuron in each layer
    
- **batch_size**  
    The Batch Size is passed as an integer that determines the size of the mini batch to be taken into consideration.

- **optimizer**  
    The optimizer value is passed as a string, that is internally converted into an instance of the specified optimizer class. The optimizer classes are present inside the file `optimizers.py`. An instance of the class can be created by passing the corresponding parameters:
    + Normal: step_size   
        (default: step_size=0.01)
    + Momentum: step_size, momentum   
        (default: step_size=1e-3, momentum=0.9)
    + Nesterov: step_size, momentum   
        (default: step_size=1e-3, gamma=0.9)
    + RMSProp: beta, step_size , eps    
        (default: beta=0.9, step_size = 1e-3, eps = 1e-7)
    + Adam: beta1, beta2, step_size, eps   
        (default: beta1=0.9, beta2=0.999, step_size=1e-2, eps=1e-8)
    + Nadam: beta1, beta2, step_size, eps   
        (default: beta1=0.9, beta2=0.999, step_size=1e-3, eps=1e-7)

- **weight_init**: A string - `"random"` or `"Xavier"` can be passed to change the initialization of the weights in the model.

- **epochs**: The number of epochs is passed as an integer to the neural network.

- **loss**: The loss type is passed as a string, that is internally converted into an instance of the specified loss class. "cross_entropy" or "MSE" 



### Training the `NeuralNetwork`
The model can be trained by calling the member function: `forward_propogation`, followed by `backward_propogation`. It is done as follows:

```python
Net.forward_prop()
Net.back_prop()
```

### Testing the `NeuralNetwork`
The model can be tested by calling the `accuracy` member function, with the testing dataset(normalized & scaled) and the expected `test_labels`. The `test_labels` values are only used for calculating the test accuracy. It is done in the following manner:

```python
acc_test= Net.accuracy(test_data_scaled_normalized, test_labels)
```

## Question 7
The confusion matrix is logged using the following code:

```python
    cm = confusion_matrix(y_true, y_pred)
classes = ["T-Shirt/Top","Trouser","Pullover","Dress","Shirts","Sandal","Coat","Sneaker","Bag","Ankle boot"]



# Calculate the percentages
percentages = (cm / np.sum(cm)) * 100

# Define the text for each cell
cell_text = []
for i in range(len(classes)):
    row_text = []
    for j in range(len(classes)):

        txt = "Total "+f'{cm[i, j]}<br>Per. ({percentages[i, j]:.3f})'
        if(i==j):
          txt ="Correcty Predicted " +classes[i]+"<br>"+txt
        if(i!=j):
          txt ="Predicted " +classes[j]+" For "+classes[i]+"<br>"+txt
        row_text.append(txt)
    cell_text.append(row_text)

# Define the trace
trace = go.Heatmap(z=percentages,
                   x=classes,
                   y=classes,
                   colorscale='Blues',
                   colorbar=dict(title='Percentage'),
                   hovertemplate='%{text}%<extra></extra>',
                   text=cell_text,
                   )

# Define the layout
layout = go.Layout(title='Confusion Matrix',
                   xaxis=dict(title='Predicted Classes'),
                   yaxis=dict(title='True Classes'),
                   )

# Plot the figure
fig = go.Figure(data=[trace], layout=layout)
wandb.log({'confusion_matrix': (fig)})
```


## Configuration for train.py
```Python
python train.py -wp 'Assignment1(MSE vs cross)' -we 'Exp2' -e 10 -b 16 -l 'MSE' -o 'nadam' -lr 0.001 -w_d 0.005 -w_i 'Xavier' -nhl 5 -sz 128  -a 'tanh'  
```
---
The codes are organized as follows:

| Question | Location | Function | 
|----------|----------|----------|
| Question 1 | [Question-1](https://github.com/Shivam10816/cs6910_assignment1/blob/main/Q1.ipynb) | Logging Representative Images | 
| Question 2 | [Question-2](https://github.com/Shivam10816/cs6910_assignment1/blob/main/neural_network.py) | Feedforward Architecture |
| Question 3 | [Question-3](https://github.com/Shivam10816/cs6910_assignment1/blob/main/neural_network.py) | Complete Neural Network |
| Question 4 | [Question-4](https://github.com/Shivam10816/cs6910_assignment1/blob/main/neural_network.py) | Hyperparameter sweeps using `wandb` |
| Question 10 | [Question-10](https://github.com/Shivam10816/cs6910_assignment1/blob/main/train.py) | Hyperparameter configurations for MNIST data (Q10) | 
