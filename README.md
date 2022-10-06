# Stock-Prediction-using-LSTM-algorithm-
Stock Price Forecast using Python's Long Short-Term Memory (LSTM) Model. One form of recurrent neural network used to learn order dependency in sequence prediction issues is called Long Short-Term Memory (LSTM). The LSTM is particularly effective at forecasting stock values because it can store historical data. Firstly, one should put return_sequences = True for multiple LSTM layers so that the output can then be provided to the next layer. Secondly, to improve the accuracy, I consistently added dropout layers after each LSTM layer that would help to reduce overfitting, something which neural networks are highly prone to. In this project I selected stocks of 5 FMCG companies which are ITC, Nestle, Hindustan Unilever, Britannia and Marico.

The dataset is been web scraped from Yahoo Finance from 2015-08-30 to 2022-10-06 7 years of data of each company. Using the past stock prices and applying deep learning algorithm LSTM.

Recurrent neural networks' memory is increased using the long short term memory (LSTM) paradigm. Recurrent neural networks act as a sort of short-term memory by enabling the use of previously determined information in the present neural networks. The earlier data is used for urgent activities. It's possible that we don't have a complete list of the neural node's prior data. LSTMs are a particularly popular type of neural network in RNNs. They should be used to a variety of sequence modelling issues in several application areas, including video, NLP, GIS, and time-series.
The vanishing gradient problem, which arises as a result of the RNN blocks' repetitive usage of the same parameters at each step, is one of the primary problems with RNN.

At each time step, we must attempt to employ various settings to solve this issue.
In such a circumstance, we attempt to strike a balance. While generalising variable-length sequences and maintaining a fixed number of learnable parameters overall, we introduce unique parameters at each step. We provide gated RNN cells, such as the LSTM and GRU.
Internal variables known as Gates are stored in gated cells. Every time step's information, including early stages, determines the value of each gate. The many variables of interest are then multiplied by the gate's value in order to affect them. Data collected in a time-series format over a period of time allows us to track changes over time.Data collected in a time-series format over a period of time allows us to track changes over time. Time-series data may monitor development over a period of seconds, days, or even years. When we first started using time-series data, we thought of it as being more static: the daily highs and lows of the weather, the opening and closing amounts of the stock market. We'll now move on to the coding portion. We will use the stocks dataset to apply LSTM.

1. Dataset

<img width="650" alt="image" src="https://user-images.githubusercontent.com/64062901/194409195-e471ad73-d2ac-432a-afaf-a476faef6d26.png">

<img width="622" alt="image" src="https://user-images.githubusercontent.com/64062901/194409330-06c06aa4-5829-4950-a391-ff83b82d311d.png">

<img width="641" alt="image" src="https://user-images.githubusercontent.com/64062901/194409384-ba936a1e-94be-4cd6-92ec-0295b41c98d5.png">

<img width="687" alt="image" src="https://user-images.githubusercontent.com/64062901/194409438-5ba241d4-2358-4546-9025-5aaf94cc540d.png">

2. Exploring Data

There are 6 columns in the dataset that are related to time series, including the date and various variables like close, high, low, and volume. We will explore with time series using LSTM using opening and closing values.
Here, we have extracted a few features. From the global date variable, we just take the dates. Now that the data is accessible, we can use matplotlib to visualise it and check how our pricing values in the data are being shown. For the price-date graph, the open variable was represented by the colour green, and the close variable was represented by the colour red.

3. Data Pre-Processing

Prior to employing the LSTM to calculate the stock price, we must preprocess this data. Apply the fit transform function to the values in our data to transform them. The data is scaled using the min-max scaler so that all of the price values are on the same scale. Then, we divide the data into training and testing halves, using 80% of the data for each. A function is created so that we can create the sequence for training and testing.

<img width="560" alt="image" src="https://user-images.githubusercontent.com/64062901/194410075-1ce86c7a-fbf7-41c6-8ca3-4e48599bd036.png">

4. Implementation of LSTM

In the next step, we create our LSTM model.  In this article, we will use the Sequential model imported from Keras and required libraries are imported. 
In our model, we employ drop out for regularisation between the two LSTM layers. With a 10% dropout, fifty units are assigned in the LSTM parameter. Mean squared error is the loss function for optimising the problem with adam optimizer. Mean absolute error is the metric used in our LSTM network as it is associated with time-series data.

<img width="494" alt="image" src="https://user-images.githubusercontent.com/64062901/194410594-252bf21d-2327-4d9c-8068-08b4e9705279.png">

<img width="723" alt="image" src="https://user-images.githubusercontent.com/64062901/194410653-d3d8b81c-0603-4009-adfd-777a73325d33.png">


5. Visualization

After our model has been used to fit the data, we utilise it to make predictions. To restore the original value using the altered function, we must utilise inverse transformation. We can now visualise the forecast using this data. 

<img width="734" alt="image" src="https://user-images.githubusercontent.com/64062901/194410999-13e3b095-233a-4143-921f-83519182a0dc.png">

<img width="729" alt="image" src="https://user-images.githubusercontent.com/64062901/194411099-bd11d07e-b04a-4e7f-8d30-68136bbb72fe.png">

<img width="726" alt="image" src="https://user-images.githubusercontent.com/64062901/194411171-7c000a5f-49ac-42c8-9a98-f472e1fab48a.png">

<img width="727" alt="image" src="https://user-images.githubusercontent.com/64062901/194411255-d62d08d6-9b66-4165-a09f-2ea51df02909.png">


