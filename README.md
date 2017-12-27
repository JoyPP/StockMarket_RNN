# StockMarket_RNN

Try to use CNN/RNN/LSTM to process the context, and use the outcome to match with the target labels.

## LSTM Model
One single LSTM layer, combine ten persons' articles as time sequence to process.   
In essence, concatenate all contexts of ten persons and process with LSTM.  

## CNN_LSTM Model
First preprocess each context (matrix) into a vector with CNN, the use LSTM to process and obtain the outcome.  

## GRU_LSTM Model
First preprocess each context (matrix) into a vector with GRU, the use LSTM to process and obtain the outcome.  
