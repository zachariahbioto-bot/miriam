//+------------------------------------------------------------------+
//|                                          layer_description.mqh   |
//|                                   Copyright 2025, Your Name Ltd. |
//|                                      Neural Network Business IP  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, zachariah bioto Ltd."
#property link      "https://www.mql5.com"

#include "defines.mqh"
#include <Object.mqh>

//+------------------------------------------------------------------+
//|  Class for describing the structure of a neural layer            |
//+------------------------------------------------------------------+
class CLayerDescription : public CObject
  {
public:
   CLayerDescription(void);
   ~CLayerDescription(void) {};

   //--- Architecture Parameters
   int                     type;                // Type of neural layer (e.g., defNeuronBase, defNeuronLSTM)
   int                     count;               // Number of neurons in the layer
   int                     window;              // Input data window size
   int                     window_out;          // Output window size (for specific layers like Attention)
   int                     step;                // Input data window step
   int                     layers;              // Number of internal layers (for block architectures)
   int                     batch;               // Batch size for weight updates
   
   //--- Training & Activation Parameters
   ENUM_ACTIVATION_FUNCTION activation;         // Activation function type (e.g., AF_TANH, AF_RELU)
   VECTOR                  activation_params[2];// Parameters for activation functions (if needed)
   ENUM_OPTIMIZATION       optimization;        // Weight optimization method (e.g., Adam, SGD)
   TYPE                    probability;         // Masking probability (used only for Dropout layers)
  };
//+------------------------------------------------------------------+
//| Constructor with default values                                  |
//+------------------------------------------------------------------+
CLayerDescription::CLayerDescription(void) : type(defNeuronBase),
                                             count(100),
                                             window(100),
                                             step(100),
                                             layers(1),
                                             activation(AF_TANH),
                                             optimization(Adam),
                                             probability(0.1),
                                             batch(100)
  {
   // Initialize vector with 2 elements set to default
   activation_params = VECTOR::Ones(2);
   activation_params[1] = 0;
  }
//+------------------------------------------------------------------+
