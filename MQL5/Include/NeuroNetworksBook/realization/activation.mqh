//+------------------------------------------------------------------+
//|                                                   activation.mqh |
//|                                   Copyright 2025, Your Name Ltd. |
//|                                      Neural Network Business IP  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Your Name Ltd."
#property link      "https://www.mql5.com"

#include "defines.mqh"
#include <Object.mqh>

//--- Activation Function Types
enum ENUM_ACTIVATION_FUNCTION
  {
   AF_LINEAR,       // Identity function
   AF_SIGMOID,      // Classic S-curve (squashes output to [0, 1])
   AF_TANH,         // Hyperbolic tangent (squashes output to [-1, 1])
   AF_RELU,         // Rectified Linear Unit (max(0, x))
   AF_LEAKY_RELU,   // Leaky ReLU
   AF_SOFTMAX,      // For probability distribution in classification
   AF_ELU,          // Exponential Linear Unit
   AF_SELU,         // Scaled Exponential Linear Unit
   AF_GELU,         // Gaussian Error Linear Unit (common in modern Transformers)
   AF_MISH          // Swish-like function
  };

//+------------------------------------------------------------------+
//| Class for calculating Activation Functions and their Derivatives |
//+------------------------------------------------------------------+
class CActivation : public CObject
  {
public:
   CActivation(void) { }
  
   //--- Core methods
   static bool      Function(ENUM_ACTIVATION_FUNCTION func, MATRIX &input, const VECTOR &params);
   static bool      Derivative(ENUM_ACTIVATION_FUNCTION func, const MATRIX &output, const VECTOR &params, MATRIX &d_output);
   
protected:
   //--- Individual Function Implementations (Static helpers)
   static bool      Sigmoid(MATRIX &input);
   static bool      Tanh(MATRIX &input);
   static bool      ReLU(MATRIX &input);
   // ... (Other functions like Softmax, Leaky ReLU, etc. would be defined here)
   
   //--- Individual Derivative Implementations
   static bool      DerivativeSigmoid(const MATRIX &output, MATRIX &d_output);
   static bool      DerivativeTanh(const MATRIX &output, MATRIX &d_output);
   static bool      DerivativeReLU(const MATRIX &output, MATRIX &d_output);
   // ... (Other derivatives would be defined here)
  };
//+------------------------------------------------------------------+
//| Implementation of the ReLU function (Example)                    |
//+------------------------------------------------------------------+
bool CActivation::ReLU(MATRIX &input)
  {
   uint rows = input.Rows();
   uint cols = input.Cols();
   
   for (uint r = 0; r < rows; r++)
     {
      for (uint c = 0; c < cols; c++)
        {
         if (input[r][c] < 0)
           {
            input[r][c] = 0;
           }
        }
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Implementation of the ReLU derivative (Example)                  |
//+------------------------------------------------------------------+
bool CActivation::DerivativeReLU(const MATRIX &output, MATRIX &d_output)
  {
   uint rows = output.Rows();
   uint cols = output.Cols();
   
   d_output.Resize(rows, cols);
   
   for (uint r = 0; r < rows; r++)
     {
      for (uint c = 0; c < cols; c++)
        {
         // Derivative is 1 if output > 0, and 0 if output <= 0
         d_output[r][c] = (output[r][c] > 0) ? (TYPE)1.0 : (TYPE)0.0;
        }
     }
   return true;
  }
//+------------------------------------------------------------------+
// Note: The full implementations of all functions and derivatives
// are extensive and omitted here for brevity, but the structure 
// is correctly defined.
//+------------------------------------------------------------------+
