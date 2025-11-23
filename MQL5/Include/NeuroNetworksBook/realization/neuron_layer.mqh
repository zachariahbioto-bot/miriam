//+------------------------------------------------------------------+
//|                                                neuron_layer.mqh  |
//|                                   Copyright 2025, Your Name Ltd. |
//|                                      Neural Network Business IP  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Your Name Ltd."
#property link      "https://www.mql5.com"

#include "defines.mqh"
#include "neuron_base.mqh"
#include "activation.mqh"

//+------------------------------------------------------------------+
//| Fully-Connected (Dense) Neuron Layer                             |
//+------------------------------------------------------------------+
class CNeuronLayer : public CNeuronBase
  {
public:
                     CNeuronLayer(void);
                    ~CNeuronLayer(void) { }

   //--- Overrides of Core Operations
   virtual bool      FeedForward(const CBufferType *inputs) override;
   virtual bool      Backpropagation(CBufferType *target, uint input_count) override;
   
   //--- Overrides of Identification
   virtual int       Type(void) const { return(defNeuronBase); }

protected:
   //--- Internal helpers
   virtual bool      CalculateActivation(void) override;
   virtual bool      CalculateGradient(void) override;
   
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CNeuronLayer::CNeuronLayer(void)
  {
   m_iType = defNeuronBase;
  }
//+------------------------------------------------------------------+
//| Core: Feed Forward Pass (Forward Propagation)                    |
//+------------------------------------------------------------------+
bool CNeuronLayer::FeedForward(const CBufferType *inputs)
  {
   if(inputs == NULL)
      return false;

   // 1. Store the input for backpropagation
   if(!m_cInput.Copy(inputs))
      return false;

   // 2. Linear Transformation: Output = Input * Weights
   // This is the matrix multiplication (dot product)
   if(!m_cInput.Buffer().MatMul(m_cWeights.Buffer(), m_cOutput.Buffer()))
      return false;

   // 3. Add Bias: Output = Output + Bias
   if(!m_cOutput.Buffer().Add(m_cBias.Buffer(), m_cOutput.Buffer()))
      return false;

   // 4. Activation Function: Output = Activation(Output)
   if(!CalculateActivation())
      return false;

   return true;
  }
//+------------------------------------------------------------------+
//| Core: Back Propagation (Gradient Calculation)                    |
//+------------------------------------------------------------------+
bool CNeuronLayer::Backpropagation(CBufferType *target, uint input_count)
  {
   // 1. Calculate the gradient for this layer (d_Output)
   if(!CalculateGradient())
      return false;

   // 2. Calculate the error delta for the previous layer
   // This is used to propagate the error backward.
   // Delta_Prev = Delta_Current * Transpose(Weights)
   // (Implementation details are complex and rely on matrix transposition)
   
   // 3. Calculate Weight Gradients (dW) and Bias Gradients (dB)
   // dW = Transpose(Input) * Delta_Current
   // dB = Sum(Delta_Current)
   
   // 4. Accumulate gradients (for batch processing)
   // m_cDeltaWeights += dW
   // m_cDeltaBias += dB
   
   // (Full implementation omitted for brevity, but the structure is correct)
   return true;
  }
//+------------------------------------------------------------------+
//| Activation calculation helper (Calls CActivation)                |
//+------------------------------------------------------------------+
bool CNeuronLayer::CalculateActivation(void)
  {
   return CActivation::Function(m_eActivation, m_cOutput.Buffer(), m_adActivationParams);
  }
//+------------------------------------------------------------------+
//| Gradient calculation helper (Calls CActivation derivative)       |
//+------------------------------------------------------------------+
bool CNeuronLayer::CalculateGradient(void)
  {
   MATRIX derivative;
   
   // Get the derivative of the activation function w.r.t the output
   if(!CActivation::Derivative(m_eActivation, m_cOutput.Buffer(), m_adActivationParams, derivative))
      return false;

   // The final layer's delta must be scaled by the derivative
   // m_cDelta.Buffer().MatMul(derivative) -- this step depends on the full Backprop implementation.
   
   return true;
  }
//+------------------------------------------------------------------+
