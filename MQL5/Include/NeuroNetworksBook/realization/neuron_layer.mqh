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
//+------------------------------------------------------------------+
//| Core: Back Propagation (Gradient Calculation)                    |
//+------------------------------------------------------------------+
bool CNeuronLayer::Backpropagation(CBufferType *target, uint input_count)
  {
   // Check if the delta buffer from the next layer is ready.
   if(!m_cDelta)
      return false;

   // 1. Calculate the Derivative of the Activation Function (d_Output)
   // This is used to scale the error signal.
   MATRIX d_activation;
   if(!CActivation::Derivative(m_eActivation, m_cOutput.Buffer(), m_adActivationParams, d_activation))
      return false;

   // 2. Calculate the Error Delta for this layer (d_Output * d_Activation)
   // The error (m_cDelta) is propagated back from the next layer.
   // Delta = Delta_from_Next * Derivative_of_Activation (Element-wise)
   if(!m_cDelta.Buffer().MatMulElementWise(d_activation, m_cDelta.Buffer()))
      return false;

   // 3. Calculate Weight Gradients (dW)
   // dW = Transpose(Input) * Delta
   // Input is CBufferType m_cInput, Delta is CBufferType m_cDelta
   MATRIX input_transposed;
   if(!m_cInput.Buffer().Transpose(input_transposed))
      return false;
      
   // Calculate dW and accumulate it in m_cDeltaWeights (batch accumulation)
   if(!input_transposed.MatMul(m_cDelta.Buffer(), m_cDeltaWeights.Buffer(), true)) // 'true' for accumulation
      return false;
      
   // 4. Calculate Bias Gradients (dB)
   // dB = Sum(Delta) along the rows (over the batch)
   if(!m_cDelta.Buffer().Sum(m_cDeltaBias.Buffer(), true)) // 'true' for accumulation
      return false;

   // 5. Calculate the Delta to Propagate to the Previous Layer (dInput)
   // dInput = Delta * Transpose(Weights)
   // This is stored in m_cDelta and used by the *previous* layer in the chain.
   MATRIX weights_transposed;
   if(!m_cWeights.Buffer().Transpose(weights_transposed))
      return false;
   
   if(!m_cDelta.Buffer().MatMul(weights_transposed, m_cDelta.Buffer()))
      return false;

   return true;
  }
//+------------------------------------------------------------------+
//| Core: Update Weights (Applying the Learning)                     |
//+------------------------------------------------------------------+
bool CNeuronLayer::UpdateWeights(uint batch_size = 1)
  {
   // Note: The implementation of the Adam/SGD optimizer logic is complex.
   // We only structure the method here. It will use m_cDeltaWeights, 
   // m_cDeltaBias, m_cM1, and m_cV1 to calculate the final weight and bias updates.
   
   if(batch_size == 0)
      return false;

   // 1. Normalize Gradients
   // m_cDeltaWeights.Buffer().Div(batch_size);
   // m_cDeltaBias.Buffer().Div(batch_size);

   // 2. Apply Optimization (Adam, SGD, etc.)
   // This step uses the m_eOptimization flag.
   
   // 3. Update Weights and Bias
   // m_cWeights.Buffer() -= final_update_w
   // m_cBias.Buffer() -= final_update_b
   
   // 4. Zero the gradient accumulation buffers for the next batch
   // m_cDeltaWeights.Zero();
   // m_cDeltaBias.Zero();
   
   return true;
  }
//+------------------------------------------------------------------+
