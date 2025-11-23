//+------------------------------------------------------------------+
//|                                                neuron_lstm.mqh   |
//|                                   Copyright 2025, Your Name Ltd. |
//|                                      Neural Network Business IP  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Your Name Ltd."
#property link      "https://www.mql5.com"

#include "defines.mqh"
#include "neuron_base.mqh"

//--- Forward declaration for CActivation
class CActivation;

//+------------------------------------------------------------------+
//| Long Short-Term Memory (LSTM) Neuron Layer                       |
//+------------------------------------------------------------------+
class CNeuronLSTM : public CNeuronBase
  {
protected:
   //--- Gate Buffers (Internal Memory Components)
   CBufferType* m_cWeightFGate;     // Forget Gate Weights
   CBufferType* m_cBiasFGate;       // Forget Gate Bias
   CBufferType* m_cOutputFGate;     // Forget Gate Output

   CBufferType* m_cWeightIGate;     // Input Gate Weights
   CBufferType* m_cBiasIGate;       // Input Gate Bias
   CBufferType* m_cOutputIGate;     // Input Gate Output

   CBufferType* m_cWeightCGate;     // Candidate Cell Weights
   CBufferType* m_cBiasCGate;       // Candidate Cell Bias
   CBufferType* m_cOutputCGate;     // Candidate Cell Output (Candidate State)

   CBufferType* m_cWeightOGate;     // Output Gate Weights
   CBufferType* m_cBiasOGate;       // Output Gate Bias
   CBufferType* m_cOutputOGate;     // Output Gate Output
   
   //--- State Memory
   CBufferType* m_cCellState;       // The persistent memory cell state (C)
   CBufferType* m_cDeltaCellState;  // Cell State Gradient (dC)

   //--- Previous Step Data (The "Recurrent" part of the memory)
   CBufferType* m_cPrevOutput;      // Output from the previous time step (H_t-1)
   CBufferType* m_cPrevCellState;   // Cell State from the previous time step (C_t-1)
   
public:
                     CNeuronLSTM(void);
                    ~CNeuronLSTM(void);

   //--- Initialization
   virtual bool      Init(CLayerDescription *desc) override;
   virtual bool      SetInputLayer(CBufferType *input) override;
   
   //--- Core Operations (The essential interface for all workers)
   virtual bool      FeedForward(const CBufferType *inputs) override;
   virtual bool      Backpropagation(CBufferType *target, uint input_count) override;
   
   //--- Identification
   virtual int       Type(void) const { return(defNeuronLSTM); }

protected:
   //--- Internal helpers for LSTM
   virtual bool      CreateBuffers(void) override;
   virtual bool      InitOptimizationBuffers(void) override;
   virtual bool      InitWeights(void) override;
   virtual bool      ClearRecurrentState(void);
   
  };
//+------------------------------------------------------------------+
CNeuronLSTM::CNeuronLSTM(void)
  {
   m_iType = defNeuronLSTM;
  }
//+------------------------------------------------------------------+
// Note: Implementation details for Init, FeedForward, and Backpropagation
// are complex and omitted for brevity, but the structure is defined.
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Core LSTM Feed Forward Pass (Memory Computation)                 |
//+------------------------------------------------------------------+
bool CNeuronLSTM::FeedForward(const CBufferType *inputs)
  {
   // Store input and previous output/state (Recurrent Connection)
   // For the first step, m_cPrevOutput and m_cPrevCellState are zero matrices.
   // Input is concatenated with m_cPrevOutput to form the final input vector (H_t-1 + X_t).

   // 1. Calculate Forget Gate (f_t = sigmoid(W_f * [H_t-1, X_t] + B_f))
   // This decides what parts of old memory (C_t-1) to forget.

   // 2. Calculate Input Gate (i_t = sigmoid(W_i * [H_t-1, X_t] + B_i))
   // This decides what new information (C_candidate) to add to the memory.
   
   // 3. Calculate Candidate Cell State (C~_t = tanh(W_c * [H_t-1, X_t] + B_c))
   // This creates the potential new memory content.

   // 4. Update the Main Memory Cell State (C_t)
   // C_t = (f_t * C_t-1) + (i_t * C~_t)
   // This is the core memory operation: "Forget old" + "Store new"

   // 5. Calculate Output Gate (o_t = sigmoid(W_o * [H_t-1, X_t] + B_o))
   // This decides what parts of the memory cell (C_t) to expose as output.

   // 6. Calculate Final Output (H_t)
   // H_t = o_t * tanh(C_t)
   // This output H_t becomes m_cOutput and m_cPrevOutput for the next time step.
   
   return true;
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Core LSTM Feed Forward Pass (Memory Computation)                 |
//+------------------------------------------------------------------+
bool CNeuronLSTM::FeedForward(const CBufferType *inputs)
  {
   // Store input and previous output/state (Recurrent Connection)
   // For the first step, m_cPrevOutput and m_cPrevCellState are zero matrices.
   // Input is concatenated with m_cPrevOutput to form the final input vector (H_t-1 + X_t).

   // 1. Calculate Forget Gate (f_t = sigmoid(W_f * [H_t-1, X_t] + B_f))
   // This decides what parts of old memory (C_t-1) to forget.

   // 2. Calculate Input Gate (i_t = sigmoid(W_i * [H_t-1, X_t] + B_i))
   // This decides what new information (C_candidate) to add to the memory.
   
   // 3. Calculate Candidate Cell State (C~_t = tanh(W_c * [H_t-1, X_t] + B_c))
   // This creates the potential new memory content.

   // 4. Update the Main Memory Cell State (C_t)
   // C_t = (f_t * C_t-1) + (i_t * C~_t)
   // This is the core memory operation: "Forget old" + "Store new"

   // 5. Calculate Output Gate (o_t = sigmoid(W_o * [H_t-1, X_t] + B_o))
   // This decides what parts of the memory cell (C_t) to expose as output.

   // 6. Calculate Final Output (H_t)
   // H_t = o_t * tanh(C_t)
   // This output H_t becomes m_cOutput and m_cPrevOutput for the next time step.
   
   return true;
  }
//+------------------------------------------------------------------+
