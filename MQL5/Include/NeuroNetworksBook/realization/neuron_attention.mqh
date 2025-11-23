//+------------------------------------------------------------------+
//|                                           neuron_attention.mqh   |
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
//| Self-Attention Mechanism Layer                                   |
//+------------------------------------------------------------------+
class CNeuronAttention : public CNeuronBase
  {
protected:
   //--- Q, K, V Projection Weights
   CBufferType* m_cWeightQuery;     // Weights for Query (Q) projection
   CBufferType* m_cBiasQuery;       // Bias for Query (Q) projection
   CBufferType* m_cWeightKey;       // Weights for Key (K) projection
   CBufferType* m_cBiasKey;         // Bias for Key (K) projection
   CBufferType* m_cWeightValue;     // Weights for Value (V) projection
   CBufferType* m_cBiasValue;       // Bias for Value (V) projection
   
   //--- Intermediate Buffers
   CBufferType* m_cQuery;           // Q matrix
   CBufferType* m_cKey;             // K matrix
   CBufferType* m_cValue;           // V matrix
   CBufferType* m_cAttentionScores; // Q * K.T (Attention Scores)
   CBufferType* m_cAttentionWeights;// Softmax(Scores) (Attention Weights)

public:
                     CNeuronAttention(void);
                    ~CNeuronAttention(void);

   //--- Initialization
   virtual bool      Init(CLayerDescription *desc) override;
   
   //--- Core Operations (How attention computes focus)
   virtual bool      FeedForward(const CBufferType *inputs) override;
   virtual bool      Backpropagation(CBufferType *target, uint input_count) override;
   
   //--- Identification
   virtual int       Type(void) const { return(defNeuronAttention); }

protected:
   virtual bool      CreateBuffers(void) override;
   virtual bool      InitWeights(void) override;
   
  };
//+------------------------------------------------------------------+
CNeuronAttention::CNeuronAttention(void)
  {
   m_iType = defNeuronAttention;
  }
//+------------------------------------------------------------------+
//| Initialization of the Attention layer                            |
//+------------------------------------------------------------------+
bool CNeuronAttention::Init(CLayerDescription *desc)
  {
   m_iNeuronsCount = desc.count;
   m_iWindow = desc.window; // Crucial for attention over a time sequence

   // 1. Create Q, K, V projection buffers
   m_cWeightQuery = new CBufferType();
   m_cBiasQuery = new CBufferType();
   
   m_cWeightKey = new CBufferType();
   m_cBiasKey = new CBufferType();
   
   m_cWeightValue = new CBufferType();
   m_cBiasValue = new CBufferType();

   // All projection matrices map inputs to the dimensionality of the attention head (m_iNeuronsCount)
   uint projection_dim = m_iNeuronsCount;
   if(!m_cWeightQuery.Create(m_iInputsCount, projection_dim) || !m_cBiasQuery.Create(1, projection_dim)) return false;
   if(!m_cWeightKey.Create(m_iInputsCount, projection_dim) || !m_cBiasKey.Create(1, projection_dim)) return false;
   if(!m_cWeightValue.Create(m_iInputsCount, projection_dim) || !m_cBiasValue.Create(1, projection_dim)) return false;

   // 2. Initialize intermediate buffers (Q, K, V)
   // Size is [Window x Projection_Dim]
   m_cQuery = new CBufferType();
   m_cKey = new CBufferType();
   m_cValue = new CBufferType();
   if(!m_cQuery.Create(m_iWindow, projection_dim) || !m_cKey.Create(m_iWindow, projection_dim) || !m_cValue.Create(m_iWindow, projection_dim)) return false;
   
   // 3. Initialize the weights randomly
   if(!InitWeights()) return false;

   return true;
  }
//+------------------------------------------------------------------+
//| Core Attention Feed Forward Pass                                 |
//+------------------------------------------------------------------+
bool CNeuronAttention::FeedForward(const CBufferType *inputs)
  {
   // 1. Project Input (X) into Q, K, V matrices
   // Q = X * W_Q + B_Q
   // K = X * W_K + B_K
   // V = X * W_V + B_V
   
   // 2. Calculate Attention Scores
   // Scores = (Q * Transpose(K)) / sqrt(d_k)
   
   // 3. Normalize Scores to get Weights
   // Weights = Softmax(Scores)
   
   // 4. Calculate Final Output (The attended, weighted value)
   // Output = Weights * V
   
   // (Full matrix operation logic relies on MQL5 matrix functions and is omitted for brevity)
   return true;
  }
//+------------------------------------------------------------------+
