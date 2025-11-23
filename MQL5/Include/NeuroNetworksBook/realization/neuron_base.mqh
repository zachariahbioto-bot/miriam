//+------------------------------------------------------------------+
//|                                                neuron_base.mqh   |
//|                                   Copyright 2025, Your Name Ltd. |
//|                                      Neural Network Business IP  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Your Name Ltd."
#property link      "https://www.mql5.com"

#include "defines.mqh"
#include "layer_description.mqh"
#include "buffer_type.mqh"
#include <Object.mqh>

//--- Forward declaration for OpenCL manager
class CMyOpenCL;

//+------------------------------------------------------------------+
//| Base class for all neural layers                                 |
//+------------------------------------------------------------------+
class CNeuronBase : public CObject
  {
protected:
   //--- Architecture variables
   int               m_iType;             // Type of neuron layer
   int               m_iNeuronsCount;     // Number of neurons in this layer
   int               m_iInputsCount;      // Number of inputs from previous layer
   int               m_iWindow;           // Input window size (for time-series/conv)
   
   //--- Hyperparameters
   ENUM_ACTIVATION_FUNCTION m_eActivation; // Activation function
   VECTOR            m_adActivationParams; // Parameters for activation function
   ENUM_OPTIMIZATION m_eOptimization;     // Optimization method (Adam, SGD, etc.)

   //--- Buffers (The Synapses and Activation)
   CBufferType* m_cWeights;         // Weights matrix
   CBufferType* m_cBias;            // Bias vector
   CBufferType* m_cInput;           // Input data buffer
   CBufferType* m_cOutput;          // Output data buffer
   
   //--- Gradient and Optimization Buffers
   CBufferType* m_cDelta;           // Error gradient from next layer (delta)
   CBufferType* m_cDeltaWeights;    // Accumulated gradient for weights (dW)
   CBufferType* m_cDeltaBias;       // Accumulated gradient for bias (dB)
   CBufferType* m_cM1;              // 1st moment vector (Adam/Momentum)
   CBufferType* m_cV1;              // 2nd moment vector (Adam/RMSProp)
   
   //--- OpenCL (GPU) context
   CMyOpenCL* m_cOpenCL;

public:
                     CNeuronBase(void);
                    ~CNeuronBase(void);

   //--- Initialization
   virtual bool      Init(CLayerDescription *desc);
   virtual bool      SetInputLayer(CBufferType *input);

   //--- Core Operations (The essential interface for all workers)
   virtual bool      FeedForward(const CBufferType *inputs);
   virtual bool      Backpropagation(CBufferType *target, uint input_count);
   virtual bool      UpdateWeights(uint batch_size = 1);

   //--- OpenCL Control
   virtual bool      SetOpenCL(CMyOpenCL *opencl);

   //--- Accessors
   CBufferType* Output(void) const { return(m_cOutput); }
   CBufferType* Delta(void) const { return(m_cDelta); }

   //--- Persistence
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);
   
   //--- Identification
   virtual int       Type(void) const { return(defNeuronBase); }

protected:
   //--- Internal helpers
   virtual bool      CreateBuffers(void);
   virtual bool      InitOptimizationBuffers(void);
   virtual bool      InitWeights(void);
   virtual bool      CalculateActivation(void);
   virtual bool      CalculateGradient(void);
   
  };
//+------------------------------------------------------------------+
//| Constructor/Destructor                                           |
//+------------------------------------------------------------------+
CNeuronBase::CNeuronBase(void) : m_iType(defNeuronBase), m_iNeuronsCount(0), m_iInputsCount(0), m_iWindow(0),
                                 m_cWeights(NULL), m_cBias(NULL), m_cInput(NULL), m_cOutput(NULL),
                                 m_cDelta(NULL), m_cDeltaWeights(NULL), m_cDeltaBias(NULL), 
                                 m_cM1(NULL), m_cV1(NULL), m_cOpenCL(NULL)
  {
  }

CNeuronBase::~CNeuronBase(void)
  {
   delete m_cWeights; delete m_cBias; delete m_cInput; delete m_cOutput;
   delete m_cDelta; delete m_cDeltaWeights; delete m_cDeltaBias;
   delete m_cM1; delete m_cV1;
  }
//+------------------------------------------------------------------+
