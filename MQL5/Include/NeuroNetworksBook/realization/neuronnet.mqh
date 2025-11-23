//+------------------------------------------------------------------+
//|                                                  neuronnet.mqh   |
//|                                   Copyright 2025, Your Name Ltd. |
//|                                      Neural Network Business IP  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Your Name Ltd."
#property link      "https://www.mql5.com"

#include "defines.mqh"
#include "layer_description.mqh"
#include <Object.mqh>

//--- Forward declarations of classes we will build next
class CArrayLayers;
class CMyOpenCL;
class CPositionEncoder;
class CBufferType;
class CNeuronBase;

//+------------------------------------------------------------------+
//| Neural Network Manager Class                                     |
//+------------------------------------------------------------------+
class CNet : public CObject
  {
protected:
   //--- State Flags
   bool              m_bTrainMode;        // Training mode flag
   bool              m_bOpenCL;           // OpenCL enabled flag
   bool              m_bPositionEncoder;  // Positional encoding flag (for Transformers)

   //--- Internal Objects (The "Organs" of the Net)
   CArrayLayers* m_cLayers;           // Dynamic array storing all neural layers
   CMyOpenCL* m_cOpenCL;           // GPU Context Manager
   CPositionEncoder* m_cPositionEncoder;  // Positional Encoder for Time Series

   //--- Training Hyperparameters
   TYPE              m_dNNLoss;           // Current Loss Value
   int               m_iLossSmoothFactor; // Smoothing factor for loss calculation
   ENUM_LOSS_FUNCTION m_eLossFunction;    // Loss function type (MSE, CrossEntropy, etc.)
   
   //--- Optimization Parameters
   TYPE              m_dLearningRate;     // Learning Rate
   VECTOR            m_adBeta;            // Beta parameters (for Adam optimizer)
   VECTOR            m_adLambda;          // Regularization parameters (L1/L2)

public:
                     CNet(void);
                    ~CNet(void);

   //--- Initialization Methods
   bool              Create(CArrayObj *descriptions);
   bool              Create(CArrayObj *descriptions, TYPE learning_rate, TYPE beta1, TYPE beta2);
   bool              Create(CArrayObj *descriptions, ENUM_LOSS_FUNCTION loss_function, TYPE lambda1, TYPE lambda2);
   bool              Create(CArrayObj *descriptions, TYPE learning_rate, TYPE beta1, TYPE beta2, ENUM_LOSS_FUNCTION loss_function, TYPE lambda1, TYPE lambda2);

   //--- OpenCL (GPU) Control
   void              UseOpenCL(bool value);
   bool              UseOpenCL(void) const { return(m_bOpenCL); }
   bool              InitOpenCL(void);

   //--- Positional Encoding Control (The "Time" Awareness)
   void              UsePositionEncoder(bool value);
   bool              UsePositionEncoder(void) const { return(m_bPositionEncoder); }

   //--- Core Operations (The "Life" of the Net)
   bool              FeedForward(const CBufferType *inputs);    // Forward Pass
   bool              Backpropagation(CBufferType *target);      // Backward Pass (Learning)
   bool              UpdateWeights(uint batch_size = 1);        // Update Synapses
   bool              GetResults(CBufferType *&result);          // Retrieve Predictions

   //--- Training Configuration
   void              SetLearningRates(TYPE learning_rate, TYPE beta1 = defBeta1, TYPE beta2 = defBeta2);
   
   //--- Loss Function Management
   bool              LossFunction(ENUM_LOSS_FUNCTION loss_function, TYPE lambda1 = defLambdaL1, TYPE lambda2 = defLambdaL2);
   ENUM_LOSS_FUNCTION LossFunction(void) const { return(m_eLossFunction); }
   ENUM_LOSS_FUNCTION LossFunction(TYPE &lambda1, TYPE &lambda2);
   
   TYPE              GetRecentAverageLoss(void) const { return(m_dNNLoss); }
   void              LossSmoothFactor(int value) { m_iLossSmoothFactor = value;}
   int               LossSmoothFactor(void) const { return(m_iLossSmoothFactor);}

   //--- State Control
   bool              TrainMode(void) const { return m_bTrainMode; }
   void              TrainMode(bool mode);

   //--- "Memory" / Persistence (Saving and Loading the Brain)
   virtual bool      Save(string file_name = NULL);
   virtual bool      Save(const int file_handle);
   virtual bool      Load(string file_name = NULL, bool common = false);
   virtual bool      Load(const int file_handle);

   //--- Identification
   virtual int       Type(void) const { return(defNeuronNet); }

   //--- Internals Access
   virtual CBufferType* GetGradient(uint layer) const;
   virtual CBufferType* GetWeights(uint layer) const;
   virtual CBufferType* GetDeltaWeights(uint layer) const;
  };
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Saves the Network's Weights (The Brain/Memory) to a file         |
//+------------------------------------------------------------------+
bool CNet::Save(string file_name = NULL)
  {
   string name = (file_name == NULL) ? "default.net" : file_name;
   int handle = FileOpen(name, FILE_WRITE | FILE_BIN);
   if (handle == INVALID_HANDLE)
      return false;

   bool result = Save(handle);
   FileClose(handle);
   return result;
  }
//+------------------------------------------------------------------+
//| Saves the Network's Weights using an open file handle            |
//+------------------------------------------------------------------+
bool CNet::Save(const int file_handle)
  {
   if (file_handle == INVALID_HANDLE)
      return false;
      
   // 1. Write Network Header
   FileWriteInteger(file_handle, Type(), INT_VALUE);
   
   // 2. Write Hyperparameters
   FileWriteDouble(file_handle, m_dLearningRate);
   FileWriteDouble(file_handle, m_adBeta[0]);
   FileWriteDouble(file_handle, m_adBeta[1]);
   FileWriteInteger(file_handle, (int)m_eLossFunction, INT_VALUE);
   
   // 3. Write Layer Information
   if (m_cLayers == NULL)
      return false;
   
   // Delegate the saving process to the CArrayLayers object
   if (!m_cLayers.Save(file_handle))
      return false;
      
   return true;
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Loads the Network's Weights (The Brain/Memory) from a file       |
//+------------------------------------------------------------------+
bool CNet::Load(string file_name = NULL, bool common = false)
  {
   string name = (file_name == NULL) ? "default.net" : file_name;
   int handle = FileOpen(name, FILE_READ | FILE_BIN | (common ? FILE_COMMON : 0));
   if (handle == INVALID_HANDLE)
      return false;

   bool result = Load(handle);
   FileClose(handle);
   return result;
  }
//+------------------------------------------------------------------+
//| Loads the Network's Weights using an open file handle            |
//+------------------------------------------------------------------+
bool CNet::Load(const int file_handle)
  {
   if (file_handle == INVALID_HANDLE)
      return false;
      
   // 1. Read Network Header (Check the file type ID)
   if (FileReadInteger(file_handle, INT_VALUE) != Type())
      return false;
      
   // 2. Read Hyperparameters
   m_dLearningRate = FileReadDouble(file_handle);
   m_adBeta[0] = FileReadDouble(file_handle);
   m_adBeta[1] = FileReadDouble(file_handle);
   m_eLossFunction = (ENUM_LOSS_FUNCTION)FileReadInteger(file_handle, INT_VALUE);
   
   // 3. Delete old layers and create a new container
   if (m_cLayers) delete m_cLayers;
   m_cLayers = new CArrayLayers();
   if (m_cLayers == NULL)
      return false;
      
   // 4. Delegate the loading process to the CArrayLayers object
   // CArrayLayers will use the file handle to determine the type and size of each layer,
   // creating the correct CNeuronBase, CNeuronLSTM, etc., objects using its factory method.
   if (!m_cLayers.Load(file_handle))
      return false;

   // 5. Initialize OpenCL if configured
   if (m_bOpenCL && m_cLayers.SetOpencl(m_cOpenCL))
      return false;
      
   return true;
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Update Synapses (Applies optimization across all layers)         |
//+------------------------------------------------------------------+
bool CNet::UpdateWeights(uint batch_size = 1)
  {
   if(m_cLayers == NULL)
      return false;

   // 1. Iterate through all layers in the network
   for (int i = 0; i < m_cLayers.Total(); i++)
     {
      // Cast the CObject* to CNeuronBase* to access the layer's methods
      CNeuronBase *layer = (CNeuronBase*)m_cLayers.At(i);
      if(layer == NULL)
         return false;

      // 2. Call the layer-specific weight update (where Adam/SGD logic resides)
      if (!layer.UpdateWeights(batch_size))
         return false;
     }

   // 3. Update global training step counter here (needed for Adam optimizer)
   // m_iStep++; // Assuming a global step counter m_iStep exists
   
   return true;
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Initialize the OpenCL (GPU) context and distribute it            |
//+------------------------------------------------------------------+
bool CNet::InitOpenCL(void)
  {
   if(m_bOpenCL) // Check if GPU use is enabled
     {
      // 1. Create the OpenCL Manager object
      m_cOpenCL = new CMyOpenCL();
      if(m_cOpenCL == NULL)
         return false;

      // 2. Initialize the OpenCL context (connect to the physical GPU device)
      if(!m_cOpenCL.Init())
         return false;

      // 3. Inform all layers about the active OpenCL context
      // This allows each layer to create its own GPU memory buffers (CBufferCL)
      if(m_cLayers != NULL)
        {
         if(!m_cLayers.SetOpencl(m_cOpenCL))
            return false;
        }
        
      return true;
     }
   // If OpenCL is not enabled, return true anyway (CPU mode)
   return true;
  }
//+------------------------------------------------------------------+
